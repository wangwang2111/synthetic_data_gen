"""
Three synthesis strategies for comparison.

Method 1 – HMA + Gaussian Copula (baseline)
    SDV's default multi-table approach.  Captures cardinality (txns-per-customer)
    and basic cross-table structure via the extension model.

Method 2 – Independent CTGAN
    CTGANSynthesizer trained separately on customers and transactions.
    CTGAN captures non-Gaussian within-table distributions better than GC.
    Cross-table cardinality is re-sampled from the real distribution.
    Cross-table correlations (income → product) are NOT explicitly modelled.

Method 3 – CTGAN + PAR hybrid
    CTGANSynthesizer for customers.
    PARSynthesizer (RNN-based) for transactions, trained with customer income and
    credit_score as context_columns so it learns "high-income → investment products"
    and temporal sequence patterns simultaneously.
    FK join done by nearest-neighbour on context features.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sdv.multi_table import HMASynthesizer
    from sdv.single_table import CTGANSynthesizer
    from sdv.sequential import PARSynthesizer
    from sdv.metadata import SingleTableMetadata

from .schema import build_metadata_2table, PRODUCTS

PRODUCT_LOOKUP = {p["product_id"]: p for p in PRODUCTS}
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Method 1: HMA + Gaussian Copula
# ─────────────────────────────────────────────────────────────────────────────

def train_hma_gc(real_data: dict, save: bool = True):
    data_2t = {k: v for k, v in real_data.items() if k in ("customers", "transactions")}
    meta    = build_metadata_2table()
    synth   = HMASynthesizer(meta)
    synth.fit(data_2t)
    if save:
        with open(DATA_DIR / "m1_hma_gc.pkl", "wb") as f:
            pickle.dump(synth, f)
    return synth


def generate_hma_gc(synth, n_customers: int = 1000, seed_size: int = 500) -> dict:
    scale     = n_customers / seed_size
    synthetic = synth.sample(scale=scale)
    return {"customers": synthetic["customers"], "transactions": synthetic["transactions"]}


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Independent CTGAN per table
# ─────────────────────────────────────────────────────────────────────────────

def _build_single_meta(df: pd.DataFrame, id_col: str,
                       datetime_cols: list | None = None) -> SingleTableMetadata:
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    meta.update_column(id_col, sdtype="id")
    meta.set_primary_key(id_col)
    for col in (datetime_cols or []):
        meta.update_column(col, sdtype="datetime", datetime_format="%Y-%m-%d")
    for col in df.select_dtypes("bool").columns:
        meta.update_column(col, sdtype="boolean")
    return meta


def train_ctgan(real_data: dict, epochs: int = 300, save: bool = True):
    c_df = real_data["customers"]
    t_df = real_data["transactions"].drop(columns=["customer_id"])

    c_meta = _build_single_meta(c_df, "customer_id")
    t_meta = _build_single_meta(t_df, "transaction_id",
                                datetime_cols=["transaction_date"])

    ctgan_c = CTGANSynthesizer(c_meta, epochs=epochs, verbose=False)
    ctgan_c.fit(c_df)

    ctgan_t = CTGANSynthesizer(t_meta, epochs=epochs, verbose=False)
    ctgan_t.fit(t_df)

    models = {"ctgan_customers": ctgan_c, "ctgan_transactions": ctgan_t}
    if save:
        with open(DATA_DIR / "m2_ctgan.pkl", "wb") as f:
            pickle.dump(models, f)
    return models


def generate_ctgan(models: dict, real_transactions: pd.DataFrame,
                   n_customers: int = 1000) -> dict:
    ctgan_c = models["ctgan_customers"]
    ctgan_t = models["ctgan_transactions"]

    syn_customers = ctgan_c.sample(n_customers)

    # Resample cardinality from real distribution
    real_counts = real_transactions.groupby("customer_id").size().values
    syn_counts  = np.random.choice(real_counts, size=n_customers, replace=True)

    txn_rows = []
    for i, (cid, n_txn) in enumerate(
            zip(syn_customers["customer_id"], syn_counts)):
        chunk = ctgan_t.sample(int(n_txn)).copy()
        chunk["customer_id"] = cid
        # rebuild transaction_ids
        chunk["transaction_id"] = [f"T{i:04d}{j:04d}" for j in range(len(chunk))]
        txn_rows.append(chunk)

    syn_transactions = pd.concat(txn_rows, ignore_index=True)
    return {"customers": syn_customers, "transactions": syn_transactions}


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: CTGAN for customers + PAR for transactions (with context)
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_COLS = ["income", "credit_score"]   # customer features PAR conditions on

def _build_par_meta(df: pd.DataFrame) -> SingleTableMetadata:
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    meta.update_column("customer_id",      sdtype="id")
    meta.update_column("transaction_date", sdtype="datetime", datetime_format="%Y-%m-%d")
    for col in df.select_dtypes("bool").columns:
        meta.update_column(col, sdtype="boolean")
    meta.set_sequence_key("customer_id")
    meta.set_sequence_index("transaction_date")
    return meta


def train_hybrid(real_data: dict, ctgan_epochs: int = 300,
                 par_epochs: int = 128, save: bool = True):
    c_df = real_data["customers"]
    t_df = real_data["transactions"]

    # CTGAN on customers
    c_meta   = _build_single_meta(c_df, "customer_id")
    ctgan_c  = CTGANSynthesizer(c_meta, epochs=ctgan_epochs, verbose=False)
    ctgan_c.fit(c_df)

    # PAR on transactions augmented with customer context columns
    t_aug = t_df.merge(c_df[["customer_id"] + CONTEXT_COLS], on="customer_id")
    t_aug = t_aug.drop(columns=["transaction_id"])   # PAR regenerates IDs
    t_aug["transaction_date"] = pd.to_datetime(t_aug["transaction_date"])

    par_meta = _build_par_meta(t_aug)
    par = PARSynthesizer(
        par_meta,
        context_columns=CONTEXT_COLS,
        epochs=par_epochs,
        cuda=False,
        verbose=False,
    )
    par.fit(t_aug)

    models = {"ctgan_customers": ctgan_c, "par_transactions": par}
    if save:
        with open(DATA_DIR / "m3_hybrid.pkl", "wb") as f:
            pickle.dump(models, f)
    return models


def generate_hybrid(models: dict, n_customers: int = 1000) -> dict:
    ctgan_c = models["ctgan_customers"]
    par     = models["par_transactions"]

    syn_customers = ctgan_c.sample(n_customers)

    # PAR.sample() generates sequences with its own sampled context.
    # We then match each PAR sequence to the nearest synthetic customer
    # by (income, credit_score) so FK consistency is meaningful.
    syn_txn_raw = par.sample(num_sequences=n_customers)

    # Extract PAR-generated context for each sequence
    par_ctx = (syn_txn_raw.groupby("customer_id")[CONTEXT_COLS]
                           .first().reset_index())

    # Nearest-neighbour match: par sequence → synthetic customer
    cust_ctx  = syn_customers[["customer_id"] + CONTEXT_COLS].copy()
    nn        = NearestNeighbors(n_neighbors=1)
    nn.fit(cust_ctx[CONTEXT_COLS].values)
    _, indices = nn.kneighbors(par_ctx[CONTEXT_COLS].values)
    matched_cids = cust_ctx.iloc[indices.flatten()]["customer_id"].values

    par_id_to_syn_id = dict(zip(par_ctx["customer_id"], matched_cids))
    syn_txn_raw["customer_id"] = syn_txn_raw["customer_id"].map(par_id_to_syn_id)

    # Drop context columns that came from customers (not part of transactions schema)
    txn_cols = ["customer_id", "transaction_date", "product_id",
                "product_category", "amount", "channel", "status", "is_first_product"]
    available = [c for c in txn_cols if c in syn_txn_raw.columns]
    syn_transactions = syn_txn_raw[available].copy()
    syn_transactions.insert(0, "transaction_id",
                            [f"T{i:08d}" for i in range(len(syn_transactions))])

    return {"customers": syn_customers, "transactions": syn_transactions}
