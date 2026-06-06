"""
Train an HMASynthesizer on seed data and generate synthetic records.
HMA (Hierarchical Modeling Algorithm) preserves referential integrity
across products → transactions ← customers.
"""

import pickle
from pathlib import Path

import pandas as pd
from sdv.multi_table import HMASynthesizer

from .schema import build_metadata
from .seed_data import make_seed_data

MODEL_PATH = Path("data/hma_model.pkl")
DATA_DIR   = Path("data")


def train(n_seed_customers: int = 500, save: bool = True) -> HMASynthesizer:
    print(f"[generate] Building {n_seed_customers}-customer seed dataset …")
    real_data = make_seed_data(n_seed_customers)
    meta      = build_metadata()

    print("[generate] Training HMASynthesizer …")
    synth = HMASynthesizer(meta)
    synth.fit(real_data)

    if save:
        DATA_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(synth, f)
        print(f"[generate] Model saved → {MODEL_PATH}")

    return synth, real_data


def load_model() -> HMASynthesizer:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def generate(
    n_customers: int = 1000,
    synth: HMASynthesizer | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    if synth is None:
        synth = load_model()

    print(f"[generate] Sampling {n_customers} synthetic customers …")
    synthetic = synth.sample(scale=n_customers / 500)   # scale relative to seed size

    if save:
        DATA_DIR.mkdir(exist_ok=True)
        for table, df in synthetic.items():
            path = DATA_DIR / f"synthetic_{table}.csv"
            df.to_csv(path, index=False)
            print(f"[generate] {table}: {len(df):,} rows → {path}")

    return synthetic


def get_real_data(n_seed_customers: int = 500) -> dict[str, pd.DataFrame]:
    """Return (or regenerate) the seed real data used for evaluation."""
    real_path = DATA_DIR / "real_customers.csv"
    if real_path.exists() and (DATA_DIR / "real_products.csv").exists():
        return {
            "products":     pd.read_csv(DATA_DIR / "real_products.csv"),
            "customers":    pd.read_csv(DATA_DIR / "real_customers.csv"),
            "transactions": pd.read_csv(DATA_DIR / "real_transactions.csv",
                                        parse_dates=["transaction_date"]),
        }
    real = make_seed_data(n_seed_customers)
    for name, df in real.items():
        df.to_csv(DATA_DIR / f"real_{name}.csv", index=False)
    return real
