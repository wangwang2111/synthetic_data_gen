"""
Extended evaluation metrics beyond SDMetrics' standard reports.

cross_table_correlation  – Spearman correlation between customer features
                           (income, credit_score, age) and per-customer
                           product-category mix.  Measures whether the
                           income→investment-product relationship is preserved.

temporal_stats           – Inter-arrival time distribution + amount
                           autocorrelation (lag-1).  Measures whether
                           transaction sequences are temporally realistic.

compare_methods          – Runs all metrics across multiple synthetic datasets
                           and returns a tidy comparison DataFrame.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
from sdmetrics.reports.multi_table import QualityReport, DiagnosticReport

from .schema import build_metadata_2table

FEATURE_COLS   = ["income", "credit_score", "age"]
CATEGORY_COLS  = ["Banking", "Credit", "Insurance", "Investment"]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-table correlation
# ─────────────────────────────────────────────────────────────────────────────

def _customer_category_pcts(customers: pd.DataFrame,
                             transactions: pd.DataFrame) -> pd.DataFrame:
    cat = (transactions.groupby(["customer_id", "product_category"])
                       .size().unstack(fill_value=0))
    for col in CATEGORY_COLS:
        if col not in cat.columns:
            cat[col] = 0
    cat = cat[CATEGORY_COLS]
    cat_pct = cat.div(cat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return customers.set_index("customer_id")[FEATURE_COLS].join(cat_pct, how="inner")


def cross_table_correlation(customers: pd.DataFrame,
                             transactions: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation matrix: features × product categories."""
    merged = _customer_category_pcts(customers, transactions)
    rows = []
    for feat in FEATURE_COLS:
        for cat in CATEGORY_COLS:
            corr, pval = spearmanr(merged[feat], merged[cat])
            rows.append({"feature": feat, "category": cat,
                         "spearman_r": round(corr, 4), "p_value": round(pval, 4)})
    return pd.DataFrame(rows)


def cross_table_score(real_customers, real_transactions,
                      syn_customers, syn_transactions) -> dict:
    """
    Mean absolute difference of Spearman correlations between real and synthetic.
    Lower = better.  Also returns the per-pair delta for inspection.
    """
    real_corr = cross_table_correlation(real_customers, real_transactions)
    syn_corr  = cross_table_correlation(syn_customers,  syn_transactions)
    merged    = real_corr.merge(syn_corr, on=["feature","category"],
                                suffixes=("_real","_syn"))
    merged["delta"] = (merged["spearman_r_real"] - merged["spearman_r_syn"]).abs()
    return {
        "mean_abs_delta": round(merged["delta"].mean(), 4),
        "max_abs_delta":  round(merged["delta"].max(),  4),
        "detail":         merged,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Temporal realism
# ─────────────────────────────────────────────────────────────────────────────

def temporal_stats(transactions: pd.DataFrame) -> dict:
    """
    Inter-arrival times (days) and amount autocorrelation per customer.
    Requires transaction_date column parseable as datetime.
    """
    df = transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values(["customer_id", "transaction_date"])
    df["prev_date"] = df.groupby("customer_id")["transaction_date"].shift(1)
    df["inter_arrival"] = (df["transaction_date"] - df["prev_date"]).dt.days
    ia = df["inter_arrival"].dropna()

    # Per-customer lag-1 autocorrelation of amounts
    def _autocorr(x):
        return x.autocorr(lag=1) if len(x) > 2 else np.nan

    autocorr_vals = (df.groupby("customer_id")["amount"]
                       .apply(_autocorr)
                       .dropna())

    return {
        "inter_arrival_mean":   round(float(ia.mean()),   2),
        "inter_arrival_median": round(float(ia.median()), 2),
        "inter_arrival_std":    round(float(ia.std()),    2),
        "amount_autocorr_mean": round(float(autocorr_vals.mean()), 4),
        "inter_arrival_values": ia.values,        # kept for KS test
        "autocorr_values":      autocorr_vals.values,
    }


def temporal_score(real_transactions: pd.DataFrame,
                   syn_transactions:  pd.DataFrame) -> dict:
    """
    KS test on inter-arrival distribution + autocorrelation MAE.
    Higher KS p-value = more similar distributions (better).
    Lower autocorr_mae = better.
    """
    real_stats = temporal_stats(real_transactions)
    syn_stats  = temporal_stats(syn_transactions)

    ks_stat, ks_pval = ks_2samp(real_stats["inter_arrival_values"],
                                  syn_stats["inter_arrival_values"])
    autocorr_mae = abs(real_stats["amount_autocorr_mean"] -
                       syn_stats["amount_autocorr_mean"])

    return {
        "ia_ks_statistic":  round(ks_stat,      4),
        "ia_ks_pvalue":     round(ks_pval,      4),   # higher = more similar
        "autocorr_mae":     round(autocorr_mae, 4),   # lower  = better
        "syn_ia_mean":      syn_stats["inter_arrival_mean"],
        "syn_ia_median":    syn_stats["inter_arrival_median"],
        "syn_ia_std":       syn_stats["inter_arrival_std"],
        "syn_autocorr":     syn_stats["amount_autocorr_mean"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standard SDMetrics quality / diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def _sdmetrics_scores(real_data: dict, syn_data: dict) -> dict:
    meta = build_metadata_2table()
    meta_dict = meta.to_dict()

    q = QualityReport()
    q.generate(real_data, syn_data, meta_dict, verbose=False)
    q_score = q.get_score()

    d = DiagnosticReport()
    d.generate(real_data, syn_data, meta_dict, verbose=False)
    d_score = d.get_score()

    shapes = q.get_details("Column Shapes")
    pairs  = q.get_details("Column Pair Trends")

    cust_shapes = shapes.loc[shapes["Table"] == "customers", "Score"].mean()
    cust_pairs  = pairs.loc[pairs["Table"]  == "customers", "Score"].mean()
    txn_shapes  = shapes.loc[shapes["Table"] == "transactions", "Score"].mean()

    return {
        "quality_score":        round(q_score, 4),
        "diagnostic_score":     round(d_score, 4),
        "cust_column_shapes":   round(float(cust_shapes), 4),
        "cust_pair_trends":     round(float(cust_pairs),  4),
        "txn_column_shapes":    round(float(txn_shapes),  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compare all methods
# ─────────────────────────────────────────────────────────────────────────────

def compare_methods(real_data: dict,
                    synthetic_datasets: dict[str, dict]) -> pd.DataFrame:
    """
    real_data           : {"customers": df, "transactions": df}
    synthetic_datasets  : {"Method 1 – HMA GC": {...}, "Method 2 – CTGAN": {...}, ...}
    Returns a tidy DataFrame with all metrics as columns, methods as rows.
    """
    rows = []
    for method_name, syn in synthetic_datasets.items():
        print(f"  Evaluating: {method_name} …")
        row = {"method": method_name}

        # Standard metrics
        try:
            std = _sdmetrics_scores(real_data, syn)
            row.update(std)
        except Exception as e:
            print(f"    ⚠ SDMetrics error: {e}")

        # Cross-table correlation
        try:
            ct = cross_table_score(
                real_data["customers"], real_data["transactions"],
                syn["customers"],       syn["transactions"],
            )
            row["cross_table_mad"]     = ct["mean_abs_delta"]
            row["cross_table_max_err"] = ct["max_abs_delta"]
        except Exception as e:
            print(f"    ⚠ Cross-table error: {e}")

        # Temporal realism
        try:
            ts = temporal_score(real_data["transactions"], syn["transactions"])
            row["ia_ks_pvalue"]   = ts["ia_ks_pvalue"]
            row["ia_ks_stat"]     = ts["ia_ks_statistic"]
            row["autocorr_mae"]   = ts["autocorr_mae"]
            row["syn_ia_mean"]    = ts["syn_ia_mean"]
        except Exception as e:
            print(f"    ⚠ Temporal error: {e}")

        rows.append(row)

    df = pd.DataFrame(rows).set_index("method")
    return df
