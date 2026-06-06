"""
Evaluate synthetic data quality using SDMetrics.

Metrics computed:
  - QualityReport  : column shapes (KSComplement / TVComplement)
                     + column pair trends (CorrelationSimilarity / ContingencySimilarity)
  - DiagnosticReport: referential integrity, boundary adherence, coverage
  - New-row synthesis: % rows in synthetic not seen verbatim in real
  - Category coverage: do synthetic categoricals cover real domain?
"""

import json
from pathlib import Path

import pandas as pd
from sdmetrics.reports.multi_table import DiagnosticReport, QualityReport
from .schema import build_metadata

REPORTS_DIR = Path("reports")


def _pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def run_quality_report(
    real_data: dict[str, pd.DataFrame],
    synthetic_data: dict[str, pd.DataFrame],
    verbose: bool = True,
) -> QualityReport:
    meta = build_metadata()
    report = QualityReport()
    report.generate(real_data, synthetic_data, meta.to_dict(), verbose=verbose)
    return report


def run_diagnostic_report(
    real_data: dict[str, pd.DataFrame],
    synthetic_data: dict[str, pd.DataFrame],
    verbose: bool = True,
) -> DiagnosticReport:
    meta = build_metadata()
    report = DiagnosticReport()
    report.generate(real_data, synthetic_data, meta.to_dict(), verbose=verbose)
    return report


def _new_row_rate(real: pd.DataFrame, synth: pd.DataFrame, id_col: str) -> float:
    """Fraction of synthetic rows whose non-ID values don't appear verbatim in real."""
    r = real.drop(columns=[id_col]).astype(str)
    s = synth.drop(columns=[id_col]).astype(str)
    merged = s.merge(r, how="left", indicator=True)
    not_in_real = (merged["_merge"] == "left_only").sum()
    return not_in_real / max(len(s), 1)


def _category_coverage(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    cat_cols = real.select_dtypes(include="object").columns
    result = {}
    for col in cat_cols:
        real_vals  = set(real[col].dropna().unique())
        synth_vals = set(synth[col].dropna().unique())
        coverage   = len(real_vals & synth_vals) / max(len(real_vals), 1)
        result[col] = round(coverage, 4)
    return result


def evaluate(
    real_data: dict[str, pd.DataFrame],
    synthetic_data: dict[str, pd.DataFrame],
    save: bool = True,
) -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  SYNTHETIC DATA QUALITY EVALUATION")
    print("=" * 60)

    # ── Quality Report ────────────────────────────────────────
    print("\n[1/3] Quality Report (column shapes + pair trends) …")
    q_report = run_quality_report(real_data, synthetic_data)
    q_score  = q_report.get_score()
    print(f"      Overall Quality Score : {_pct(q_score)}")

    # get_properties() is global; use get_details() to break down per table
    details_by_table = {}
    for table in real_data:
        shapes_details = q_report.get_details(property_name="Column Shapes")
        pairs_details  = q_report.get_details(property_name="Column Pair Trends")

        shapes_score = shapes_details.loc[shapes_details["Table"] == table, "Score"].mean()
        pairs_score  = pairs_details.loc[pairs_details["Table"] == table,  "Score"].mean()

        details_by_table[table] = {
            "column_shapes":      round(float(shapes_score), 4) if not pd.isna(shapes_score) else 0.0,
            "column_pair_trends": round(float(pairs_score),  4) if not pd.isna(pairs_score)  else 0.0,
        }
        print(f"      [{table}] shapes={_pct(details_by_table[table]['column_shapes'])}  "
              f"pair_trends={_pct(details_by_table[table]['column_pair_trends'])}")

    # ── Diagnostic Report ─────────────────────────────────────
    print("\n[2/3] Diagnostic Report (integrity + coverage + boundaries) …")
    d_report = run_diagnostic_report(real_data, synthetic_data)
    d_score  = d_report.get_score()
    print(f"      Overall Diagnostic Score: {_pct(d_score)}")

    # ── Custom metrics ────────────────────────────────────────
    print("\n[3/3] Custom metrics …")
    custom = {}
    table_id_map = [("products", "product_id"), ("customers", "customer_id"),
                    ("transactions", "transaction_id")]
    for table, id_col in [(t, c) for t, c in table_id_map if t in real_data]:
        real_df  = real_data[table]
        synth_df = synthetic_data[table]

        new_row = _new_row_rate(real_df, synth_df, id_col)
        cov     = _category_coverage(real_df, synth_df)
        avg_cov = sum(cov.values()) / max(len(cov), 1)

        print(f"      [{table}]  new-row rate={_pct(new_row)}  avg-category-coverage={_pct(avg_cov)}")
        custom[table] = {
            "new_row_rate":           round(new_row, 4),
            "category_coverage":      cov,
            "avg_category_coverage":  round(avg_cov, 4),
        }

    # ── Summary ───────────────────────────────────────────────
    summary = {
        "quality_score":    round(q_score, 4),
        "diagnostic_score": round(d_score, 4),
        "table_details":    details_by_table,
        "custom_metrics":   custom,
    }

    if save:
        path = REPORTS_DIR / "evaluation_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n      Report saved → {path}")

    print("\n" + "=" * 60 + "\n")
    return summary
