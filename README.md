# Synthetic Customer & Transaction Data Pipeline

Generate, evaluate, and compare synthetic financial data for customer product recommendation.
Three synthesis strategies are benchmarked head-to-head using both standard SDMetrics reports
and two custom metrics designed for multi-table relational data.

---

## Overview

The pipeline generates a realistic two-table dataset:

```
customers  (demographics: age, income, occupation, credit score, …)
    └──FK──► transactions  (product purchases with date, amount, channel)
```

A static product catalog (15 financial products across Banking, Credit, Insurance, Investment)
is used as a lookup — not synthesized.

The synthetic data feeds a Claude-powered product recommendation engine that suggests
the 3 most suitable products for each customer based on their profile and transaction history.

---

## Methods Compared

| # | Method | Synthesizer | Key property |
|---|---|---|---|
| 1 | **HMA + Gaussian Copula** | SDV default multi-table | Native FK integrity + cardinality |
| 2 | **Independent CTGAN** | CTGANSynthesizer per table | Better within-table distributions |
| 3 | **CTGAN + PAR Hybrid** | CTGAN (customers) + PAR (transactions) | Temporal realism + cross-table context |

---

## Evaluation Metrics

**Standard (SDMetrics)**
- Column Shapes — KSComplement / TVComplement per column
- Column Pair Trends — CorrelationSimilarity / ContingencySimilarity
- Diagnostic — referential integrity, boundary adherence, data structure

**Custom**
- **Cross-table correlation** — Spearman correlation between customer features
  (income, credit score, age) and per-customer product-category mix. Measures whether
  the "high income → investment products" signal is preserved.
- **Temporal realism** — KS test on inter-arrival time distribution + amount
  autocorrelation (lag-1). Measures whether transaction sequences are temporally realistic.

---

## Results (1 000 seed customers)

| Metric | M1 HMA GC | M2 CTGAN | M3 CTGAN+PAR | Winner |
|---|---|---|---|---|
| Overall quality score | 0.826 | **0.871** | 0.537 | M2 |
| Diagnostic / FK integrity | **1.000** | **1.000** | 0.798 | M1 / M2 |
| Cust. column shapes | **0.952** | 0.897 | 0.898 | M1 |
| Cust. pair trends | **0.707** | 0.563 | 0.530 | M1 |
| Txn. column shapes | 0.841 | **0.893** | 0.725 | M2 |
| Cross-table MAD ↓ | 0.259 | 0.288 | **0.235** | M3 |
| IA KS p-value ↑ | 0.000 | **0.003** | 0.000 | M2 |
| Autocorr MAE ↓ | **0.049** | 0.054 | 0.185 | M1 |

**Recommendation:** M1 (HMA + Gaussian Copula) wins for this use case. The LLM recommendation
layer depends on customer demographic correlations (pair trends = 0.707) more than
transaction-level shape fidelity. M2 becomes competitive above ~1 000 seed rows and may
overtake M1 at 2 000+. Drop M3 unless explicit temporal sequence modelling is required.

---

## Project Structure

```
synthetic_data_gen/
├── src/
│   ├── schema.py            # MultiTableMetadata: 3-table (+ 2-table variant)
│   ├── seed_data.py         # Business-rule-driven real data generator
│   ├── generate.py          # HMASynthesizer train + sample helpers
│   ├── methods.py           # All 3 training + generation functions
│   ├── evaluate.py          # Standard SDMetrics quality / diagnostic
│   └── metrics_extended.py  # Cross-table correlation + temporal realism
├── main.py                  # CLI: train / generate / evaluate / suggest / all
├── synthetic_data_pipeline.ipynb  # End-to-end notebook with comparison
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run the full pipeline (train all 3 methods, generate, evaluate)
python main.py all

# LLM product suggestion for a customer (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
python main.py suggest C00001
```

Or open the notebook for the full interactive comparison:

```bash
jupyter lab synthetic_data_pipeline.ipynb
```

---

## CLI Reference

```
python main.py train       # generate seed data, train HMA model
python main.py generate    # sample synthetic data from trained model
python main.py evaluate    # evaluate synthetic vs real
python main.py discover    # SDV auto-detect schema vs hand-crafted comparison
python main.py suggest <id># LLM product suggestion for a customer ID
python main.py all         # train + generate + evaluate
```

---

## Dependencies

- [SDV](https://github.com/sdv-dev/SDV) — multi-table synthesis (HMA, CTGAN, PAR)
- [SDMetrics](https://github.com/sdv-dev/SDMetrics) — quality and diagnostic reports
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude API for LLM suggestions
- pandas, numpy, scikit-learn, scipy, matplotlib
