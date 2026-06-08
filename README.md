# Synthetic Customer & Transaction Data Pipeline

Benchmarks three synthetic data generation strategies on a realistic financial dataset.
The goal is to find which strategy best preserves the demographic-to-product signals that
an LLM recommendation engine depends on.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Seed Data Generation                          │
│  seed_data.py — 500 customers × ~2 000 transactions                 │
│  Business rules: income→product eligibility, age→channel preference  │
│  credit score→product tier, occupation→income distribution           │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  real_*.csv
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
┌─────────────────────┐  ┌──────────────────┐  ┌────────────────────────┐
│  Method 1           │  │  Method 2        │  │  Method 3              │
│  HMA + Gaussian     │  │  Independent     │  │  CTGAN + PAR Hybrid    │
│  Copula             │  │  CTGAN           │  │                        │
│                     │  │                  │  │  CTGAN → customers     │
│  MultiTableMetadata │  │  SingleTable     │  │  PAR   → transactions  │
│  (hand-crafted FK   │  │  metadata per    │  │  conditioned on income │
│   relationship)     │  │  table (auto-    │  │  + credit score        │
│  HMASynthesizer     │  │  detect + patch) │  │  NN match restores FK  │
└──────────┬──────────┘  └────────┬─────────┘  └───────────┬────────────┘
           │                      │                         │
           └──────────────────────┼─────────────────────────┘
                                  │  synthetic data (1 000 customers)
                       ┌──────────▼──────────────┐
                       │       Evaluation         │
                       │  SDMetrics Quality       │
                       │  SDMetrics Diagnostic    │
                       │  Cross-table Correlation │
                       │  Temporal Realism        │
                       └──────────┬───────────────┘
                                  │
                       ┌──────────▼───────────────┐
                       │   LLM Recommendation      │
                       │   DeepSeek (default)      │
                       │   or Claude (fallback)    │
                       │   → 3 products/customer   │
                       └───────────────────────────┘
```

---

## Pipeline Walkthrough

### Step 1 — Seed data generation (`src/seed_data.py`)

`make_seed_data(n_customers=500)` generates the "real" training data that SDV learns from.
Each customer is built from scratch using explicit business rules:

- **Occupation** determines income distribution (e.g. Doctor: μ=$150k, σ=$40k; Student: μ=$20k, σ=$8k).
- **Income + education** jointly set credit score: `base = 500 + income/1000`, then add an
  education boost (0–80 pts) plus Gaussian noise, clipped to [300, 850].
- **Product eligibility** is rule-gated: credit card premium requires score ≥ 720, home loan
  requires income ≥ $60k and age ≥ 25, investment funds require income ≥ $80k, etc.
- **Channel preference** is age-stratified: under-30s skew Mobile App (60%), over-50s skew
  Branch (40%).
- **Transaction count** is drawn from Poisson(λ=4) per customer, minimum 1.
- **Churn rate** is fixed at 12% (random draw, independent of other features).

The result is a dataset with non-trivial correlations baked in — income predicts product
category, credit score predicts product tier, age predicts channel — which are exactly the
signals the evaluation metrics and the LLM recommendation layer depend on.

### Step 2 — Schema definition (`src/schema.py`)

Two metadata objects are defined by hand rather than relying on SDV auto-detection:

- `build_metadata()` — full 3-table schema: `products` and `customers` are parent tables,
  `transactions` is a child of both via FK relationships.
- `build_metadata_2table()` — used by M1 and M2: `products` is treated as a static lookup,
  so `product_id` in transactions is typed as `categorical` rather than as a FK. Only the
  `customers → transactions` relationship is registered.

Each column's `sdtype`, format, and representation are specified explicitly (e.g.
`datetime_format="%Y-%m-%d"`, `computer_representation="Int64"` for integer numerics,
`regex_format="C[0-9]{5}"` for IDs).

### Step 3 — Training (`src/methods.py`)

Three strategies are trained independently on the same seed data.

**Method 1 — HMA + Gaussian Copula**

```python
meta  = build_metadata_2table()       # hand-crafted relational schema
synth = HMASynthesizer(meta)
synth.fit({"customers": ..., "transactions": ...})
```

HMASynthesizer learns a hierarchical model: it fits a Gaussian Copula on the customers
table, then fits an "extension" model that captures how many transactions each customer
has (cardinality) and how the transaction columns relate to parent customer columns.
All FK integrity is managed internally by SDV.

**Method 2 — Independent CTGAN**

```python
c_meta = SingleTableMetadata()
c_meta.detect_from_dataframe(c_df)   # auto-infer types
c_meta.update_column("customer_id", sdtype="id")
c_meta.set_primary_key("customer_id")
# same pattern for transactions
ctgan_c.fit(c_df)
ctgan_t.fit(t_df_no_fk)              # customer_id dropped before training
```

Each table is trained in isolation. At generation time, cardinality is restored by
resampling from the empirical transaction-count distribution of the real data
(`np.random.choice(real_counts, size=n_customers, replace=True)`), and `customer_id`
is re-attached by position. Cross-table correlations (income → product category) are
**not explicitly modelled**.

**Method 3 — CTGAN + PAR Hybrid**

```python
# Customer side: same as M2
ctgan_c.fit(c_df)

# Transaction side: augment with context columns before training
t_aug = t_df.merge(c_df[["customer_id", "income", "credit_score"]], on="customer_id")
par_meta.set_sequence_key("customer_id")
par_meta.set_sequence_index("transaction_date")
par = PARSynthesizer(par_meta, context_columns=["income", "credit_score"])
par.fit(t_aug)
```

PAR (Probabilistic AutoRegressive model) treats each customer's transactions as a
time-ordered sequence, conditioned on the customer's income and credit score. This lets
the model learn "high-income customers buy investment products" as a temporal pattern.

At generation time, PAR generates sequences with its own sampled context values.
A nearest-neighbour match on `(income, credit_score)` reassigns each sequence to the
closest synthetic customer, restoring FK consistency.

### Step 4 — Generation

Each method generates 1 000 synthetic customers and their transactions (scale factor = 2×
relative to the 500-row seed). Generation is deterministic given a fixed seed.

### Step 5 — Evaluation (`src/evaluate.py`, `src/metrics_extended.py`)

Four metric groups are computed for each method against the real data:

1. **Column Shapes** — per-column marginal distribution similarity (KSComplement for
   numericals, TVComplement for categoricals). Scored 0–1, higher is better.
2. **Column Pair Trends** — pairwise correlation similarity (CorrelationSimilarity for
   numeric pairs, ContingencySimilarity for categorical). Scored 0–1, higher is better.
3. **Diagnostic** — referential integrity (FK violations), boundary adherence (values in
   range), data structure validity. Scored 0–1.
4. **Cross-table correlation (custom)** — Spearman ρ between `{income, credit_score, age}`
   and per-customer product-category mix (Banking/Credit/Insurance/Investment share).
   Reported as mean absolute delta vs. real Spearman matrix. Lower is better.
5. **Temporal realism (custom)** — KS test on inter-arrival time distribution between
   consecutive transactions per customer + mean absolute error on per-customer lag-1
   amount autocorrelation. KS p-value higher is better; autocorr MAE lower is better.

### Step 6 — LLM recommendation (`src/llm_suggest.py`)

`suggest(customer_id, customers_df, transactions_df)` builds a JSON profile of the
customer (demographics + current product holdings) and sends it to an LLM with a
system prompt containing the full 15-product catalog.

The LLM returns exactly 3 recommendations (products not already held, ranked by fit).
Two backends are supported:

- **DeepSeek** (`deepseek-chat`) — default when `DEEPSEEK_API_KEY` is set. Uses the
  OpenAI-compatible API.
- **Claude** (Anthropic) — fallback. Uses prompt caching on the product catalog system
  prompt to reduce repeated token costs across batch calls.

---

## Data Model

```
products  (static lookup — 15 rows, not synthesized)
  product_id PK | name | category | min_amount | risk_level | is_premium

customers  (root, synthesized — 500 seed → 1 000 synthetic)
  customer_id PK | age | gender | income | education | occupation
  marital_status | region | num_dependents | credit_score
  tenure_years | is_churned

transactions  (child of customers — ~2 000 seed → ~4 000 synthetic)
  transaction_id PK | customer_id FK | product_id (categorical in M1/M2)
  product_category | amount | transaction_date | channel | status
  is_first_product
```

Product catalog spans 4 categories (Banking, Credit, Insurance, Investment) across
15 SKUs, price-anchored from $0 (savings accounts) to $50 000 (home loan minimum).

---

## Methods Compared

| # | Method | Synthesizer | Metadata source |
|---|---|---|---|
| 1 | **HMA + Gaussian Copula** | `HMASynthesizer` | Hand-crafted `MultiTableMetadata` |
| 2 | **Independent CTGAN** | `CTGANSynthesizer` × 2 | `detect_from_dataframe` + patches |
| 3 | **CTGAN + PAR Hybrid** | `CTGANSynthesizer` + `PARSynthesizer` | detect + patches + sequence keys |

---

## Evaluation Metrics

| Metric | Source | Direction | What it measures |
|---|---|---|---|
| Overall quality score | SDMetrics | ↑ | Weighted average of shapes + pair trends |
| Diagnostic / FK integrity | SDMetrics | ↑ | Referential integrity and boundary validity |
| Customer column shapes | SDMetrics | ↑ | Marginal distribution fidelity per column |
| Customer pair trends | SDMetrics | ↑ | Pairwise correlation preservation |
| Transaction column shapes | SDMetrics | ↑ | Marginal fidelity for transaction columns |
| Cross-table MAD | Custom | ↓ | Income/age/credit → product-category signal |
| Inter-arrival KS p-value | Custom | ↑ | Transaction timing distribution similarity |
| Autocorrelation MAE | Custom | ↓ | Sequential amount pattern fidelity |

---

## Results

### Summary table (1 000 seed customers)

| Metric | M1 HMA GC | M2 CTGAN | M3 CTGAN+PAR | Winner |
|---|---|---|---|---|
| Overall quality score | 0.826 | **0.871** | 0.537 | M2 |
| Diagnostic / FK integrity | **1.000** | **1.000** | 0.798 | M1 / M2 |
| Customer column shapes | **0.952** | 0.897 | 0.898 | M1 |
| Customer pair trends | **0.707** | 0.563 | 0.530 | M1 |
| Transaction column shapes | 0.841 | **0.893** | 0.725 | M2 |
| Cross-table MAD ↓ | 0.259 | 0.288 | **0.235** | M3 |
| Inter-arrival KS p-value ↑ | 0.000 | **0.003** | 0.000 | M2 |
| Autocorrelation MAE ↓ | **0.049** | 0.054 | 0.185 | M1 |

### Findings

**M1 (HMA + Gaussian Copula) is the best choice for the LLM recommendation use case.**
The recommendation engine conditions on customer demographics and product history.
M1's customer pair trends score (0.707) — the metric most directly tied to
"does income predict product category?" — is 26% higher than M2 (0.563) and 33% higher
than M3 (0.530). The Gaussian Copula captures joint demographics well at 500 seed rows.

**M2 (Independent CTGAN) wins on marginal distributions but loses correlations.**
Overall quality 0.871 is the highest, driven by better transaction column shapes (0.893).
However, because the two tables are trained independently, cross-table correlations are
not preserved — pair trends drop to 0.563. At larger seed sizes (est. 2 000+), CTGAN
typically converges to tighter marginals and may close the pair-trends gap with M1.

**M3 (CTGAN + PAR Hybrid) underperforms at this data scale.**
Despite the RNN-based PAR model being designed for temporal realism, autocorrelation MAE
is actually the worst (0.185 vs M1's 0.049). PAR needs substantially more sequences
than 500 to learn transaction timing reliably. FK integrity also drops to 79.8% because
the nearest-neighbour FK reassignment can map multiple PAR sequences to the same synthetic
customer, creating orphaned transaction groups. M3's only win is cross-table MAD (0.235),
suggesting the context conditioning does marginally preserve the income→product signal,
but not enough to offset its other regressions.

**Temporal modelling is the hardest metric to satisfy at small scale.**
All three methods produce near-zero KS p-values for inter-arrival times, meaning none
reproduces the real transaction timing distribution well. This is primarily a data-size
problem: with ~4 transactions per customer on average, there are too few inter-arrival
intervals per sequence to learn a distribution.

---

## Design Decisions

**Hand-crafted metadata over SDV auto-detection.**
`python main.py discover` shows that SDV auto-detection misidentifies boolean columns
as categorical, fails to infer datetime formats, and cannot discover FK relationships
without user input. The hand-crafted `schema.py` ensures every column type, format,
and relationship is correct before training begins.

**Products treated as a static lookup, not synthesized.**
The 3-table `build_metadata()` exists but is not used in training. If the products table
were included as a synthesized root, SDV would generate new product rows with nonsensical
combinations of `name`, `category`, and `min_amount`. The catalog is a fixed business
artifact; synthesizing it provides no value and would break product-ID referential
integrity in downstream recommendations.

**Two-table schema for M1 and M2.**
With `product_id` typed as `categorical` (not FK) in transactions, M1/M2 treat product
selection as a distributional column rather than a relational join. This is the correct
framing: the 15 valid product IDs are learned from the marginal distribution of
`transactions.product_id`, not enforced by a parent-table constraint.

**Spearman over Pearson for cross-table correlation.**
The income→product-category relationships are monotonic but not linear (e.g. investment
products appear only above $80k income with a hard cutoff, not a gradual linear increase).
Spearman ρ captures monotonic rank correlation without assuming linearity.

**Nearest-neighbour FK reassignment in M3.**
PAR generates transaction sequences tagged with its own internal `customer_id` values
that don't match the synthetic customer table. NN on `(income, credit_score)` is the
simplest semantically meaningful remapping — it ensures each PAR sequence is assigned
to the synthetic customer with the most similar context features. The cost is that
uniqueness is not guaranteed (multiple sequences can map to one customer).

---

## Limitations

**Small seed size constrains GAN quality.** CTGAN (M2, M3) is sensitive to training set
size. At 500 rows, the generator often fails to fully capture multimodal distributions.
The quality gap between M1 and M2 on pair trends is expected to narrow significantly
above 2 000 seed rows.

**PAR requires more sequences than available.** PARSynthesizer is designed for datasets
with long per-entity sequences (100+ steps). With ~4 transactions per customer on average,
the RNN cannot learn meaningful temporal patterns, explaining M3's poor autocorrelation
MAE and near-zero inter-arrival KS p-value.

**No train/test split on seed data.** All 500 seed customers are used for training. The
evaluation compares synthetic data against the same seed data used for fitting, so
reported scores are optimistic upper bounds. A held-out evaluation set would give a
less biased picture, at the cost of further reducing the training set.

**FK integrity is not guaranteed in M3.** The NN reassignment step can map multiple PAR
sequences to the same synthetic customer (many-to-one), creating implicit FK violations
at the transaction level. The 79.8% diagnostic score reflects this. A bijective
assignment (e.g. Hungarian algorithm) would fix uniqueness but is O(n²) and was
not implemented given the scale.

**Temporal independence of transactions.** In the seed data, transaction dates are
drawn uniformly at random per customer (`_random_date`), so there is no actual temporal
autocorrelation in the real data to learn. Measuring autocorrelation MAE against a
dataset with effectively i.i.d. dates means all three methods are evaluated against a
near-zero target — the metric is meaningful only if seed data uses a real temporal
process.

**LLM recommendations are not evaluated.** The suggestion layer is qualitatively
validated but not benchmarked. There is no ground-truth held-out set to measure
recommendation precision or recall against.

---

## Assumptions

- 500 seed customers is sufficient for HMA + Gaussian Copula (a parametric model) to
  capture the joint distribution. Empirically confirmed by M1's pair trends score (0.707).
- The business rules in `seed_data.py` (income brackets, credit-score gates, channel
  age-weights) are representative enough of real bank data to make the synthetic data
  useful for downstream LLM recommendation testing.
- A Poisson(λ=4) transaction count per customer is a reasonable proxy for a retail bank's
  product cross-sell rate over a 4-year window.
- A 12% churn rate (independent of demographics) is a conservative baseline; in practice
  churn is correlated with tenure and product holdings, but this correlation is not encoded.
- The 15-product catalog and its eligibility rules (credit card premium at score ≥ 720,
  home loan at income ≥ $60k + age ≥ 25, etc.) are stable business logic, not subject
  to synthesis.

---

## Project Structure

```
synthetic_data_gen/
├── src/
│   ├── schema.py            # MultiTableMetadata: 3-table (full) + 2-table (M1/M2)
│   ├── seed_data.py         # Business-rule-driven real data generator
│   ├── generate.py          # HMASynthesizer train + sample helpers (legacy M1 path)
│   ├── methods.py           # Train + generate functions for all 3 methods
│   ├── evaluate.py          # SDMetrics quality / diagnostic wrapper
│   ├── metrics_extended.py  # Cross-table correlation + temporal realism metrics
│   └── llm_suggest.py       # LLM product recommendation (DeepSeek / Claude)
├── main.py                  # CLI entry point
├── data/                    # Generated CSVs and pickled models
├── reports/                 # SDMetrics HTML reports + auto-detected metadata
├── synthetic_data_pipeline.ipynb  # End-to-end notebook with comparison dashboard
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate seed data, train all 3 methods, generate synthetic data, evaluate
python main.py all

# LLM product suggestion for a customer (requires DEEPSEEK_API_KEY)
export DEEPSEEK_API_KEY=sk-...
python main.py suggest C00001
```

Open the notebook for the full interactive comparison with charts:

```bash
jupyter lab synthetic_data_pipeline.ipynb
```

---

## CLI Reference

```
python main.py train              # generate seed data + train HMA model (M1)
python main.py generate           # sample synthetic data from trained HMA model
python main.py evaluate           # evaluate synthetic vs real (SDMetrics + custom)
python main.py discover           # SDV auto-detect schema vs hand-crafted comparison
python main.py suggest <id>       # LLM product recommendation for a customer ID
python main.py all                # train + generate + evaluate (M1 only via CLI)
```

For M2 and M3 training, use the notebook or call `src/methods.py` functions directly.

---

## Dependencies

- [SDV](https://github.com/sdv-dev/SDV) — multi-table synthesis (HMA, CTGAN, PAR)
- [SDMetrics](https://github.com/sdv-dev/SDMetrics) — quality and diagnostic reports
- [DeepSeek API](https://platform.deepseek.com) / [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — LLM recommendations
- pandas, numpy, scikit-learn, scipy, matplotlib
