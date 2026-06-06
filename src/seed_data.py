"""
Generate realistic seed (real) data that SDV will learn from.
Encodes business rules:
  - Higher income / credit score → premium and investment products
  - Younger customers → mobile channel preference
  - Older customers → insurance / fixed deposits
"""

import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

from .schema import PRODUCTS, PRODUCTS_DF_COLS, OCCUPATIONS, EDUCATION, MARITAL_STATUS, REGIONS, CHANNELS, TXN_STATUS

random.seed(42)
np.random.seed(42)

PRODUCT_LOOKUP = {p["product_id"]: p for p in PRODUCTS}

# income ranges by occupation
INCOME_PARAMS = {
    "Engineer":       (90_000,  25_000),
    "Doctor":        (150_000,  40_000),
    "Teacher":        (55_000,  12_000),
    "Lawyer":        (120_000,  35_000),
    "Business Owner":(110_000,  50_000),
    "Retail Worker":  (35_000,   8_000),
    "Nurse":          (65_000,  15_000),
    "Accountant":     (80_000,  20_000),
    "Manager":       (100_000,  30_000),
    "Student":        (20_000,   8_000),
    "Retired":        (45_000,  15_000),
}

def _income(occ: str) -> float:
    mu, sigma = INCOME_PARAMS[occ]
    return max(15_000, round(np.random.normal(mu, sigma), 2))

def _credit_score(income: float, edu: str) -> int:
    base = 500 + income / 1_000
    edu_boost = {"High School": 0, "Associate": 20, "Bachelor": 40, "Master": 60, "PhD": 80}
    raw = base + edu_boost.get(edu, 0) + np.random.normal(0, 30)
    return int(np.clip(raw, 300, 850))

def _eligible_products(income: float, age: int, credit_score: int) -> list[str]:
    eligible = ["P001", "P002"]  # basic/premium savings always available
    if credit_score >= 650:
        eligible += ["P006"]     # credit card basic
    if credit_score >= 720:
        eligible += ["P007"]     # credit card premium
    if income >= 30_000 and credit_score >= 600:
        eligible += ["P003"]     # personal loan
    if income >= 60_000 and credit_score >= 680 and age >= 25:
        eligible += ["P004"]     # home loan
    if income >= 40_000 and credit_score >= 640:
        eligible += ["P005"]     # auto loan
    if age >= 25:
        eligible += ["P008"]     # term life
    if income >= 50_000:
        eligible += ["P009", "P010"]  # health insurance
    if income >= 80_000:
        eligible += ["P011", "P012", "P015"]  # investment funds
    if income >= 40_000:
        eligible += ["P013", "P014"]  # fixed deposits
    return list(set(eligible))

def _channel(age: int) -> str:
    if age < 30:
        return random.choices(CHANNELS, weights=[5, 25, 60, 10])[0]
    elif age < 50:
        return random.choices(CHANNELS, weights=[20, 40, 30, 10])[0]
    else:
        return random.choices(CHANNELS, weights=[40, 30, 10, 20])[0]

def _txn_amount(product_id: str) -> float:
    base = PRODUCT_LOOKUP[product_id]["min_amount"]
    if base == 0.0:
        return round(random.uniform(0, 50), 2)   # fee-free products
    noise = np.random.normal(1.0, 0.15)
    return round(max(1.0, base * noise), 2)

def _random_date(start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()

START_DATE = date(2022, 1, 1)
END_DATE   = date(2025, 12, 31)


def make_seed_data(n_customers: int = 500) -> dict[str, pd.DataFrame]:
    customers, transactions = [], []

    for i in range(1, n_customers + 1):
        cid      = f"C{i:05d}"
        age      = int(np.clip(np.random.normal(42, 14), 18, 80))
        gender   = random.choice(["Male", "Female", "Non-binary"])
        edu      = random.choices(EDUCATION, weights=[15, 10, 40, 25, 10])[0]
        occ      = random.choice(OCCUPATIONS)
        income   = _income(occ)
        marital  = random.choices(MARITAL_STATUS, weights=[30, 50, 12, 8])[0]
        region   = random.choice(REGIONS)
        deps     = int(np.clip(np.random.poisson(1.2), 0, 6))
        credit   = _credit_score(income, edu)
        tenure   = round(random.uniform(0.1, 20.0), 1)
        churned  = random.random() < 0.12

        customers.append({
            "customer_id":    cid,
            "age":            age,
            "gender":         gender,
            "income":         income,
            "education":      edu,
            "occupation":     occ,
            "marital_status": marital,
            "region":         region,
            "num_dependents": deps,
            "credit_score":   credit,
            "tenure_years":   tenure,
            "is_churned":     churned,
        })

        eligible = _eligible_products(income, age, credit)
        n_txns   = max(1, int(np.random.poisson(4)))
        seen     = set()
        for j in range(n_txns):
            pid    = random.choice(eligible)
            prod   = PRODUCT_LOOKUP[pid]
            tid    = f"T{(i * 100 + j):06d}"
            txdate = _random_date(START_DATE, END_DATE)
            channel = _channel(age)
            status  = random.choices(TXN_STATUS, weights=[88, 8, 4])[0]

            transactions.append({
                "transaction_id":   tid,
                "customer_id":      cid,
                "product_id":       pid,
                "product_category": prod["category"],
                "amount":           _txn_amount(pid),
                "transaction_date": txdate,
                "channel":          channel,
                "status":           status,
                "is_first_product": pid not in seen,
            })
            seen.add(pid)

    df_customers    = pd.DataFrame(customers)
    df_transactions = pd.DataFrame(transactions)
    df_transactions["transaction_date"] = pd.to_datetime(df_transactions["transaction_date"])

    df_products = pd.DataFrame(PRODUCTS)[PRODUCTS_DF_COLS]

    return {"products": df_products, "customers": df_customers, "transactions": df_transactions}
