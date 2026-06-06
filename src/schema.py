"""
Multi-table schema definition for SDV.

Tables
------
products     (root)      ──FK──> transactions
customers    (root)      ──FK──> transactions
transactions (child of both)

Referential integrity is enforced via two HMA relationships so that every
synthetic transaction references a valid customer AND a valid product.
"""

from sdv.metadata import MultiTableMetadata

# Static product catalog — used by seed_data.py for generation and by
# llm_suggest.py to build the LLM prompt context.
PRODUCTS = [
    {"product_id": f"P{i:03d}", "name": name, "category": cat, "min_amount": price,
     "risk_level": risk, "is_premium": premium}
    for i, (name, cat, price, risk, premium) in enumerate([
        ("Basic Savings Account",     "Banking",      0.0,   "Low",    False),
        ("Premium Savings Account",   "Banking",      0.0,   "Low",    True),
        ("Personal Loan",             "Credit",     500.0,   "Medium", False),
        ("Home Loan",                 "Credit",   50000.0,   "Medium", False),
        ("Auto Loan",                 "Credit",   10000.0,   "Medium", False),
        ("Credit Card Basic",         "Credit",       0.0,   "Medium", False),
        ("Credit Card Premium",       "Credit",       0.0,   "Medium", True),
        ("Term Life Insurance",       "Insurance",   50.0,   "Low",    False),
        ("Health Insurance Basic",    "Insurance",   80.0,   "Low",    False),
        ("Health Insurance Premium",  "Insurance",  200.0,   "Low",    True),
        ("Investment Fund A",         "Investment", 1000.0,  "Medium", False),
        ("Investment Fund B",         "Investment", 5000.0,  "High",   True),
        ("Fixed Deposit 1Y",          "Banking",   1000.0,   "Low",    False),
        ("Fixed Deposit 3Y",          "Banking",   2000.0,   "Low",    False),
        ("Mutual Fund Equity",        "Investment",  500.0,  "High",   False),
    ], start=1)
]

PRODUCTS_DF_COLS = ["product_id", "name", "category", "min_amount", "risk_level", "is_premium"]

PRODUCT_CATEGORIES = ["Banking", "Credit", "Insurance", "Investment"]

OCCUPATIONS    = ["Engineer", "Doctor", "Teacher", "Lawyer", "Business Owner",
                  "Retail Worker", "Nurse", "Accountant", "Manager", "Student", "Retired"]
EDUCATION      = ["High School", "Associate", "Bachelor", "Master", "PhD"]
MARITAL_STATUS = ["Single", "Married", "Divorced", "Widowed"]
REGIONS        = ["North", "South", "East", "West", "Central"]
CHANNELS       = ["Branch", "Online", "Mobile App", "Phone"]
TXN_STATUS     = ["Completed", "Pending", "Cancelled"]


def build_metadata() -> MultiTableMetadata:
    meta = MultiTableMetadata()

    # ── products (root) ───────────────────────────────────────
    meta.add_table("products")
    meta.add_column("products", "product_id",   sdtype="id", regex_format="P[0-9]{3}")
    meta.add_column("products", "name",         sdtype="categorical")
    meta.add_column("products", "category",     sdtype="categorical")
    meta.add_column("products", "min_amount",   sdtype="numerical", computer_representation="Float")
    meta.add_column("products", "risk_level",   sdtype="categorical")
    meta.add_column("products", "is_premium",   sdtype="boolean")
    meta.set_primary_key("products", "product_id")

    # ── customers (root) ──────────────────────────────────────
    meta.add_table("customers")
    meta.add_column("customers", "customer_id",    sdtype="id",        regex_format="C[0-9]{5}")
    meta.add_column("customers", "age",            sdtype="numerical", computer_representation="Int64")
    meta.add_column("customers", "gender",         sdtype="categorical")
    meta.add_column("customers", "income",         sdtype="numerical", computer_representation="Float")
    meta.add_column("customers", "education",      sdtype="categorical")
    meta.add_column("customers", "occupation",     sdtype="categorical")
    meta.add_column("customers", "marital_status", sdtype="categorical")
    meta.add_column("customers", "region",         sdtype="categorical")
    meta.add_column("customers", "num_dependents", sdtype="numerical", computer_representation="Int64")
    meta.add_column("customers", "credit_score",   sdtype="numerical", computer_representation="Int64")
    meta.add_column("customers", "tenure_years",   sdtype="numerical", computer_representation="Float")
    meta.add_column("customers", "is_churned",     sdtype="boolean")
    meta.set_primary_key("customers", "customer_id")

    # ── transactions (child of customers + products) ──────────
    meta.add_table("transactions")
    meta.add_column("transactions", "transaction_id",   sdtype="id",        regex_format="T[0-9]{6}")
    meta.add_column("transactions", "customer_id",      sdtype="id",        regex_format="C[0-9]{5}")
    meta.add_column("transactions", "product_id",       sdtype="id",        regex_format="P[0-9]{3}")
    meta.add_column("transactions", "product_category", sdtype="categorical")  # denormalized for analytics
    meta.add_column("transactions", "amount",           sdtype="numerical", computer_representation="Float")
    meta.add_column("transactions", "transaction_date", sdtype="datetime",  datetime_format="%Y-%m-%d")
    meta.add_column("transactions", "channel",          sdtype="categorical")
    meta.add_column("transactions", "status",           sdtype="categorical")
    meta.add_column("transactions", "is_first_product", sdtype="boolean")
    meta.set_primary_key("transactions", "transaction_id")

    # referential integrity: customers → transactions
    meta.add_relationship(
        parent_table_name="customers",
        parent_primary_key="customer_id",
        child_table_name="transactions",
        child_foreign_key="customer_id",
    )
    # referential integrity: products → transactions
    meta.add_relationship(
        parent_table_name="products",
        parent_primary_key="product_id",
        child_table_name="transactions",
        child_foreign_key="product_id",
    )

    return meta


def build_metadata_2table() -> MultiTableMetadata:
    """
    Two-table schema for Method 1 & 2: customers → transactions.
    products is treated as a static lookup, not a synthesized root table.
    product_id in transactions is categorical (not FK to products).
    """
    meta = MultiTableMetadata()

    meta.add_table("customers")
    meta.add_column("customers", "customer_id",    sdtype="id",        regex_format="C[0-9]{5}")
    meta.add_column("customers", "age",            sdtype="numerical", computer_representation="Int64")
    meta.add_column("customers", "gender",         sdtype="categorical")
    meta.add_column("customers", "income",         sdtype="numerical", computer_representation="Float")
    meta.add_column("customers", "education",      sdtype="categorical")
    meta.add_column("customers", "occupation",     sdtype="categorical")
    meta.add_column("customers", "marital_status", sdtype="categorical")
    meta.add_column("customers", "region",         sdtype="categorical")
    meta.add_column("customers", "num_dependents", sdtype="numerical", computer_representation="Int64")
    meta.add_column("customers", "credit_score",   sdtype="numerical", computer_representation="Int64")
    meta.add_column("customers", "tenure_years",   sdtype="numerical", computer_representation="Float")
    meta.add_column("customers", "is_churned",     sdtype="boolean")
    meta.set_primary_key("customers", "customer_id")

    meta.add_table("transactions")
    meta.add_column("transactions", "transaction_id",   sdtype="id",          regex_format="T[0-9]{6}")
    meta.add_column("transactions", "customer_id",      sdtype="id",          regex_format="C[0-9]{5}")
    meta.add_column("transactions", "product_id",       sdtype="categorical")  # lookup, not FK
    meta.add_column("transactions", "product_category", sdtype="categorical")
    meta.add_column("transactions", "amount",           sdtype="numerical",   computer_representation="Float")
    meta.add_column("transactions", "transaction_date", sdtype="datetime",    datetime_format="%Y-%m-%d")
    meta.add_column("transactions", "channel",          sdtype="categorical")
    meta.add_column("transactions", "status",           sdtype="categorical")
    meta.add_column("transactions", "is_first_product", sdtype="boolean")
    meta.set_primary_key("transactions", "transaction_id")

    meta.add_relationship(
        parent_table_name="customers",
        parent_primary_key="customer_id",
        child_table_name="transactions",
        child_foreign_key="customer_id",
    )
    return meta
