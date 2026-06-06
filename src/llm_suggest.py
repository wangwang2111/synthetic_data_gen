"""
LLM-powered product suggestion using customer profile + transaction history.
Uses Claude with prompt caching on the product catalog (static context).
"""

import json
import anthropic
import pandas as pd

from .schema import PRODUCTS

_CATALOG_TEXT = "\n".join(
    f"- {p['product_id']} | {p['name']} | Category: {p['category']} | "
    f"Min Amount: ${p['min_amount']:,.0f} | Risk: {p['risk_level']} | Premium: {p['is_premium']}"
    for p in PRODUCTS
)

SYSTEM_PROMPT = f"""You are a financial product advisor for a bank. \
Your task is to recommend the most suitable products to a customer \
based on their demographic profile and transaction history.

Available products:
{_CATALOG_TEXT}

Rules:
- Only recommend products NOT already held by the customer.
- Rank by expected fit (highest fit first).
- Give exactly 3 recommendations.
- Respond in JSON with this schema:
  {{"recommendations": [{{"product_id": "...", "name": "...", "reason": "..."}}]}}
"""


def _build_profile(
    customer: pd.Series,
    transactions: pd.DataFrame,
) -> str:
    held = (
        transactions[transactions["customer_id"] == customer["customer_id"]]
        [["product_id", "amount", "channel"]]
        .drop_duplicates("product_id")
        .to_dict("records")
    )
    return json.dumps(
        {
            "customer": {
                "age":            int(customer["age"]),
                "gender":         customer["gender"],
                "income":         round(float(customer["income"]), 2),
                "education":      customer["education"],
                "occupation":     customer["occupation"],
                "marital_status": customer["marital_status"],
                "region":         customer["region"],
                "num_dependents": int(customer["num_dependents"]),
                "credit_score":   int(customer["credit_score"]),
                "tenure_years":   float(customer["tenure_years"]),
            },
            "current_products": held,
        },
        indent=2,
    )


def suggest(
    customer_id: str,
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    model: str = "claude-sonnet-4-6",
) -> dict:
    row = customers_df[customers_df["customer_id"] == customer_id]
    if row.empty:
        raise ValueError(f"Customer {customer_id} not found")

    customer = row.iloc[0]
    profile  = _build_profile(customer, transactions_df)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache static product catalog
            }
        ],
        messages=[{"role": "user", "content": f"Customer profile:\n{profile}"}],
    )

    raw = response.content[0].text.strip()
    # strip markdown fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    result = json.loads(raw)
    result["customer_id"] = customer_id
    result["cache_tokens"] = {
        "input":          response.usage.input_tokens,
        "cache_creation": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read":     getattr(response.usage, "cache_read_input_tokens", 0),
    }
    return result


def batch_suggest(
    customer_ids: list[str],
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    results = []
    for cid in customer_ids:
        try:
            r = suggest(cid, customers_df, transactions_df, model)
            results.append(r)
        except Exception as exc:
            results.append({"customer_id": cid, "error": str(exc)})
    return results
