"""
Entry point.

Usage:
  python main.py train           # generate seed data, train HMA model
  python main.py generate        # sample synthetic data from trained model
  python main.py evaluate        # evaluate synthetic vs real
  python main.py discover        # auto-detect schema using SDV from saved CSVs
  python main.py suggest C00001  # LLM product suggestion for a customer
  python main.py all             # train + generate + evaluate
"""

import sys
from pathlib import Path

DATA_DIR = Path("data")


def cmd_train():
    from src.generate import train

    synth, real_data = train(n_seed_customers=500, save=True)
    DATA_DIR.mkdir(exist_ok=True)
    for name, df in real_data.items():
        df.to_csv(DATA_DIR / f"real_{name}.csv", index=False)
    print("[main] Training complete.")


def cmd_generate():
    from src.generate import generate
    synthetic = generate(n_customers=1000, save=True)
    print(f"[main] Generated {len(synthetic['customers']):,} customers, "
          f"{len(synthetic['transactions']):,} transactions.")


def cmd_evaluate():
    import pandas as pd
    from src.evaluate import evaluate

    real_data = {
        "products":     pd.read_csv(DATA_DIR / "real_products.csv"),
        "customers":    pd.read_csv(DATA_DIR / "real_customers.csv"),
        "transactions": pd.read_csv(DATA_DIR / "real_transactions.csv",
                                    parse_dates=["transaction_date"]),
    }
    synthetic_data = {
        "products":     pd.read_csv(DATA_DIR / "synthetic_products.csv"),
        "customers":    pd.read_csv(DATA_DIR / "synthetic_customers.csv"),
        "transactions": pd.read_csv(DATA_DIR / "synthetic_transactions.csv",
                                    parse_dates=["transaction_date"]),
    }
    evaluate(real_data, synthetic_data, save=True)


def cmd_suggest(customer_id: str):
    import pandas as pd
    from src.llm_suggest import suggest

    customers_df    = pd.read_csv(DATA_DIR / "synthetic_customers.csv")
    transactions_df = pd.read_csv(DATA_DIR / "synthetic_transactions.csv")

    result = suggest(customer_id, customers_df, transactions_df)
    print(f"\nRecommendations for {customer_id}:")
    for i, rec in enumerate(result.get("recommendations", []), 1):
        print(f"  {i}. [{rec['product_id']}] {rec['name']}")
        print(f"     Reason: {rec['reason']}")
    print(f"\nToken usage: {result.get('cache_tokens', {})}")


def cmd_discover():
    """
    Use SDV metadata auto-detection on the real CSVs, then compare to the
    hand-crafted schema.  Useful for validating that SDV agrees with our
    column-type assignments and for onboarding new tables.
    """
    import json
    import pandas as pd
    from sdv.metadata import MultiTableMetadata

    tables = {
        "products":     pd.read_csv(DATA_DIR / "real_products.csv"),
        "customers":    pd.read_csv(DATA_DIR / "real_customers.csv"),
        "transactions": pd.read_csv(DATA_DIR / "real_transactions.csv"),
    }

    print("\n── SDV Auto-Detected Metadata ───────────────────────────────────")
    auto_meta = MultiTableMetadata()
    auto_meta.detect_from_dataframes(tables)

    # print detected column types per table
    meta_dict = auto_meta.to_dict()
    for table_name, table_info in meta_dict.get("tables", {}).items():
        print(f"\n  [{table_name}]")
        for col, props in table_info.get("columns", {}).items():
            print(f"    {col:<25} {props}")

    # save for reference
    out = Path("reports/auto_detected_metadata.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"\n  Saved → {out}")


def cmd_all():
    cmd_train()
    cmd_generate()
    cmd_evaluate()


COMMANDS = {
    "train":    cmd_train,
    "generate": cmd_generate,
    "evaluate": cmd_evaluate,
    "discover": cmd_discover,
    "all":      cmd_all,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    if cmd == "suggest":
        if len(args) < 2:
            print("Usage: python main.py suggest <customer_id>")
            sys.exit(1)
        cmd_suggest(args[1])
    elif cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
