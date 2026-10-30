"""Evaluate saved models on a labeled CSV without retraining."""
import argparse
import json
import joblib
import pandas as pd
from src.core import ROOT, MODEL_FILES, read_test_csv, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(ROOT / "test_data.csv"))
    parser.add_argument("--model", choices=["all", *MODEL_FILES], default="all")
    parser.add_argument("--output", help="Optional output metrics CSV")
    args = parser.parse_args()
    _, X, y = read_test_csv(args.csv)
    names = MODEL_FILES if args.model == "all" else [args.model]
    rows = []
    for name in names:
        model = joblib.load(ROOT / "model" / MODEL_FILES[name])
        result = evaluate(model, X, y)
        rows.append({"Model": name, **result["metrics"]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if args.output:
        table.to_csv(args.output, index=False)
