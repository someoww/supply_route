import argparse
import json
import os
import pickle

import pandas as pd


def load_bundle(model_path: str) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    required_keys = {"model", "feature_columns", "medians", "task"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle missing keys: {missing}")

    return bundle


def prepare_frame(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    medians = bundle["medians"]

    X = df.reindex(columns=feature_columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(medians)
    return X


def _make_records(bundle: dict, X: pd.DataFrame, proba: bool) -> list:
    model = bundle["model"]
    task = bundle["task"]

    predictions = model.predict(X).tolist()
    records = []

    if task == "classification" and proba and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = getattr(model, "classes_", [])

        for i, pred in enumerate(predictions):
            records.append(
                {
                    "prediction": pred,
                    "probabilities": {
                        str(cls): float(prob)
                        for cls, prob in zip(classes, probabilities[i])
                    },
                }
            )
    else:
        for pred in predictions:
            records.append({"prediction": pred})

    return records


def predict_csv(bundle: dict, input_path: str, proba: bool) -> list:
    df = pd.read_csv(input_path)
    X = prepare_frame(df, bundle)
    return _make_records(bundle, X, proba)


def predict_json_payload(bundle: dict, payload, proba: bool) -> list:
    if isinstance(payload, dict) and "instances" in payload:
        payload = payload["instances"]

    if isinstance(payload, dict):
        payload = [payload]

    df = pd.json_normalize(payload)
    X = prepare_frame(df, bundle)
    return _make_records(bundle, X, proba)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict supply-route condition from saved model."
    )
    parser.add_argument("--model", type=str, default="model.bin")
    parser.add_argument("--csv", type=str, help="Input CSV file path")
    parser.add_argument(
        "--json",
        dest="json_input",
        type=str,
        help="JSON string or JSON file path"
    )
    parser.add_argument("--out", type=str, help="Output file path")
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Include probabilities for classification models"
    )
    args = parser.parse_args()

    bundle = load_bundle(args.model)

    if args.csv:
        records = predict_csv(bundle, args.csv, args.proba)
    elif args.json_input:
        if os.path.isfile(args.json_input):
            with open(args.json_input, "r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = json.loads(args.json_input)

        records = predict_json_payload(bundle, payload, args.proba)
    else:
        parser.error("Provide either --csv or --json")

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if args.out.endswith(".csv"):
            pd.DataFrame(
                {"prediction": [r["prediction"] for r in records]}
            ).to_csv(args.out, index=False)
        else:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

        print(f"Saved predictions to {args.out}")
    else:
        limit = min(20, len(records))
        print(json.dumps(records[:limit], indent=2))
        if len(records) > limit:
            print(f"... {len(records) - limit} more records not printed")