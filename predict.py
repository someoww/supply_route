"""Score routes with a saved bundle. Input: CSV path, JSON file, or inline JSON (object, list, {"instances": [...]})."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from train import LABELS


def load(path):
    return pickle.loads(Path(path).read_bytes())  # pickle runs code on load: open only bundles you trained


def read(src):
    if src.endswith(".csv"):
        return pd.read_csv(src)
    data = json.loads(Path(src).read_text() if Path(src).is_file() else src)
    data = data.get("instances", [data]) if isinstance(data, dict) else data
    return pd.json_normalize(data)


def predict(bundle, df):
    cols = bundle["features"]
    absent = [c for c in cols if c not in df]
    if len(absent) == len(cols):
        sys.exit("none of the model's features are in the input")
    if absent:  # the model treats absent inputs as unknown, and says less about them than you might hope
        print(f"warning: {len(absent)}/{len(cols)} features missing, treated as unknown: {absent[:5]}...", file=sys.stderr)
    X = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    model = bundle["model"]
    out = pd.DataFrame({"prediction": model.predict(X)})
    if bundle["task"] == "classification":
        out["label"] = out.prediction.map(dict(enumerate(LABELS)))
        proba = pd.DataFrame(model.predict_proba(X), columns=[f"p_{LABELS[k]}" for k in model.classes_])
        out = out.join(proba.round(4))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--model", default="models/model.bin")
    p.add_argument("--out", help=".csv or .json; prints the first rows if omitted")
    a = p.parse_args()
    res = predict(load(a.model), read(a.input))
    if not a.out:
        print(res.head(20).to_string(index=False))
    elif a.out.endswith(".csv"):
        res.to_csv(a.out, index=False)
    else:
        res.to_json(a.out, orient="records", indent=1)
