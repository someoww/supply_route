"""Train a dispatch-time route-condition model on ex-ante features only."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as Clf
from sklearn.ensemble import HistGradientBoostingRegressor as Reg
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                             r2_score, recall_score, root_mean_squared_error)
from sklearn.model_selection import TimeSeriesSplit

from data_generator import exante

TASKS = {"condition_class": "classification", "condition_score": "regression", "delay_hours": "regression"}
LABELS = ["normal", "degraded", "critical"]
GAP_DAYS = 30  # autocorrelated weather and unrest: keep a buffer between train and test


def fit(X, y, task, rounds=600):
    """Boost to `rounds`, keep the iteration count that is best on the newest 15% of rows, refit on all."""
    make = lambda n: (Clf if task == "classification" else Reg)(
        max_iter=n, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1., early_stopping=False, random_state=0)
    cut = int(len(X) * .85)
    Xv, yv = X.iloc[cut:], y.iloc[cut:]
    probe = make(rounds).fit(X.iloc[:cut], y.iloc[:cut])
    if task == "classification":
        losses = [log_loss(yv, p, labels=probe.classes_) for p in probe.staged_predict_proba(Xv)]
    else:
        losses = [np.mean((yv - p) ** 2) for p in probe.staged_predict(Xv)]
    return make(1 + int(np.argmin(losses))).fit(X, y)


def score(model, X, y, task):
    p = model.predict(X)
    if task == "regression":
        m = dict(mae=mean_absolute_error(y, p), rmse=root_mean_squared_error(y, p), r2=r2_score(y, p))
    else:
        recall = recall_score(y, p, labels=model.classes_, average=None, zero_division=0)
        m = dict(accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"),
                 log_loss=log_loss(y, model.predict_proba(X), labels=model.classes_),
                 **{f"recall_{LABELS[k]}": r for k, r in zip(model.classes_, recall)})
    return {k: round(float(v), 4) for k, v in m.items()}


def main(a):
    task = TASKS[a.target]
    df = pd.read_csv(a.data, parse_dates=["timestamp"]).sort_values("timestamp", kind="stable", ignore_index=True)
    cols = exante(df.columns)
    days = np.sort(df.timestamp.unique())
    cutoff = days[int(len(days) * (1 - a.holdout))]
    tr = df[df.timestamp < cutoff - pd.Timedelta(days=GAP_DAYS)]
    te = df[df.timestamp >= cutoff]
    X, y = tr[cols], tr[a.target]

    gap = GAP_DAYS * len(tr) // tr.timestamp.nunique()
    folds = [score(fit(X.iloc[i], y.iloc[i], task), X.iloc[j], y.iloc[j], task)
             for i, j in TimeSeriesSplit(a.folds, gap=gap).split(X)]
    cv = {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
          for k, v in pd.DataFrame(folds).items()}

    model = fit(X, y, task)
    holdout = score(model, te[cols], te[a.target], task)
    out = Path(a.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(dict(model=model, features=cols, target=a.target, task=task,
                                      rounds=model.n_iter_, cv=cv, holdout=holdout)))
    print(json.dumps(dict(target=a.target, features=len(cols), rounds=model.n_iter_,
                          cv={k: f"{m:.3f} ± {s:.3f}" for k, (m, s) in cv.items()}, holdout=holdout), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/routes.csv")
    p.add_argument("--target", default="condition_class", choices=TASKS)
    p.add_argument("--model-out", default="models/model.bin")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--holdout", type=float, default=.2, help="newest fraction of days held out")
    main(p.parse_args())
