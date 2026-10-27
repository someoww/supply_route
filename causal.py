"""Causal effects on route outcomes, checked against the SCM's own do() simulations.

Four numbers per question, all for the same contrast:
  oracle    re-simulate the world under do(T), same noise (known only because the data is synthetic)
  naive     mean outcome difference between treated and untreated rows
  backdoor  g-formula: outcome model on T and T's direct causes, averaged over the observed causes
  all-feat  the same g-formula, but adjusting for every dispatch-time feature (what flipping a flag
            in the predictive model does); this blocks the effect's own mediators
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as Reg

from data_generator import SCM, exante, simulate

ONE = lambda x: np.ones_like(x)
ZERO = lambda x: np.zeros_like(x)
SAME = lambda x: x
UP = lambda k, cap=np.inf: lambda x: np.minimum(x + k, cap)

# treatment, outcome, label, (treated, control) values as functions of the observed treatment column
QUESTIONS = [
    ("strike", "condition_score", "transport strike on vs off", ONE, ZERO),
    ("tcb", "shortfall", "state market intervention on vs off", ONE, ZERO),
    ("fuel_bdt", "condition_score", "local diesel +10 BDT/litre", UP(10), SAME),
    ("rain_mm", "condition_score", "+20 mm rain", UP(20), SAME),
    ("cold_chain", "spoilage_pct", "cold chain +0.1", UP(.1, 1), SAME),
]


def g_formula(df, t, y, adjust, hi, lo):
    X = df[[t, *adjust]]
    m = Reg(max_iter=250, learning_rate=.08, min_samples_leaf=20, random_state=0).fit(X, df[y])
    return float(np.mean(m.predict(X.assign(**{t: hi(df[t])})) - m.predict(X.assign(**{t: lo(df[t])}))))


def effects(df, t, y, hi, lo, blocks=5):
    parents = list(SCM[t][0])
    everything = [c for c in exante(df.columns) if c != t]
    runs = {"backdoor": parents, "all_feat": everything}
    est = {k: g_formula(df, t, y, z, hi, lo) for k, z in runs.items()}
    day = df.timestamp.rank(method="dense")
    part = pd.qcut(day, blocks, labels=False)
    se = np.std([g_formula(df[part == b], t, y, parents, hi, lo) for b in range(blocks)], ddof=1) / np.sqrt(blocks)
    return est, se


def run(days, seed):
    df = simulate(days, seed)
    rows = []
    for t, y, label, hi, lo in QUESTIONS:
        truth = (simulate(days, seed, do={t: hi})[y] - simulate(days, seed, do={t: lo})[y]).mean()
        treated = df[t] > df[t].median() if df[t].nunique() > 2 else df[t] == 1
        est, se = effects(df, t, y, hi, lo)
        rows.append(dict(question=label, outcome=y, oracle=truth, naive=df[y][treated].mean() - df[y][~treated].mean(),
                         backdoor=est["backdoor"], se=se, all_feat=est["all_feat"]))
    return pd.DataFrame(rows).set_index("question").round(3)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    print(run(a.days, a.seed).to_string())
