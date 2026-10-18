"""rate_label_leakage.py - is event rate a label shortcut? (paper Sec. "rate/class confound")

For a preprocessed dataset, quantifies how much class information the scalar
per-recording log event rate ``log_lambda_raw`` carries, using only that scalar:

  * one-way ANOVA of log-rate across classes (F, p)
  * mutual information I(log-rate; y) in nats (kNN estimator, k=3, mean of 5 seeds) vs. class
    entropy H(y)  -> "% of H(y) explained". A binned plug-in estimate is also stored; it is
    biased upwards when classes are many and samples few, so use the kNN number.
  * between-class share of log-rate variance (eta^2) and per-class mean spread
  * rate-only classifiers (logistic regression, small MLP) fit on the TRAIN split,
    scored on the TEST split, vs. the most-frequent-class chance level

The paper's argument: if rate were merely a leaked label, the benefit of rate
conditioning should grow with leakage. Across datasets it does not.

    python scripts/analysis/rate_label_leakage.py --config configs/ncaltech101.yaml --workers 8
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.config import load_config, results_dir  # noqa: E402


def _load_one(path):
    d = torch.load(path, weights_only=False) if "weights_only" in torch.load.__code__.co_varnames \
        else torch.load(path)
    return float(d.log_lambda_raw.view(-1)[0]), int(d.y.view(-1)[0])


def load_split(processed_dir, split, workers):
    files = sorted(glob.glob(os.path.join(processed_dir, split, "*.pt")))
    with mp.Pool(workers) as pool:
        out = pool.map(_load_one, files, chunksize=32)
    lam = np.array([o[0] for o in out], dtype=np.float64)
    y = np.array([o[1] for o in out], dtype=np.int64)
    return lam, y


def entropy_nats(y):
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mutual_information_nats(lam, y, n_bins=32):
    """Plug-in MI between an equal-frequency-binned scalar and the label."""
    edges = np.quantile(lam, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.digitize(lam, edges)
    joint = np.zeros((n_bins, y.max() + 1))
    np.add.at(joint, (b, y), 1)
    joint /= joint.sum()
    pb, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pb @ py)[nz])).sum())


def mutual_information_knn_nats(lam, y, k=3, seeds=5):
    from sklearn.feature_selection import mutual_info_classif
    return float(np.mean([
        mutual_info_classif(lam[:, None], y, discrete_features=False, n_neighbors=k, random_state=s)[0]
        for s in range(seeds)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n_bins", type=int, default=32)
    ap.add_argument("--name", default="rate_label_leakage")
    args = ap.parse_args()

    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    cfg = load_config(args.config)
    proc = cfg["paths"]["processed"]
    lam_tr, y_tr = load_split(proc, "train", args.workers)
    lam_te, y_te = load_split(proc, "test", args.workers)

    groups = [lam_te[y_te == c] for c in np.unique(y_te) if (y_te == c).sum() > 1]
    F, p = stats.f_oneway(*groups)
    H = entropy_nats(y_te)
    mi = mutual_information_knn_nats(lam_te, y_te)
    mi_plugin = mutual_information_nats(lam_te, y_te, args.n_bins)

    class_means = np.array([g.mean() for g in groups])
    tot_var = lam_te.var()
    between = float(sum(len(g) * (g.mean() - lam_te.mean()) ** 2 for g in groups) / len(lam_te))

    mu, sd = lam_tr.mean(), lam_tr.std() + 1e-9
    xtr, xte = ((lam_tr - mu) / sd)[:, None], ((lam_te - mu) / sd)[:, None]
    majority = float(np.bincount(y_te).max() / len(y_te))
    lr = LogisticRegression(max_iter=2000).fit(xtr, y_tr).score(xte, y_te)
    mlp = MLPClassifier((64, 64), max_iter=500, random_state=0).fit(xtr, y_tr).score(xte, y_te)

    res = {
        "dataset": cfg["dataset"]["name"], "n_test": int(len(y_te)), "n_classes": int(len(np.unique(y_te))),
        "log_rate_range_nats": float(lam_te.max() - lam_te.min()),
        "anova_F": float(F), "anova_p": float(p),
        "class_entropy_nats": H, "mutual_information_nats": mi,
        "mi_fraction_of_entropy": mi / H,
        "mi_plugin_binned_biased_nats": mi_plugin,
        "between_class_variance_share": between / tot_var,
        "per_class_mean_span_nats": float(class_means.max() - class_means.min()),
        "per_class_mean_std_nats": float(class_means.std()),
        "most_frequent_class_acc": majority, "rate_only_logreg_acc": float(lr), "rate_only_mlp_acc": float(mlp),
    }
    out = results_dir(cfg, args.name) / "leakage.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"[leakage] wrote {out}")


if __name__ == "__main__":
    main()
