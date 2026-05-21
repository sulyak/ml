"""
Benchmark script for evaluating RandomForest variants on class-imbalance datasets.

Quick test (first dataset only):
    python benchmark.py

Full run over all datasets:
    python benchmark.py --all
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from mla.ensemble.random_forest import RandomForestClassifier
from mla.ensemble.variants import (
    BalancedBootstrapRF, WeightedEntropyRF, AdaptiveThresholdRF,
    WeightedLeafRF, PerTreeUndersampledRF, CombinedRF,
)

DATA_PATH = "class_imbalance/"
N_ESTIMATORS = 20
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
N_SPLITS = 5
RANDOM_STATE = 42


def _rf():
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)

def _opt1():
    return BalancedBootstrapRF(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)

def _opt2():
    return WeightedEntropyRF(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)

def _opt3():
    return AdaptiveThresholdRF(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)

def _opt4():
    return WeightedLeafRF(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)

def _opt5():
    return PerTreeUndersampledRF(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)

def _combined():
    return CombinedRF(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)


def imbalance_ratio(y):
    counts = Counter(y)
    return max(counts.values()) / min(counts.values())


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    target_col = df.columns[-1]
    X_df = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].copy()

    for col in X_df.columns:
        if X_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = X_df[col].fillna(X_df[col].median())
            else:
                X_df[col] = X_df[col].fillna(X_df[col].mode()[0])

    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X = X_df.values.astype(np.float64)
    y = LabelEncoder().fit_transform(y_raw.astype(str))

    counts = Counter(y)
    if counts[0] < counts[1]:
        y = 1 - y

    return X, y


def load_all_datasets(data_path=DATA_PATH):
    datasets = []
    for fname in sorted(os.listdir(data_path)):
        if not fname.endswith(".csv"):
            continue
        X, y = load_dataset(os.path.join(data_path, fname))
        datasets.append((X, y, fname.replace(".csv", "")))
    return datasets


def evaluate(clf_factory, X, y, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1_min, f1_mac, aucs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = clf_factory()
        clf.fit(X_train, y_train)
        probs = clf.predict(X_test)
        if hasattr(clf, "threshold_"):
            y_pred = (probs[:, 1] >= clf.threshold_).astype(int)
        else:
            y_pred = np.argmax(probs, axis=1)

        accs.append(np.mean(y_pred == y_test))
        f1_min.append(f1_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0))
        f1_mac.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        try:
            aucs.append(roc_auc_score(y_test, probs[:, 1]))
        except ValueError:
            aucs.append(float("nan"))

    return {
        "Accuracy":    np.mean(accs),
        "F1_Minority": np.mean(f1_min),
        "F1_Macro":    np.mean(f1_mac),
        "AUC_ROC":     np.nanmean(aucs),
    }


def _run(clf_factory, datasets, label):
    rows = []
    print(f"\n=== {label} ===")
    for X, y, name in datasets:
        print(f"  {name} ...", end=" ", flush=True)
        m = evaluate(clf_factory, X, y)
        ir = imbalance_ratio(y)
        row = {"Dataset": name, "IR": round(ir, 1)} | {k: round(v, 3) for k, v in m.items()}
        rows.append(row)
        print(f"IR={ir:.0f}  F1_min={m['F1_Minority']:.3f}  F1_mac={m['F1_Macro']:.3f}  AUC={m['AUC_ROC']:.3f}")
    df = pd.DataFrame(rows).sort_values("IR").reset_index(drop=True)
    print(f"  Mean — F1_min={df['F1_Minority'].mean():.3f}  F1_mac={df['F1_Macro'].mean():.3f}  AUC={df['AUC_ROC'].mean():.3f}")
    return df


def benchmark_baseline(datasets=None):
    if datasets is None:
        datasets = load_all_datasets()
    return _run(_rf, datasets, "Baseline RF")


def benchmark_opt1(datasets=None):
    if datasets is None:
        datasets = load_all_datasets()
    return _run(_opt1, datasets, "Option 1 — Balanced Bootstrap")


def benchmark_opt2(datasets=None):
    if datasets is None:
        datasets = load_all_datasets()
    return _run(_opt2, datasets, "Option 2 — Weighted Entropy")


def benchmark_opt3(datasets=None):
    if datasets is None:
        datasets = load_all_datasets()
    return _run(_opt3, datasets, "Option 3 — Adaptive Threshold")


def benchmark_opt4(datasets=None):
    if datasets is None:
        datasets = load_all_datasets()
    return _run(_opt4, datasets, "Option 4 — Weighted Leaf")


def benchmark_opt5(datasets=None):
    if datasets is None:
        datasets = load_all_datasets()
    return _run(_opt5, datasets, "Option 5 — Per-Tree Undersampling")


def benchmark_combined(datasets=None):
    if datasets is None:
        datasets = load_all_datasets()
    return _run(_combined, datasets, "Combined — Opt 1 + 2 + 3")


if __name__ == "__main__":
    all_datasets = load_all_datasets()
    datasets = all_datasets if "--all" in sys.argv else all_datasets[:1]

    benchmark_baseline(datasets)
    benchmark_opt1(datasets)
    benchmark_opt2(datasets)
    benchmark_opt3(datasets)
    benchmark_opt4(datasets)
    benchmark_opt5(datasets)
    benchmark_combined(datasets)
