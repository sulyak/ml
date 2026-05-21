# coding:utf-8
"""
Phase 2 variants of RandomForestClassifier for class imbalance.

Option 1 — BalancedBootstrapRF
    Per-tree balanced bootstrap: all minority samples (with replacement) +
    equal number of majority samples (with replacement). Fixes the missing
    bootstrap in the base code and balances each tree's view of the data.

Option 2 — WeightedEntropyRF
    Class-weighted entropy split criterion. Each class is weighted by
    n_total / (n_classes * n_c), so minority samples dominate the impurity
    calculation. Leaf values are also weighted to be consistent.

Option 3 — AdaptiveThresholdRF
    Standard RF fit, but the decision threshold (default 0.5) is replaced by
    one learned from cross-validation on the training fold that maximises
    F1-minority. AUC-ROC is preserved; only the argmax boundary moves.

Option 4 — WeightedLeafRF
    Standard entropy splits, but leaf class probabilities are re-weighted by
    inverse class frequency before normalising. Tree structure is identical to
    baseline; only the output probabilities change.

Option 5 — PerTreeUndersampledRF
    Each tree receives a distinct, non-overlapping random subsample of majority
    class examples (without replacement per tree), combined with all minority
    examples. Rotating through the majority set ensures the whole dataset is
    covered across trees while maximising diversity between them.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from mla.ensemble.random_forest import RandomForestClassifier
from mla.ensemble.base import information_gain
from mla.ensemble.tree import Tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balanced_class_weights(y):
    """w_c = n_total / (n_classes * n_c)  — same formula as sklearn 'balanced'."""
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    n_classes = len(classes)
    w = np.zeros(n_classes)
    for c, count in zip(classes, counts):
        w[c] = n_total / (n_classes * float(count))
    return w


def _make_weighted_entropy_criterion(class_weights):
    n_classes = len(class_weights)

    def weighted_entropy(y):
        counts = np.bincount(y, minlength=n_classes).astype(float)
        weighted = counts * class_weights
        total = weighted.sum()
        if total == 0:
            return 0.0
        p = weighted / total
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def weighted_information_gain(y, splits):
        splits_entropy = sum(
            weighted_entropy(split) * (float(split.shape[0]) / y.shape[0])
            for split in splits
        )
        return weighted_entropy(y) - splits_entropy

    return weighted_information_gain


def _init_trees(n_estimators, criterion, class_weights=None):
    return [Tree(criterion=criterion, class_weights=class_weights) for _ in range(n_estimators)]


def _tree_train_kwargs(rf):
    return dict(
        max_features=rf.max_features,
        min_samples_split=rf.min_samples_split,
        max_depth=rf.max_depth,
    )


# ---------------------------------------------------------------------------
# Option 1
# ---------------------------------------------------------------------------

class BalancedBootstrapRF(RandomForestClassifier):
    """Per-tree balanced bootstrap sampling."""

    def _train(self):
        minority_idx = np.where(self.y == 1)[0]
        majority_idx = np.where(self.y == 0)[0]
        n_minority = len(minority_idx)
        rng = np.random.default_rng(42)

        for tree in self.trees:
            boot_min = rng.choice(minority_idx, size=n_minority, replace=True)
            boot_maj = rng.choice(majority_idx, size=n_minority, replace=True)
            idx = rng.permutation(np.concatenate([boot_min, boot_maj]))
            tree.train(self.X[idx], self.y[idx], **_tree_train_kwargs(self))


# ---------------------------------------------------------------------------
# Option 2
# ---------------------------------------------------------------------------

class WeightedEntropyRF(RandomForestClassifier):
    """Class-weighted entropy criterion + weighted leaf values."""

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert X.shape[1] > self.max_features

        cw = _balanced_class_weights(y)
        criterion = _make_weighted_entropy_criterion(cw)
        # Weighted leaves are required: without them the criterion change is
        # offset by raw-count leaf values that still favour the majority class.
        self.trees = _init_trees(self.n_estimators, criterion, class_weights=cw)
        self._train()


# ---------------------------------------------------------------------------
# Option 3
# ---------------------------------------------------------------------------

class AdaptiveThresholdRF(RandomForestClassifier):
    """Standard RF with a CV-tuned minority-class decision threshold."""

    def __init__(self, n_threshold_splits=5, **kwargs):
        super().__init__(**kwargs)
        self.n_threshold_splits = n_threshold_splits
        self.threshold_ = 0.5

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert X.shape[1] > self.max_features

        self.threshold_ = self._find_threshold(X, y)
        self._train()

    def _find_threshold(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_threshold_splits, shuffle=True, random_state=42)
        thresholds = np.linspace(0.05, 0.95, 19)
        f1_totals = np.zeros(len(thresholds))

        for train_idx, val_idx in skf.split(X, y):
            fold_rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            fold_rf.fit(X[train_idx], y[train_idx])
            probs = fold_rf.predict(X[val_idx])

            for j, t in enumerate(thresholds):
                y_pred = (probs[:, 1] >= t).astype(int)
                f1_totals[j] += f1_score(y[val_idx], y_pred, pos_label=1, average="binary", zero_division=0)

        best = thresholds[np.argmax(f1_totals)]
        return float(best)


# ---------------------------------------------------------------------------
# Option 4
# ---------------------------------------------------------------------------

class WeightedLeafRF(RandomForestClassifier):
    """Cost-sensitive leaf probabilities via inverse-frequency weighting."""

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert X.shape[1] > self.max_features

        cw = _balanced_class_weights(y)
        self.trees = _init_trees(self.n_estimators, information_gain, class_weights=cw)
        self._train()


# ---------------------------------------------------------------------------
# Combined: Options 1 + 2 + 3
# ---------------------------------------------------------------------------

class CombinedRF(RandomForestClassifier):
    """
    Options 1 + 2 + 3 stacked:
      - Balanced bootstrap per tree (opt 1)
      - Weighted entropy criterion + weighted leaf values (opt 2)
      - CV-tuned decision threshold (opt 3)

    The threshold is found using CV folds trained with opt1+opt2 so that it
    is calibrated to the same model that will be used at test time.
    """

    def __init__(self, n_threshold_splits=5, **kwargs):
        super().__init__(**kwargs)
        self.n_threshold_splits = n_threshold_splits
        self.threshold_ = 0.5

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert X.shape[1] > self.max_features

        # Option 2: weighted criterion + weighted leaf for the final model
        cw = _balanced_class_weights(y)
        criterion = _make_weighted_entropy_criterion(cw)
        self.trees = _init_trees(self.n_estimators, criterion, class_weights=cw)

        # Option 3: find threshold using CV with the same opt1+opt2 approach
        self.threshold_ = self._find_threshold(X, y)

        # Option 1+2: train final model with balanced bootstrap
        self._train()

    def _train(self):
        """Option 1: balanced bootstrap over already-configured weighted trees."""
        minority_idx = np.where(self.y == 1)[0]
        majority_idx = np.where(self.y == 0)[0]
        n_minority = len(minority_idx)
        rng = np.random.default_rng(42)

        for tree in self.trees:
            boot_min = rng.choice(minority_idx, size=n_minority, replace=True)
            boot_maj = rng.choice(majority_idx, size=n_minority, replace=True)
            idx = rng.permutation(np.concatenate([boot_min, boot_maj]))
            tree.train(self.X[idx], self.y[idx], **_tree_train_kwargs(self))

    def _find_threshold(self, X, y):
        """CV threshold search using opt1+opt2 fold models for correct calibration."""
        skf = StratifiedKFold(n_splits=self.n_threshold_splits, shuffle=True, random_state=42)
        thresholds = np.linspace(0.05, 0.95, 19)
        f1_totals = np.zeros(len(thresholds))

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            fold_cw = _balanced_class_weights(y_tr)
            fold_criterion = _make_weighted_entropy_criterion(fold_cw)
            fold_trees = _init_trees(self.n_estimators, fold_criterion, class_weights=fold_cw)

            min_idx = np.where(y_tr == 1)[0]
            maj_idx = np.where(y_tr == 0)[0]
            n_min = len(min_idx)
            rng = np.random.default_rng(42)

            for tree in fold_trees:
                boot_min = rng.choice(min_idx, size=n_min, replace=True)
                boot_maj = rng.choice(maj_idx, size=n_min, replace=True)
                idx = rng.permutation(np.concatenate([boot_min, boot_maj]))
                tree.train(X_tr[idx], y_tr[idx], max_features=self.max_features,
                           min_samples_split=self.min_samples_split, max_depth=self.max_depth)

            probs = np.zeros((len(val_idx), 2))
            for i in range(len(val_idx)):
                row_pred = np.zeros(2)
                for tree in fold_trees:
                    row_pred += tree.predict_row(X_val[i])
                probs[i] = row_pred / len(fold_trees)

            for j, t in enumerate(thresholds):
                y_pred = (probs[:, 1] >= t).astype(int)
                f1_totals[j] += f1_score(y_val, y_pred, pos_label=1, average="binary", zero_division=0)

        return float(thresholds[np.argmax(f1_totals)])


# ---------------------------------------------------------------------------
# Option 5
# ---------------------------------------------------------------------------

class PerTreeUndersampledRF(RandomForestClassifier):
    """Per-tree non-overlapping majority undersampling for maximum diversity."""

    def _train(self):
        minority_idx = np.where(self.y == 1)[0]
        majority_idx = np.where(self.y == 0)[0]
        n_minority = len(minority_idx)
        n_majority = len(majority_idx)
        rng = np.random.default_rng(42)

        # Shuffle majority once; slide a window of size n_minority per tree
        shuffled_maj = rng.permutation(majority_idx)

        for i, tree in enumerate(self.trees):
            start = (i * n_minority) % n_majority
            end = start + n_minority
            if end <= n_majority:
                maj_subset = shuffled_maj[start:end]
            else:
                maj_subset = np.concatenate([shuffled_maj[start:], shuffled_maj[: end % n_majority]])

            idx = rng.permutation(np.concatenate([minority_idx, maj_subset]))
            tree.train(self.X[idx], self.y[idx], **_tree_train_kwargs(self))
