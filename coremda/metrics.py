"""Dependency-light binary ranking metrics used by the reproducibility scripts."""

from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_value = np.exp(logits[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum = ranks[positive].sum()
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-wise average precision with tied scores sharing one threshold (as in scikit-learn),
    so the value does not depend on the order of tied pairs."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels, ordered_scores = labels[order], scores[order]
    n_positive = int(ordered_labels.sum())
    if n_positive == 0:
        return float("nan")
    last = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), len(ordered_scores) - 1]
    true_positive = np.cumsum(ordered_labels)[last]
    precision = true_positive / (last + 1)
    recall_step = np.diff(np.r_[0, true_positive]) / n_positive
    return float((precision * recall_step).sum())


def binary_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = sigmoid(logits)
    predictions = probabilities >= threshold
    positive = labels == 1
    true_positive = int(np.logical_and(predictions, positive).sum())
    false_positive = int(np.logical_and(predictions, ~positive).sum())
    false_negative = int(np.logical_and(~predictions, positive).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "auc": roc_auc(labels, logits),
        "aupr": average_precision(labels, logits),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def top_k_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    k_values: tuple[int, ...] = (50, 100),
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    n_positive = int(labels.sum())
    metrics: dict[str, float] = {}
    for requested_k in k_values:
        k = min(requested_k, len(labels))
        hits = int(labels[order[:k]].sum())
        metrics[f"precision_at_{requested_k}"] = hits / max(1, k)
        metrics[f"recall_at_{requested_k}"] = hits / max(1, n_positive)
    return metrics
