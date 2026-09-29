"""Evaluation splits (five folds, seed 42): random, cold-disease, cold-miRNA, and cold-both, plus the development
splits carved from each training fold. Test pairs are identical to those of all methods compared in the paper."""

from __future__ import annotations

import numpy as np

from .data import exclude_pairs, sample_pairs, unlabeled_pool

DEV_FRACTION = 0.1
COLD_SIDES = {"colddisease": "disease", "coldmirna": "mirna", "coldboth": "both"}


def make_positive_folds(positive_pairs: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(positive_pairs))
    return [positive_pairs[index] for index in np.array_split(permutation, n_folds)]


def random_folds(bench, n_folds=5, seed=42):
    """Yield (fold, train_positive, train_unlabeled_pool, test_positive, test_unlabeled) for the random split."""
    complete = unlabeled_pool(bench.full_association)
    pos_folds = make_positive_folds(bench.positive_pairs, n_folds, seed)
    for f in range(n_folds):
        test_pos = pos_folds[f]
        train_pos = np.vstack([pos_folds[i] for i in range(n_folds) if i != f])
        test_unl = sample_pairs(complete, len(test_pos), np.random.default_rng(seed + f * 100))
        yield f, train_pos, exclude_pairs(complete, test_unl, bench.n_disease), test_pos, test_unl


def split_by_nodes(pos, pool, held_m, held_d, both=False):
    """Split positives and unlabeled pool into (outside, inside) the held-out nodes.

    One side: inside = pairs touching a held node. Both sides: inside = pairs whose miRNA and disease are both held,
    outside = pairs touching neither (mixed pairs are dropped)."""
    def masks(x):
        m, d = np.isin(x[:, 0], held_m), np.isin(x[:, 1], held_d)
        return (m & d, ~m & ~d) if both else (m | d, ~(m | d))
    (ip, op), (iu, ou) = masks(pos), masks(pool)
    return pos[op], pool[ou], pos[ip], pool[iu]


def cold_folds(bench, side, n_folds=5, seed=42):
    """Cold-disease, cold-miRNA ("mirna"), or cold-both folds: held-out entities have no training association."""
    complete = unlabeled_pool(bench.full_association)
    dis_groups = np.array_split(np.random.default_rng(seed).permutation(bench.n_disease), n_folds)
    mir_groups = np.array_split(np.random.default_rng(seed + 1).permutation(bench.n_mirna), n_folds)
    none = np.empty(0, dtype=np.int64)
    for f in range(n_folds):
        held_m = mir_groups[f] if side in ("mirna", "both") else none
        held_d = dis_groups[f] if side in ("disease", "both") else none
        train_pos, train_pool, test_pos, test_cand = split_by_nodes(
            bench.positive_pairs, complete, held_m, held_d, side == "both")
        test_unl = sample_pairs(test_cand, len(test_pos), np.random.default_rng(seed + f * 100))
        yield f, train_pos, train_pool, test_pos, test_unl


def dev_carve(bench, split, fold, train_pos, train_pool):
    """Development split with the structure of the test split, carved from the training data of one fold."""
    rng = np.random.default_rng(7000 + fold)
    if split == "random":
        order = rng.permutation(len(train_pos))
        n_dev = int(round(DEV_FRACTION * len(train_pos)))
        dev_pos, inner_pos = train_pos[order[:n_dev]], train_pos[order[n_dev:]]
        dev_unl = sample_pairs(train_pool, n_dev, rng)
        return inner_pos, exclude_pairs(train_pool, dev_unl, bench.n_disease), dev_pos, dev_unl
    side = COLD_SIDES[split]
    frac = 2 * DEV_FRACTION if side == "both" else DEV_FRACTION  # both sides: 20% of each type
    none = np.empty(0, dtype=np.int64)
    held_m, held_d = none, none
    if side in ("disease", "both"):
        train_dis = np.unique(np.r_[train_pos[:, 1], train_pool[:, 1]])
        held_d = rng.permutation(train_dis)[: int(round(frac * len(train_dis)))]
    if side in ("mirna", "both"):
        train_mir = np.unique(np.r_[train_pos[:, 0], train_pool[:, 0]])
        held_m = rng.permutation(train_mir)[: int(round(frac * len(train_mir)))]
    inner_pos, inner_pool, dev_pos, dev_cand = split_by_nodes(train_pos, train_pool, held_m, held_d, side == "both")
    return inner_pos, inner_pool, dev_pos, sample_pairs(dev_cand, len(dev_pos), rng)


def folds_for(bench, split):
    return random_folds(bench) if split == "random" else cold_folds(bench, COLD_SIDES[split])
