"""Train and evaluate MKGMDA under the strict protocol (all association-derived inputs rebuilt per fold).

  python run_mkgmda.py --preset mkgmda --dataset hmdd_v32 --split colddisease --eval test --output results/v32_cd.json

--preset expands to the exact options of a model variant in configs/presets.json (mkgmda = the reported model;
the other presets are the ablations of Table 4); options given after it override the preset (sensitivity analysis).
--eval dev evaluates on the development split carved from each training fold; --eval test on the five test folds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from mkgmda.data import association_matrix, labeled_pairs
from mkgmda.metrics import binary_metrics
from mkgmda.preprocessing import build_fold_inputs, build_fold_views, load_benchmark
from mkgmda.similarity import load_intrinsic, profile_aware_integration
from mkgmda.splits import COLD_SIDES, dev_carve, folds_for
from mkgmda.training import TrainingConfig, score_pairs, train_model

ROOT = Path(__file__).resolve().parent
PRESETS = json.loads((ROOT / "configs" / "presets.json").read_text())["presets"]


def expand_preset(argv: list[str]) -> list[str]:
    if "--preset" not in argv:
        return argv
    i = argv.index("--preset")
    name = argv[i + 1]
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name] + argv[:i] + argv[i + 2:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--dataset", choices=["hmdd_v20", "hmdd_v32", "hmdd_v40"], default="hmdd_v32")
    p.add_argument("--split", choices=["random", *COLD_SIDES], default="random")
    p.add_argument("--eval", choices=["dev", "test"], default="test")
    p.add_argument("--external", choices=["none", "mean", "max"], default="none",
                   help="fusion of the gene-knowledge similarity into the intrinsic similarity (the model uses max)")
    p.add_argument("--global-view", choices=["dense", "guided", "diffusion", "intrinsic", "none"], default="dense")
    p.add_argument("--loss", choices=["bce", "nnpu"], default="bce")
    p.add_argument("--pu-prior", type=float, default=0.05)
    p.add_argument("--gamma-cl", type=float, default=0.2, help="contrastive weight lambda_cl")
    p.add_argument("--no-contrastive", action="store_true")
    p.add_argument("--no-proximity", action="store_true")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--fusion", choices=["attention", "mean"], default="attention")
    p.add_argument("--recovery", choices=["none", "knn", "dcp"], default="none", help="dcp = dual prediction")
    p.add_argument("--recovery-weight", type=float, default=0.1, help="lambda_rec")
    p.add_argument("--score-prior", choices=["none", "nodebias"], default="none")
    p.add_argument("--anova", action="store_true", help="degree-corrected functional-ANOVA decoder")
    p.add_argument("--view-dropout", type=float, default=0.0)
    p.add_argument("--recover-homogeneous", action="store_true")
    p.add_argument("--degree-recovery", action="store_true")
    p.add_argument("--degree-mode", choices=["regress", "propagate"], default="regress")
    p.add_argument("--degree-dropout", type=float, default=0.0, help="missing-view augmentation fraction p")
    p.add_argument("--cold-branch", action="store_true", help="association-free branch for cold-both pairs")
    p.add_argument("--branch-neighbour-only", action="store_true", help="branch encodes a node by its intrinsic neighbours only")
    p.add_argument("--branch-bilinear", action="store_true")
    p.add_argument("--degree-offset", action="store_true", help="degree coefficient initialized at the block-model value")
    p.add_argument("--drop-intrinsic-view", action="store_true")
    p.add_argument("--negative-mode", choices=["resample", "fixed"], default="resample")
    p.add_argument("--no-seq-cold", action="store_true", help="no sequence fallback in the homogeneous miRNA similarity")
    p.add_argument("--gip-fix", action="store_true", help="GIP only between two warm nodes")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--hidden", type=int, default=256, help="embedding dimension F")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--knn-k", type=int, default=20, help="nearest neighbours kappa")
    p.add_argument("--mu-prox", type=float, default=0.5, help="lambda_prox")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--output", required=True)
    p.add_argument("--save-scores", action="store_true", help="also write pairs, labels, and scores of every fold (.npz)")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(expand_preset(list(sys.argv[1:] if argv is None else argv)))


def make_config(args: argparse.Namespace, seed: int) -> TrainingConfig:
    return TrainingConfig(
        seed=seed, epochs=args.epochs, hidden=args.hidden, dropout=args.dropout, learning_rate=args.lr,
        knn_k=args.knn_k, gamma_contrastive=args.gamma_cl, mu_proximity=args.mu_prox,
        resample_unlabeled_each_epoch=args.negative_mode == "resample",
        use_global_view=args.global_view != "none" and not args.drop_intrinsic_view,
        use_global_prior=args.global_view == "guided",
        global_mode=args.global_view if args.global_view in ("diffusion", "intrinsic") else "attention",
        use_homogeneous_gate=args.gate, fusion_mode=args.fusion,
        impute_cold=args.recovery == "knn", view_recovery=args.recovery == "dcp", recovery_weight=args.recovery_weight,
        score_prior=args.score_prior, anova_decoder=args.anova, view_dropout=args.view_dropout,
        recover_homogeneous=args.recover_homogeneous, degree_recovery=args.degree_recovery,
        degree_mode=args.degree_mode, degree_dropout=args.degree_dropout, cold_branch=args.cold_branch,
        branch_neighbour_only=args.branch_neighbour_only, branch_bilinear=args.branch_bilinear,
        degree_offset=args.degree_offset, use_contrastive=not args.no_contrastive,
        use_proximity=not args.no_proximity, loss_mode=args.loss, pu_prior=args.pu_prior, device=args.device,
    )


def pair_mask(pairs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[pairs[:, 0], pairs[:, 1]] = True
    return mask


def fold_similarities(args, bench, train_assoc, sequence):
    """Association-derived similarities from the training associations of the fold only."""
    mirna_sim, disease_sim, _ = build_fold_inputs(train_assoc, bench.disease_semantic_similarity)
    if args.gip_fix:
        views = build_fold_views(train_assoc, bench.disease_semantic_similarity)
        mirna_sim = profile_aware_integration(views["mirna_functional"], views["mirna_gip"], train_assoc.sum(axis=1) > 0)
        disease_sim = profile_aware_integration(views["disease_semantic"], views["disease_gip"], train_assoc.sum(axis=0) > 0)
    empty = train_assoc.sum(axis=1) == 0
    if not args.no_seq_cold and empty.any():
        mirna_sim[empty, :] = sequence[empty, :]
        mirna_sim[:, empty] = sequence[:, empty]
        np.fill_diagonal(mirna_sim, 1.0)
    return mirna_sim, disease_sim


def fit(args, bench, train_pos, pool, sequence, semantic, seed):
    shape = (bench.n_mirna, bench.n_disease)
    train_assoc = association_matrix(train_pos, *shape)
    mirna_sim, disease_sim = fold_similarities(args, bench, train_assoc, sequence)
    model, tensors, _ = train_model(
        train_association=train_assoc, full_association=train_assoc,
        mirna_similarity=mirna_sim, disease_similarity=disease_sim,
        train_positive=train_pos, train_unlabeled_pool=pool, config=make_config(args, seed),
        # proximity negatives come from the training unlabeled pool only
        additional_proximity_exclusions=~pair_mask(pool, shape),
        intrinsic_similarities=(sequence, semantic) if args.global_view == "intrinsic" else None,
    )
    return model, tensors, train_assoc


def main() -> None:
    args = parse_args()
    bench = load_benchmark(args.data_root, args.dataset)
    sequence, semantic = load_intrinsic(Path(args.data_root), bench, args.external)
    wanted = {int(f) for f in args.folds.split(",")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for fold, train_pos, pool, test_pos, test_unl in folds_for(bench, args.split):
        if fold not in wanted:
            continue
        if args.eval == "dev":
            train_pos, pool, eval_pos, eval_unl = dev_carve(bench, args.split, fold, train_pos, pool)
        else:
            eval_pos, eval_unl = test_pos, test_unl
        model, tensors, train_assoc = fit(args, bench, train_pos, pool, sequence, semantic, args.seed + fold * 100)
        assert not train_assoc[eval_pos[:, 0], eval_pos[:, 1]].any()
        pairs, labels = labeled_pairs(eval_pos, eval_unl)
        scores = score_pairs(model, tensors, pairs)
        metrics = binary_metrics(labels, scores)
        if args.save_scores:
            np.savez(Path(args.output).with_suffix(f".fold{fold}.npz"), pairs=pairs, labels=labels, scores=scores)
        rows.append({"fold": fold, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}})
        print(f"[{args.dataset} {args.split} {args.eval} fold{fold}] auc={metrics['auc']:.4f} aupr={metrics['aupr']:.4f}", flush=True)
    keys = [k for k in rows[0] if k != "fold"]
    out = {"args": vars(args), "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
           "std": {k: float(np.std([r[k] for r in rows])) for k in keys}, "folds": rows, "seconds": time.time() - t0}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"MEAN auc={out['mean']['auc']:.4f} aupr={out['mean']['aupr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
