"""Case studies on HMDD v3.2 with the reported model (preset coremda).

  --mode holdout : all HMDD v3.2 associations of the disease are removed, every association-derived input is rebuilt,
                   and the disease is excluded from the unlabeled training pairs; all miRNAs are ranked (new disease).
  --mode novel   : the model is trained on all HMDD v3.2 associations; miRNAs without a recorded association to the
                   disease are ranked (novel associations).
  --related      : further HMDD v3.2 diseases of the same family (repeatable). In holdout mode their associations are
                   removed as well; in novel mode miRNAs recorded with any of them are not candidates.

  python run_case_study.py --mode holdout --disease "Colonic Neoplasms" \
      --related "Colorectal Neoplasms" --related "Rectal Neoplasms" --output results/case/colon_holdout.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import run_coremda as R
from coremda.data import unlabeled_pool
from coremda.metrics import sigmoid
from coremda.preprocessing import load_benchmark
from coremda.similarity import load_intrinsic
from coremda.training import score_pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["holdout", "novel"], required=True)
    p.add_argument("--disease", required=True, help='HMDD v3.2 disease name, e.g. "Breast Neoplasms"')
    p.add_argument("--related", action="append", default=[], help="HMDD v3.2 disease of the same family (repeatable)")
    p.add_argument("--output", required=True, help="ranked candidates (.csv)")
    p.add_argument("--device", default="cuda:0")
    own = p.parse_args()
    args = R.parse_args(["--preset", "coremda", "--dataset", "hmdd_v32", "--output", own.output, "--device", own.device])

    bench = load_benchmark(args.data_root, "hmdd_v32")
    sequence, semantic = load_intrinsic(Path(args.data_root), bench, args.external)
    names = [" ".join(str(n).casefold().split()) for n in bench.disease_names]
    target = names.index(" ".join(own.disease.casefold().split()))
    family = [target] + [names.index(" ".join(d.casefold().split())) for d in own.related]
    known = bench.positive_pairs[:, 1] == target
    in_family = np.isin(bench.positive_pairs[:, 1], family)
    pool = unlabeled_pool(bench.full_association)
    if own.mode == "holdout":
        train_pos, pool = bench.positive_pairs[~in_family], pool[~np.isin(pool[:, 1], family)]
        candidates = np.arange(bench.n_mirna)
    else:
        train_pos = bench.positive_pairs
        recorded = set(bench.positive_pairs[in_family][:, 0].tolist())
        candidates = np.array([i for i in range(bench.n_mirna) if i not in recorded], dtype=np.int64)
    model, tensors, train_assoc = R.fit(args, bench, train_pos, pool, sequence, semantic, args.seed)
    if own.mode == "holdout":
        assert not train_assoc[:, family].any()
    pairs = np.column_stack([candidates, np.full(len(candidates), target)]).astype(np.int64)
    scores = sigmoid(score_pairs(model, tensors, pairs))
    order = np.argsort(-scores, kind="mergesort")
    held = set(bench.positive_pairs[known][:, 0].tolist())
    held_family = set(bench.positive_pairs[in_family][:, 0].tolist())
    out = Path(own.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        extra = ["held_out_hmdd_v32"] + (["held_out_family_hmdd_v32"] if own.related else [])
        w.writerow(["rank", "mirna", "score"] + (extra if own.mode == "holdout" else []))
        for rank, k in enumerate(order, start=1):
            i = int(candidates[k])
            flags = [i in held] + ([i in held_family] if own.related else [])
            w.writerow([rank, str(bench.mirna_names[i]), float(scores[k])] + (flags if own.mode == "holdout" else []))
    out.with_suffix(".json").write_text(json.dumps({"mode": own.mode, "disease": str(bench.disease_names[target]),
                                                    "related": [str(bench.disease_names[d]) for d in family[1:]],
                                                    "n_candidates": len(candidates), "n_recorded_v32": len(held),
                                                    "n_recorded_family_v32": len(held_family)}, indent=2))
    print(f"[{own.mode}] {bench.disease_names[target]}: {len(candidates)} candidates ranked -> {out}")


if __name__ == "__main__":
    main()
