"""HMDD v4.0 benchmark in the MINIMDA file layout.

Nodes and associations: the HMDD v4.0 files released with RGCNMDA (Hou et al., Bioinformatics 2026;
Zenodo 10.5281/zenodo.21358475, dataset/HMDD31296/common/data). Its splits, fixed negatives and pathway bridge are not used.
Disease semantic similarity: Wang et al. (2010), delta = 0.5, on the MeSH 2026 descriptors; disease names are normalized
(lower case, punctuation removed) and matched to descriptor names and entry terms. If fewer than 90% of the diseases
match, supplementary concept records are also matched and attached below their HeadingMappedTo descriptors.

Usage: python scripts/build_hmdd_v40.py            (writes data/hmdd_v40/)
       python scripts/build_hmdd_v40.py --check-v32 (only compares the implementation with the MINIMDA HMDD v3.2 matrix)
"""

import gzip
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CODE = Path(__file__).resolve().parents[1]  # repository root
MESH = CODE / "external_data" / "mesh"  # MeSH 2026 desc/supp XML (see README)
SRC = CODE / "external_data" / "rgcnmda" / "HMDD31296" / "common" / "data"  # RGCNMDA Zenodo release
OUT = CODE / "data" / "hmdd_v40"
DELTA = 0.5


def norm(name):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", name.lower())).strip()


def load_descriptors():
    parents, names, terms = {}, {}, {}
    tree_to_ui, ui_trees = {}, {}
    with gzip.open(MESH / "desc2026.gz") as fh:
        for _, rec in ET.iterparse(fh):
            if rec.tag != "DescriptorRecord":
                continue
            ui = rec.findtext("DescriptorUI")
            names[ui] = rec.findtext("DescriptorName/String")
            ui_trees[ui] = [t.text for t in rec.findall("TreeNumberList/TreeNumber")]
            for t in ui_trees[ui]:
                tree_to_ui[t] = ui
            terms[ui] = {s.text for s in rec.findall("ConceptList/Concept/TermList/Term/String")}
            rec.clear()
    for ui, trees in ui_trees.items():
        parents[ui] = {tree_to_ui[t.rsplit(".", 1)[0]] for t in trees if "." in t and t.rsplit(".", 1)[0] in tree_to_ui}
    return parents, names, terms


def load_supplementary():
    parents, names, terms = {}, {}, {}
    with gzip.open(MESH / "supp2026.gz") as fh:
        for _, rec in ET.iterparse(fh):
            if rec.tag != "SupplementalRecord":
                continue
            ui = rec.findtext("SupplementalRecordUI")
            mapped = {h.findtext("DescriptorReferredTo/DescriptorUI").lstrip("*")
                      for h in rec.findall("HeadingMappedToList/HeadingMappedTo")}
            names[ui] = rec.findtext("SupplementalRecordName/String")
            terms[ui] = {s.text for s in rec.findall("ConceptList/Concept/TermList/Term/String")}
            parents[ui] = mapped
            rec.clear()
    return parents, names, terms


def build_index(names, terms):
    """normalized string -> concept UI; preferred names win over entry terms."""
    index = {}
    for ui, t in terms.items():
        for s in t:
            index.setdefault(norm(s), ui)
    for ui, n in names.items():
        index[norm(n)] = ui
    return index


def contributions(ui, parents):
    """Wang et al. semantic contribution of every ancestor t of ui: DELTA ** (shortest distance)."""
    out, queue = {ui: 1.0}, deque([ui])
    while queue:
        c = queue.popleft()
        for p in parents.get(c, ()):
            if p not in out:
                out[p] = out[c] * DELTA
                queue.append(p)
    return out


def wang_similarity(uis, parents):
    concepts, rows, cols, vals = {}, [], [], []
    for i, ui in enumerate(uis):
        if ui is None:
            continue
        for t, v in contributions(ui, parents).items():
            rows.append(i); cols.append(concepts.setdefault(t, len(concepts))); vals.append(v)
    m = sp.csr_matrix((vals, (rows, cols)), shape=(len(uis), max(1, len(concepts))))
    b = m.copy(); b.data[:] = 1.0
    num = (m @ b.T + b @ m.T).toarray()
    dv = np.asarray(m.sum(axis=1)).ravel()
    den = dv[:, None] + dv[None, :]
    sim = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def match(disease_names, index):
    return [index.get(norm(n)) for n in disease_names]


def check_v32(parents, index):
    names = [l.strip() for l in (CODE / "data" / "hmdd_v32" / "disease number.txt").read_text().splitlines() if l.strip()]
    ref = np.loadtxt(CODE / "data" / "hmdd_v32" / "disease semantic similarity matrix.txt", dtype=np.float32)
    uis = match(names, index)
    ok = np.array([u is not None for u in uis])
    sim = wang_similarity(uis, parents)
    iu = np.triu_indices(len(names), 1)
    both = ok[iu[0]] & ok[iu[1]]
    r = float(np.corrcoef(sim[iu][both], ref[iu][both])[0, 1])
    exact = float(np.mean(np.abs(sim[iu][both] - ref[iu][both]) < 1e-3))
    return {"v32_diseases": len(names), "v32_matched": int(ok.sum()), "pearson_offdiag_matched": r,
            "fraction_within_1e-3": exact}


def main():
    d_par, d_names, d_terms = load_descriptors()
    d_index = build_index(d_names, d_terms)
    report = {"mesh": "desc2026.gz", "delta": DELTA, "v32_check": check_v32(d_par, d_index)}
    print(json.dumps(report["v32_check"], indent=2), flush=True)
    if "--check-v32" in sys.argv:
        return
    mir = pd.read_csv(SRC / "mirna_mapping.csv", encoding="utf-8-sig").sort_values("miRNA_index")
    dis = pd.read_csv(SRC / "disease_mapping.csv", encoding="utf-8-sig").sort_values("disease_index")
    assert (mir.miRNA_index.values == np.arange(len(mir))).all() and (dis.disease_index.values == np.arange(len(dis))).all()
    pairs = pd.read_csv(SRC / "pairs_0based.csv", header=None).values.astype(np.int64)
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    diseases = dis.disease.astype(str).tolist()
    uis = match(diseases, d_index)
    coverage = float(np.mean([u is not None for u in uis]))
    report.update(n_mirna=len(mir), n_disease=len(dis), n_associations=len(pairs), descriptor_coverage=coverage)
    parents, all_names, source = d_par, dict(d_names), ["descriptor" if u else None for u in uis]
    if coverage < 0.90:  # pre-registered fallback: supplementary concepts under their mapped descriptors
        s_par, s_names, s_terms = load_supplementary()
        s_index = build_index(s_names, s_terms)
        parents = {**d_par, **s_par}
        all_names.update(s_names)
        for i, n in enumerate(diseases):
            if uis[i] is None and norm(n) in s_index:
                uis[i], source[i] = s_index[norm(n)], "supplementary"
        report["coverage_with_supplementary"] = float(np.mean([u is not None for u in uis]))
    sim = wang_similarity(uis, parents)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "miRNA number.txt").write_text("\n".join(mir.miRNA.astype(str)) + "\n")
    (OUT / "disease number.txt").write_text("\n".join(diseases) + "\n")
    np.savetxt(OUT / "known miRNA-disease association.txt", pairs + 1, fmt="%d")
    np.savetxt(OUT / "disease semantic similarity matrix.txt", sim, fmt="%.6f")
    pd.DataFrame({"disease": diseases, "mesh_ui": uis, "source": source,
                  "mesh_name": [all_names.get(u) if u else None for u in uis]}).to_csv(OUT / "mesh_mapping.tsv", sep="\t", index=False)
    (OUT / "build_info.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
