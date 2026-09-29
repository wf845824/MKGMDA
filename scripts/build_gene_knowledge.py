"""Gene-knowledge similarities K^m and K^d (data/<dataset>/gene_knowledge_similarity.npz), built without HMDD labels.

K^m: best-match average, over the HumanNet-XN functional gene network, between the functional miRTarBase targets of two
miRNAs; K^d: the same between the DisGeNET v7 genes of two diseases. HumanNet contains no miRNA gene, so DisGeNET records
of miRNA genes, which would restate miRNA-disease associations, cannot enter the disease gene sets.
Raw inputs go to external_data/ (see README). Usage: python scripts/build_gene_knowledge.py
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numba import njit, prange, set_num_threads

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_hmdd_v40 import load_descriptors, load_supplementary, norm  # noqa: E402
from coremda.preprocessing import load_benchmark  # noqa: E402

CODE = Path(__file__).resolve().parents[1]  # repository root
OUT = CODE / "build" / "gene_knowledge"  # gene lists and name mappings (provenance)
RAW = CODE / "external_data"  # HumanNet-XN.tsv, all_gene_disease_associations.tsv.gz
MTI = RAW / "miRTarBase_10_hsa_MTI.csv"
DATASETS = ("hmdd_v20", "hmdd_v32", "hmdd_v40")


def mir_key(s):
    return re.sub(r"-(5p|3p)$", "", s.strip().lower())


def name_candidates(name, available):
    """Exact, arm removal, then numeric locus suffix only; never loose prefixes."""
    key = mir_key(name)
    if key in available:
        return [key], "exact_or_arm"
    # Only remove a locus number after an already complete miRNA number/letter.
    if re.search(r"(?:mir-\d+[a-z]*|let-\d+[a-z]*)-\d+$", key):
        base = re.sub(r"-\d+$", "", key)
        if base in available:
            return [base], "locus_to_mature"
    hits = sorted(k for k in available if re.fullmatch(re.escape(key) + r"-\d+", k))
    return hits, "numeric_loci" if hits else "missing"


def load_humannet():
    edges = np.loadtxt(RAW / "HumanNet-XN.tsv", comments="#")
    genes = np.unique(edges[:, :2].astype(np.int64))
    index = {int(g): i for i, g in enumerate(genes)}
    row = np.array([index[int(g)] for g in edges[:, 0]])
    col = np.array([index[int(g)] for g in edges[:, 1]])
    weights = edges[:, 2]
    lo, hi = float(weights.min()), float(weights.max())
    weights = ((weights - lo) / (hi - lo)).astype(np.float32)
    mat = sp.csr_matrix((weights, (row, col)), shape=(len(genes), len(genes)))
    mat = mat.maximum(mat.T)
    mat.setdiag(1)
    mat.eliminate_zeros()
    return mat, index, {"genes": len(genes), "edges": len(edges), "lls_min": lo, "lls_max": hi}


def mirna_gene_sets(benches, index):
    targets = defaultdict(set)
    rows_used = 0
    with MTI.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["Species (miRNA)"] != "hsa" or r["Species (Target Gene)"] != "hsa":
                continue
            if r["Support Type"] not in ("Functional MTI", "Functional MTI (Weak)"):
                continue
            try:
                gene = int(float(r["Target Gene (Entrez ID)"]))
            except (ValueError, OverflowError):
                continue
            if gene in index:
                targets[mir_key(r["miRNA"])].add(index[gene])
                rows_used += 1
    out = {}
    for b in benches:
        sets, rows = [], []
        for n in b.mirna_names:
            keys, method = name_candidates(str(n), targets)
            genes = set().union(*(targets[k] for k in keys))
            sets.append(genes)
            rows.append({"name": str(n), "method": method, "matched_names": ";".join(keys),
                         "n_genes": len(genes)})
        out[b.name] = (sets, rows)
    return out, {"source_records_used": rows_used, "source_mirna_keys": len(targets)}


def disease_gene_sets(benches, index):
    """Conservative exact/MeSH synonym joins, with ambiguous aliases left unmatched."""
    _, dn, dt = load_descriptors()
    _, sn, st = load_supplementary()
    names, terms = {**dn, **sn}, {**dt, **st}
    alias_uis = defaultdict(set)
    for ui, name in names.items():
        for s in terms[ui] | {name}:
            alias_uis[norm(s)].add(ui)
    ui_by_name = {s: next(iter(u)) for s, u in alias_uis.items() if len(u) == 1}
    # Preferred names identify their concept even if also an entry term elsewhere.
    ui_by_name.update({norm(name): ui for ui, name in names.items()})
    direct, by_ui, cui_direct, cui_ui = (defaultdict(set) for _ in range(4))
    records = 0
    with gzip.open(RAW / "all_gene_disease_associations.tsv.gz", "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            gene = int(r["geneId"])
            if gene not in index:
                continue
            k, cui = norm(r["diseaseName"]), r["diseaseId"]
            direct[k].add(index[gene]); cui_direct[k].add(cui)
            if k in ui_by_name:
                ui = ui_by_name[k]
                by_ui[ui].add(index[gene]); cui_ui[ui].add(cui)
            records += 1
    out = {}
    for b in benches:
        sets, rows = [], []
        for n in b.disease_names:
            key, ui = norm(str(n)), ui_by_name.get(norm(str(n)))
            genes = set(direct.get(key, ()))
            cuis = set(cui_direct.get(key, ()))
            method = "exact_name" if genes else "missing"
            if ui in by_ui:
                genes.update(by_ui[ui]); cuis.update(cui_ui[ui])
                method = "mesh_concept"
            sets.append(genes)
            rows.append({"name": str(n), "method": method, "mesh_ui": ui or "",
                         "disgenet_cuis": ";".join(sorted(cuis)), "n_genes": len(genes)})
        out[b.name] = (sets, rows)
    return out, {"source_records_used": records, "distinct_disease_names": len(direct),
                 "mesh_concepts_matched": len(by_ui)}


@njit(cache=True, parallel=True)
def best_gene_to_set(indptr, indices, values, group_ptr, group_genes):
    """Sparse max propagation: B[j,g] = max_{h in G_j} S(g,h)."""
    nsets, ngenes = len(group_ptr) - 1, len(indptr) - 1
    best = np.zeros((nsets, ngenes), np.float32)
    for j in prange(nsets):
        for z in range(group_ptr[j], group_ptr[j + 1]):
            gene = group_genes[z]
            for e in range(indptr[gene], indptr[gene + 1]):
                g = indices[e]
                if values[e] > best[j, g]:
                    best[j, g] = values[e]
    return best


def functional_similarity(sets, graph):
    sizes = np.array([len(s) for s in sets])
    ptr = np.r_[0, sizes.cumsum()].astype(np.int64)
    idx = np.array([g for s in sets for g in sorted(s)], np.int64)
    membership = sp.csr_matrix((np.ones(len(idx), np.float32), idx, ptr),
                               shape=(len(sets), graph.shape[0]))
    best = best_gene_to_set(graph.indptr, graph.indices, graph.data, ptr, idx)
    directed = membership @ best.T.copy()
    den = sizes[:, None] + sizes[None, :]
    sim = np.divide(directed + directed.T, den, out=np.zeros_like(directed), where=den > 0)
    np.fill_diagonal(sim, 1)
    return sim


def write_tsv(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)


def validate_matrix(a):
    assert a.dtype == np.float32 and a.ndim == 2 and a.shape[0] == a.shape[1]
    assert np.isfinite(a).all() and a.min() >= -1e-6 and a.max() <= 1 + 1e-6
    assert np.allclose(a, a.T, atol=1e-6)
    assert np.allclose(np.diag(a), 1)


def main():
    set_num_threads(4)
    start = time.time()
    benches = [load_benchmark(CODE / "data", d) for d in DATASETS]
    graph, index, network_info = load_humannet()
    print("HumanNet", network_info, flush=True)
    msets, minfo = mirna_gene_sets(benches, index)
    print("miRTarBase", minfo, flush=True)
    dsets, dinfo = disease_gene_sets(benches, index)
    print("DisGeNET", dinfo, flush=True)
    report = {"humannet": network_info, "mirtarbase": minfo, "disgenet": dinfo, "datasets": {}}
    gene_ids = np.array(sorted(index))
    for b in benches:
        path = OUT / b.name
        path.mkdir(parents=True, exist_ok=True)
        mf = functional_similarity(msets[b.name][0], graph)
        df = functional_similarity(dsets[b.name][0], graph)
        for a in (mf, df):
            validate_matrix(a)
        np.savez_compressed(CODE / "data" / b.name / "gene_knowledge_similarity.npz",
                            mirna=mf.astype(np.float32), disease=df.astype(np.float32),
                            mirna_names=np.array([str(x) for x in b.mirna_names]),
                            disease_names=np.array([str(x) for x in b.disease_names]))
        write_tsv(path / "mirna_gene_mapping.tsv", msets[b.name][1])
        write_tsv(path / "disease_gene_mapping.tsv", dsets[b.name][1])
        (path / "gene_sets.json").write_text(json.dumps({
            "mirna": [[int(gene_ids[g]) for g in sorted(s)] for s in msets[b.name][0]],
            "disease": [[int(gene_ids[g]) for g in sorted(s)] for s in dsets[b.name][0]]}))
        info = {"n_mirna": b.n_mirna, "n_disease": b.n_disease,
                "mirna_with_genes": sum(bool(s) for s in msets[b.name][0]),
                "disease_with_genes": sum(bool(s) for s in dsets[b.name][0])}
        report["datasets"][b.name] = info
        print(b.name, info, flush=True)
    report["seconds"] = time.time() - start
    (OUT / "coverage.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
