"""Recalculate S6 macro-averaged AUROC 95% confidence intervals.

Target: arithmetic mean of eight profile-specific AUROCs, each measured on
identical 1,453 genes with the six specified benchmark models.
Bootstrap unit: gene identity; a class-stratified resampling of the same gene
indices is reused in all eight profiles and for all six models.

Run from Jupyter (paste cell) or `python Supplementary/analysis_scripts/paired_gene_bootstrap.py`.
The input path and output directory can be overridden via environment variables:
ORIGENE_S6_CSV, ORIGENE_S6_OUTPUT_DIR, ORIGENE_S6_EXPECTED_GENES,
ORIGENE_S6_N_BOOTSTRAP, ORIGENE_S6_SEED.
"""

from pathlib import Path
import os
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path.cwd()
CSV_PATH = Path(os.environ.get(
    "ORIGENE_S6_CSV",
    str(ROOT / "Data/Main_figures/benchmark/final_prediction_merged_all.csv"),
))
OUTDIR = Path(os.environ.get(
    "ORIGENE_S6_OUTPUT_DIR",
    str(ROOT / "results/supplementary/paired_gene_bootstrap"),
))
B = int(os.environ.get("ORIGENE_S6_N_BOOTSTRAP", "5000"))
SEED = int(os.environ.get("ORIGENE_S6_SEED", "20261026"))
EXPECTED_GENES = int(os.environ.get("ORIGENE_S6_EXPECTED_GENES", "1453"))
BATCH_SIZE = 250

MODELS = [
    "OriGENE_Mut", "OriGENE", "DORGE_all_ElasticNet",
    "DORGE_epi_ElasticNet", "DORGE_mut_ElasticNet", "20/20+",
]
DATASETS = [
    "Lung_Normal_I", "Lung_Normal_II", "Liver_Normal_I", "Liver_Normal_II",
    "Lung_Cancer_I", "Lung_Cancer_II", "Liver_Cancer_I", "Liver_Cancer_II",
]
LABELS = {
    "OriGENE_Mut": "OriGENE (Epi + Mut)",
    "OriGENE": "OriGENE (Epi)",
    "DORGE_all_ElasticNet": "DORGE (Epi + Mut)",
    "DORGE_epi_ElasticNet": "DORGE (Epi)",
    "DORGE_mut_ElasticNet": "DORGE (Mut)",
    "20/20+": "20/20+ (Mut)",
}


def file_sha256(path, blocksize=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(blocksize):
            h.update(chunk)
    return h.hexdigest()


def load_aligned_data(path):
    if not path.is_file():
        raise FileNotFoundError(f"Final prediction CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"gene", "label", "score", "model", "dataset"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in final CSV: {sorted(missing)}")

    df = df[df["model"].isin(MODELS) & df["dataset"].isin(DATASETS)].copy()
    df["gene_key"] = df["gene"].astype(str).str.strip().str.upper()
    df["score"] = pd.to_numeric(df["score"], errors="raise")
    df["label"] = pd.to_numeric(df["label"], errors="raise")
    if df.empty or df["gene_key"].eq("").any():
        raise ValueError("Empty data or gene names encountered.")
    if not np.isfinite(df["score"].to_numpy()).all():
        raise ValueError("Non-finite scores in selected benchmark data.")
    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0/1.")
    if df.duplicated(["model", "dataset", "gene_key"]).any():
        raise ValueError("Duplicate model/dataset/gene rows; cannot align benchmark.")
    if df.groupby("gene_key")["label"].nunique().gt(1).any():
        raise ValueError("A gene has conflicting CD/NG labels across models/profiles.")

    # Require every model/dataset prediction for the SAME gene; otherwise a
    # paired gene bootstrap across profiles would not be well-defined.
    expected_pairs = len(MODELS) * len(DATASETS)
    counts = df.groupby("gene_key").size()
    shared_genes = sorted(counts[counts == expected_pairs].index.tolist())
    if len(shared_genes) != EXPECTED_GENES:
        raise ValueError(
            f"Expected {EXPECTED_GENES} fully shared genes across all models and profiles, "
            f"but found {len(shared_genes)}. Inspect the model/dataset combinations "
            "and panel before interpreting bootstrap CIs."
        )
    df = df[df["gene_key"].isin(shared_genes)].copy()
    assert len(df) == len(shared_genes) * expected_pairs
    label_by_gene = df.groupby("gene_key")["label"].first().reindex(shared_genes)
    y = label_by_gene.to_numpy(dtype=np.int8)
    if set(y) != {0, 1}:
        raise ValueError("Both CD and NG classes are needed.")
    if EXPECTED_GENES == 1453 and ((y == 1).sum(), (y == 0).sum()) != (726, 727):
        raise ValueError("Expected 726 CD / 727 NG in the six-model common universe.")

    score_matrices = {}
    for model in MODELS:
        pivoted = df[df["model"].eq(model)].pivot(
            index="gene_key", columns="dataset", values="score"
        ).reindex(index=shared_genes, columns=DATASETS)
        if pivoted.isna().any().any():
            raise AssertionError(f"Unexpected missing predictions: {model}")
        score_matrices[model] = pivoted.to_numpy(dtype=np.float64)
    return shared_genes, y, score_matrices


def prepare_auc_counter(scores, pos_indices, neg_indices):
    """Precompute exact Mann-Whitney / tie-corrected AUROC components.

    Weighted AUROC for resampled gene multiplicities equals sklearn AUROC
    on the expanded bootstrap sample, including tied scores.
    """
    pos_score = scores[pos_indices]
    neg_score = scores[neg_indices]
    neg_order = np.argsort(neg_score, kind="mergesort")
    neg_sorted = neg_score[neg_order]
    left = np.searchsorted(neg_sorted, pos_score, side="left")
    right = np.searchsorted(neg_sorted, pos_score, side="right")
    return neg_order, left, right


def weighted_auc_batch(w_pos, w_neg, counters, n_pos, n_neg):
    order, left, right = counters
    cumulative = np.cumsum(w_neg[:, order], axis=1, dtype=np.int64)
    prefix = np.concatenate([
        np.zeros((len(w_neg), 1), dtype=np.int64), cumulative
    ], axis=1)
    less = prefix[:, left]
    equal = prefix[:, right] - less
    contribution = less + 0.5 * equal
    return np.sum(w_pos * contribution, axis=1) / float(n_pos * n_neg)


def main():
    if B < 1000:
        raise ValueError("Use at least 1000 bootstrap iterations for a 95% CI.")
    genes, y, matrices = load_aligned_data(CSV_PATH)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    # Point estimate: per-profile AUROC, then equal-weight mean of 8 values.
    observed = {}
    rows_per_profile = []
    counters = {}
    for model in MODELS:
        aucs = []
        for j, dataset in enumerate(DATASETS):
            score = matrices[model][:, j]
            auc_val = roc_auc_score(y, score)
            aucs.append(auc_val)
            rows_per_profile.append({
                "model": model, "dataset": dataset,
                "AUROC": auc_val,
                "AP": average_precision_score(y, score),
            })
            counters[(model, j)] = prepare_auc_counter(score, pos_idx, neg_idx)
        observed[model] = np.mean(aucs)

    # Independent sanity check of optimized formula against sklearn with
    # sample multiplicities (the same weights are applied to all 8 profiles).
    check_rng = np.random.default_rng(93)
    wp = check_rng.multinomial(n_pos, np.full(n_pos, 1 / n_pos), size=1)
    wn = check_rng.multinomial(n_neg, np.full(n_neg, 1 / n_neg), size=1)
    check_weights = np.empty(len(y), dtype=np.int64)
    check_weights[pos_idx] = wp[0]
    check_weights[neg_idx] = wn[0]
    for model in MODELS:
        for j, dataset in enumerate(DATASETS):
            fast = weighted_auc_batch(wp, wn, counters[(model, j)], n_pos, n_neg)[0]
            reference = roc_auc_score(
                y, matrices[model][:, j], sample_weight=check_weights
            )
            if not np.isclose(fast, reference, atol=1e-12, rtol=0):
                raise AssertionError(f"Bootstrap AUROC math differs from sklearn: {model} {dataset}")

    print(f"Input CSV: {CSV_PATH}")
    print(f"Shared genes: {len(genes)} | CD: {n_pos} | NG: {n_neg}")
    print(f"Models: {len(MODELS)} | profiles: {len(DATASETS)}")
    print(f"Bootstrap: stratified/paired gene resampling, B={B}, seed={SEED}")
    print("AUROC estimator: arithmetic mean of 8 profile-specific AUROCs")
    print("Weighted AUROC identity checked against sklearn on all 48 combinations.")

    # Abort before reporting CI if source/definition differs from already
    # verified S6 AUROC means in the current benchmark.
    if EXPECTED_GENES == 1453:
        expected_rounded = {
            "OriGENE_Mut": 0.786380,
            "OriGENE": 0.778059,
            "DORGE_all_ElasticNet": 0.694649,
            "DORGE_epi_ElasticNet": 0.700016,
            "DORGE_mut_ElasticNet": 0.668145,
            "20/20+": 0.664939,
        }
        for model, target in expected_rounded.items():
            if abs(observed[model] - target) > 0.000002:
                raise AssertionError(
                    f"Unexpected AUROC mean for {model}: {observed[model]:.8f} "
                    f"(expected near {target:.6f}). Check source/version."
                )

    rng = np.random.default_rng(SEED)
    bootstrap_values = np.empty((B, len(MODELS)), dtype=np.float64)
    for start in range(0, B, BATCH_SIZE):
        stop = min(B, start + BATCH_SIZE)
        b = stop - start
        # Resample n_pos CD genes and n_neg NG genes with replacement.
        # Reuse the SAME weights for all models and all profiles.
        wp = rng.multinomial(n_pos, np.full(n_pos, 1.0 / n_pos), size=b)
        wn = rng.multinomial(n_neg, np.full(n_neg, 1.0 / n_neg), size=b)
        for m_idx, model in enumerate(MODELS):
            values = np.zeros(b, dtype=np.float64)
            for j in range(len(DATASETS)):
                values += weighted_auc_batch(
                    wp, wn, counters[(model, j)], n_pos, n_neg
                )
            bootstrap_values[start:stop, m_idx] = values / len(DATASETS)

    summary_rows = []
    latex_lines = []
    for m_idx, model in enumerate(MODELS):
        lo, hi = np.quantile(bootstrap_values[:, m_idx], [0.025, 0.975])
        summary_rows.append({
            "model": model,
            "model_table_label": LABELS[model],
            "n_genes": len(genes),
            "n_cd": n_pos,
            "n_ng": n_neg,
            "n_profiles": len(DATASETS),
            "AUROC_mean": observed[model],
            "CI95_low": lo,
            "CI95_high": hi,
            "bootstrap_sd": bootstrap_values[:, m_idx].std(ddof=1),
            "n_bootstrap": B,
            "seed": SEED,
        })
        latex_lines.append(
            f"{LABELS[model]}: \\multirow{{2}}{{*}}{{[{lo:.3f},\\,{hi:.3f}]}}"
        )

    result = pd.DataFrame(summary_rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTDIR / "S6_AUROC_bootstrap_CI95.csv", index=False)
    pd.DataFrame(rows_per_profile).to_csv(
        OUTDIR / "S6_AUROC_AP_by_profile.csv", index=False
    )
    pd.DataFrame(bootstrap_values, columns=MODELS).to_csv(
        OUTDIR / "S6_bootstrap_replicate_mean_AUROCs.csv", index=False
    )
    (OUTDIR / "S6_CI95_LaTeX.txt").write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "input_csv": str(CSV_PATH),
        "input_sha256": file_sha256(CSV_PATH),
        "bootstrap_unit": "gene identity; same class-stratified gene weights across all profiles and models",
        "ci_type": "95% percentile bootstrap of arithmetic mean of profile AUROCs",
        "interpretation": "conditional on observed patient profiles; no patient resampling",
        "dataset_order": DATASETS,
        "models": MODELS,
        "n_shared_genes": len(genes),
        "CD": n_pos,
        "NG": n_neg,
        "B": B,
        "seed": SEED,
    }
    (OUTDIR / "S6_bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\n=== UPDATED S6 AUROC 95% BOOTSTRAP CI ===")
    print(result[["model_table_label", "AUROC_mean", "CI95_low", "CI95_high"]].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"
    ))
    print("\n=== READY TO PASTE IN LATEX CI COLUMN ===")
    print("\n".join(latex_lines))
    print(f"\nSaved all outputs: {OUTDIR}")
    return result


if __name__ == "__main__":
    s6_bootstrap_result = main()
