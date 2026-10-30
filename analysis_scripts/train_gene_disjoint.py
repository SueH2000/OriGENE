#!/usr/bin/env python3
"""
Train the 40-kbp paired OriGENE gene-disjoint evaluation used for Figure 7.

Design principles
-----------------
1. Preserve the original OriGENE training procedure:
   - Model_OriGENE architecture
   - StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
   - fresh model for each fold
   - 100 epochs by default
   - batch size 53
   - shuffle=False in model.fit
   - Adam learning rate 4e-4
   - binary cross-entropy
   - ModelCheckpoint monitored on training loss, save_best_only=True

2. Add explicit gene-level leakage controls without changing the split logic:
   - one fixed fold manifest per gene set
   - train genes and held-out genes are disjoint in every fold
   - all five held-out folds cover every gene exactly once
   - target data are selected by gene identity, never blindly by array index
   - cross-patient target genes are exactly the source fold's held-out genes
   - source train genes ∩ target test genes must be empty

3. Replicate the original distinction between within- and cross-patient prediction:
   - within-patient predictions use the in-memory model after the final epoch
   - cross-patient predictions use the best training-loss checkpoint for that source fold

4. Save audit-ready metrics, predictions, models, histories, and figures in
   clearly named folders.

The script is restartable. Completed folds receive an _SUCCESS marker and are skipped
unless --force is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc as skl_auc,
)
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
import keras

from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    AveragePooling1D,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam


# ============================================================================
# Original OriGENE constants
# ============================================================================

N_SPLITS = 5
FOLD_RANDOM_STATE = 4
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 53
LEARNING_RATE = 4e-4
THRESHOLD = 0.5

# Fixed Figure 7 gene-set and dataset settings.
GENE_SET_CONFIG = [{'set_id': 'original_ng',
  'diagnostic': 'results/reviewer_gene_sets/original_legacy/original_ng_diagnostic.tsv',
  'expected_genes': '1454',
  'expected_cd': '727',
  'expected_ng': '727'}]

DATASET_CONFIG = [{'dataset_id': 'Lung_I',
  'group_id': 'GSE67471',
  'pair_id': 'Lung_I',
  'source_enabled': '1',
  'target_enabled': '1',
  'fold_reference': '1',
  'description': 'Original GSE67471 Lung patient I'},
 {'dataset_id': 'Lung_II',
  'group_id': 'GSE67471',
  'pair_id': 'Lung_II',
  'source_enabled': '1',
  'target_enabled': '1',
  'fold_reference': '0',
  'description': 'Original GSE67471 Lung patient II'},
 {'dataset_id': 'Liver_I',
  'group_id': 'GSE67471',
  'pair_id': 'Liver_I',
  'source_enabled': '1',
  'target_enabled': '1',
  'fold_reference': '0',
  'description': 'Original GSE67471 Liver patient I'},
 {'dataset_id': 'Liver_II',
  'group_id': 'GSE67471',
  'pair_id': 'Liver_II',
  'source_enabled': '1',
  'target_enabled': '1',
  'fold_reference': '0',
  'description': 'Original GSE67471 Liver patient II'},
 {'dataset_id': 'Colorectal_T2N2',
  'group_id': 'GSE156613_T2N2',
  'pair_id': 'Colorectal_T2N2',
  'source_enabled': '0',
  'target_enabled': '1',
  'fold_reference': '0',
  'description': 'External colorectal T2/N2 pair'},
 {'dataset_id': 'Colorectal_T3N3',
  'group_id': 'GSE156613_T3N3',
  'pair_id': 'Colorectal_T3N3',
  'source_enabled': '0',
  'target_enabled': '1',
  'fold_reference': '0',
  'description': 'External colorectal T3/N3 pair'},
 {'dataset_id': 'Gastric_2000639',
  'group_id': 'GSE51776_2000639',
  'pair_id': 'Gastric_2000639',
  'source_enabled': '0',
  'target_enabled': '1',
  'fold_reference': '0',
  'description': 'Gastric matched normal/tumour pair 2000639'},
 {'dataset_id': 'Gastric_2000721',
  'group_id': 'GSE51776_2000721',
  'pair_id': 'Gastric_2000721',
  'source_enabled': '0',
  'target_enabled': '1',
  'fold_reference': '0',
  'description': 'Gastric matched normal/tumour pair 2000721'}]


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Automated OriGENE 5-fold + cross-patient reviewer analysis"
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="epigenetic_snakemake_pipline root (default: cwd)",
    )
    p.add_argument(
        "--preprocess-root",
        type=Path,
        default=Path("results/06_legacy_preprocessing/batch"),
        help="Root containing model_ready preprocessing outputs",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/07_model_evaluation/reviewer_original_ng"),
        help="Training/evaluation output root",
    )
    p.add_argument(
        "--gene-sets",
        nargs="+",
        default=["original_ng"],
        help="Gene-set IDs, or 'configured' for all built-in gene sets",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=["configured"],
        help="Source dataset IDs, or 'configured' for source_enabled=1",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        default=["configured"],
        help="Target dataset IDs, or 'configured' for target_enabled=1",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Epochs (default {DEFAULT_EPOCHS})",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap replicates for pooled AUROC/AUPRC CI (default 1000; 0 disables)",
    )
    p.add_argument(
        "--ci-level",
        type=float,
        default=0.90,
        help="Bootstrap CI level (default 0.90, matching original thesis)",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configs, arrays, labels and fold compatibility without training",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed folds",
    )
    return p.parse_args()


def resolve_under(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else root / path


# ============================================================================
# Exact original OriGENE architecture
# ============================================================================

def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    reg_value = 1e-2

    conv1 = Conv1D(
        f1, kernel_size=1, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(reg_value)
    )(layer_in)

    conv3 = Conv1D(
        f2_in, kernel_size=1, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(reg_value)
    )(layer_in)
    conv3 = Conv1D(
        f2_out, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(reg_value)
    )(conv3)

    conv5 = Conv1D(
        f3_in, kernel_size=1, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(reg_value)
    )(layer_in)
    conv5 = Conv1D(
        f3_out, kernel_size=5, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(reg_value)
    )(conv5)

    pool = MaxPooling1D(pool_size=3, strides=1, padding="same")(layer_in)
    pool = Conv1D(
        f4_out, kernel_size=1, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(reg_value)
    )(pool)

    return concatenate([conv1, conv3, conv5, pool], axis=-1)


def Model_OriGENE(input_shape):
    kernel_size = 3

    signal = Input(shape=input_shape, dtype=np.float32, name="signal")
    x = signal

    x = AveragePooling1D(pool_size=20)(x)
    x = Conv1D(
        4, kernel_size, activation="relu", strides=2,
        kernel_regularizer=regularizers.l2(1e-2)
    )(x)
    x = Conv1D(
        8, kernel_size, activation="relu", strides=2,
        kernel_regularizer=regularizers.l2(1e-2)
    )(x)

    x = AveragePooling1D(pool_size=2)(x)

    x = inception_module(x, f1=2, f2_in=2, f2_out=2, f3_in=2, f3_out=2, f4_out=2)
    x = inception_module(x, f1=2, f2_in=2, f2_out=2, f3_in=2, f3_out=2, f4_out=2)

    x = MaxPooling1D(pool_size=2)(x)

    x = inception_module(x, f1=4, f2_in=4, f2_out=4, f3_in=4, f3_out=4, f4_out=4)
    x = inception_module(x, f1=4, f2_in=4, f2_out=4, f3_in=4, f3_out=4, f4_out=4)

    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)

    diagn = Dense(1, activation="sigmoid")(x)

    model = Model(signal, diagn)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


# ============================================================================
# Config / input discovery
# ============================================================================

def read_configs():
    gene_cfg = pd.DataFrame(GENE_SET_CONFIG).astype(str).fillna("")
    ds_cfg = pd.DataFrame(DATASET_CONFIG).astype(str).fillna("")

    req_gene = {"set_id", "expected_genes", "expected_cd", "expected_ng"}
    req_ds = {
        "dataset_id", "group_id", "pair_id",
        "source_enabled", "target_enabled", "fold_reference"
    }

    missing = req_gene - set(gene_cfg.columns)
    if missing:
        raise ValueError(f"Gene-set config missing: {sorted(missing)}")

    missing = req_ds - set(ds_cfg.columns)
    if missing:
        raise ValueError(f"Dataset config missing: {sorted(missing)}")

    if gene_cfg["set_id"].duplicated().any():
        raise ValueError("Duplicated set_id in gene-set config")

    if ds_cfg["dataset_id"].duplicated().any():
        raise ValueError("Duplicated dataset_id in dataset config")

    for c in ["source_enabled", "target_enabled", "fold_reference"]:
        ds_cfg[c] = pd.to_numeric(ds_cfg[c], errors="raise").astype(int)
        if not ds_cfg[c].isin([0, 1]).all():
            raise ValueError(f"{c} must contain only 0/1")

    if ds_cfg["fold_reference"].sum() != 1:
        raise ValueError("Dataset config must have exactly one fold_reference=1 row")

    return gene_cfg, ds_cfg


def choose_ids(df, id_col, requested, flag_col=None):
    available = df[id_col].tolist()

    if requested == ["configured"]:
        if flag_col is None:
            return available
        return df.loc[df[flag_col].eq(1), id_col].tolist()

    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Unknown {id_col}: {missing}; available={available}")
    return requested


def dataset_paths(project_root, preprocess_root, gene_set, row):
    base = preprocess_root / gene_set / row.group_id / "model_ready"
    stem = f"{gene_set}_{row.pair_id}"
    return {
        "X": base / f"{stem}_X.npy",
        "y": base / f"{stem}_y.npy",
        "genes": base / f"{stem}_genes.npy",
    }


def load_dataset(paths, mmap=True):
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)

    X = np.load(paths["X"], mmap_mode="r" if mmap else None)
    y = np.load(paths["y"]).astype(np.int32)
    genes = np.load(paths["genes"], allow_pickle=True).astype(str)

    if X.ndim != 3 or X.shape[1:] != (40000, 2):
        raise ValueError(f"Unexpected X shape {X.shape}: {paths['X']}")
    if len(y) != X.shape[0] or len(genes) != X.shape[0]:
        raise ValueError(f"X/y/gene size mismatch for {paths['X']}")
    if len(np.unique(genes)) != len(genes):
        raise ValueError(f"Duplicated genes in {paths['genes']}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"Non-binary labels in {paths['y']}")

    return X, y, genes


# ============================================================================
# Gene fold manifest and leakage audit
# ============================================================================

def create_or_validate_fold_manifest(
    gene_set: str,
    ds_cfg: pd.DataFrame,
    preprocess_root: Path,
    gene_out: Path,
    expected_genes: int,
    expected_cd: int,
    expected_ng: int,
):
    ref_row = ds_cfg.loc[ds_cfg["fold_reference"].eq(1)].iloc[0]
    ref_paths = dataset_paths(Path("."), preprocess_root, gene_set, ref_row)
    _, y_ref, genes_ref = load_dataset(ref_paths, mmap=True)

    if len(genes_ref) != expected_genes:
        raise ValueError(
            f"{gene_set}: expected {expected_genes} genes, found {len(genes_ref)}"
        )
    if int((y_ref == 1).sum()) != expected_cd:
        raise ValueError(f"{gene_set}: CD count mismatch")
    if int((y_ref == 0).sum()) != expected_ng:
        raise ValueError(f"{gene_set}: NG count mismatch")

    fold_dir = gene_out / "fold_assignments"
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fold_dir / "fold_assignments.tsv"

    # Recreate deterministically and compare if a prior manifest exists.
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=FOLD_RANDOM_STATE,
    )

    fold_assignment = np.zeros(len(y_ref), dtype=int)
    for fold, (_, heldout_idx) in enumerate(
        skf.split(np.zeros(len(y_ref)), y_ref), start=1
    ):
        fold_assignment[heldout_idx] = fold

    generated = pd.DataFrame(
        {
            "reference_index": np.arange(len(genes_ref), dtype=int),
            "gene": genes_ref,
            "label": y_ref,
            "test_fold": fold_assignment,
        }
    )

    assert generated["test_fold"].isin(range(1, N_SPLITS + 1)).all()
    assert generated["gene"].is_unique

    counts = generated.groupby("gene").size()
    assert (counts == 1).all()

    if manifest_path.exists():
        old = pd.read_csv(manifest_path, sep="\t")
        cols = ["reference_index", "gene", "label", "test_fold"]

        missing_cols = [c for c in cols if c not in old.columns]
        if missing_cols:
            raise RuntimeError(
                f"Existing fold manifest is missing required columns "
                f"{missing_cols}: {manifest_path}"
            )

        # pandas.DataFrame.equals() is dtype-sensitive.  A manifest written
        # to TSV and read back may promote integer dtypes (for example
        # int32/int8 -> int64) even when every value is identical.  Normalize
        # types before comparing so reruns validate values, not serialization
        # dtypes.
        old_cmp = old[cols].copy().reset_index(drop=True)
        new_cmp = generated[cols].copy().reset_index(drop=True)

        for c in ["reference_index", "label", "test_fold"]:
            old_cmp[c] = pd.to_numeric(old_cmp[c], errors="raise").astype(np.int64)
            new_cmp[c] = pd.to_numeric(new_cmp[c], errors="raise").astype(np.int64)

        old_cmp["gene"] = old_cmp["gene"].astype(str)
        new_cmp["gene"] = new_cmp["gene"].astype(str)

        if not old_cmp.equals(new_cmp):
            # Report concrete value-level differences rather than a generic
            # mismatch.  This remains a hard stop if the actual fold assignment
            # or gene/label identity changed.
            if len(old_cmp) != len(new_cmp):
                detail = (
                    f"row-count differs: existing={len(old_cmp)}, "
                    f"regenerated={len(new_cmp)}"
                )
            else:
                neq = (old_cmp != new_cmp).any(axis=1)
                bad_idx = np.flatnonzero(neq.to_numpy())
                preview = []
                for i in bad_idx[:10]:
                    preview.append(
                        {
                            "row": int(i),
                            "existing": old_cmp.iloc[i].to_dict(),
                            "regenerated": new_cmp.iloc[i].to_dict(),
                        }
                    )
                detail = (
                    f"value-level differences in {len(bad_idx)} rows; "
                    f"first differences={preview}"
                )

            raise RuntimeError(
                f"Existing fold manifest differs from deterministic regeneration: "
                f"{manifest_path}; {detail}"
            )
    else:
        generated.to_csv(manifest_path, sep="\t", index=False)

    # Fold-level audit files.
    audit_rows = []
    all_heldout = []

    for fold in range(1, N_SPLITS + 1):
        train = generated.loc[generated["test_fold"].ne(fold)].copy()
        heldout = generated.loc[generated["test_fold"].eq(fold)].copy()

        overlap = set(train["gene"]) & set(heldout["gene"])
        if overlap:
            raise RuntimeError(f"{gene_set} fold {fold}: leakage {sorted(overlap)[:20]}")

        train.to_csv(
            fold_dir / f"fold_{fold:02d}_train_genes.tsv",
            sep="\t", index=False
        )
        heldout.to_csv(
            fold_dir / f"fold_{fold:02d}_heldout_genes.tsv",
            sep="\t", index=False
        )

        all_heldout.extend(heldout["gene"].tolist())

        audit_rows.append(
            {
                "gene_set": gene_set,
                "fold": fold,
                "n_train": len(train),
                "n_heldout": len(heldout),
                "train_cd": int((train["label"] == 1).sum()),
                "train_ng": int((train["label"] == 0).sum()),
                "heldout_cd": int((heldout["label"] == 1).sum()),
                "heldout_ng": int((heldout["label"] == 0).sum()),
                "train_heldout_overlap": len(overlap),
            }
        )

    if len(all_heldout) != len(genes_ref):
        raise RuntimeError("Five held-out folds do not contain N genes total")
    if len(set(all_heldout)) != len(genes_ref):
        raise RuntimeError("A gene appears in held-out data more than once")
    if set(all_heldout) != set(genes_ref):
        raise RuntimeError("Five held-out folds do not cover the full gene universe")

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(
        fold_dir / "fold_leakage_audit.tsv",
        sep="\t", index=False
    )

    # Verify every configured dataset has exactly the same gene identities and labels.
    ref_label_map = dict(zip(genes_ref, y_ref))
    compatibility_rows = []

    for row in ds_cfg.itertuples(index=False):
        paths = dataset_paths(Path("."), preprocess_root, gene_set, row)
        _, y, genes = load_dataset(paths, mmap=True)

        if set(genes) != set(genes_ref):
            missing = sorted(set(genes_ref) - set(genes))
            extra = sorted(set(genes) - set(genes_ref))
            raise RuntimeError(
                f"{gene_set}/{row.dataset_id}: gene universe differs; "
                f"missing={missing[:20]} extra={extra[:20]}"
            )

        label_map = dict(zip(genes, y))
        disagreement = [
            g for g in genes_ref
            if int(label_map[g]) != int(ref_label_map[g])
        ]
        if disagreement:
            raise RuntimeError(
                f"{gene_set}/{row.dataset_id}: label disagreement "
                f"for {disagreement[:20]}"
            )

        compatibility_rows.append(
            {
                "gene_set": gene_set,
                "dataset_id": row.dataset_id,
                "n_genes": len(genes),
                "same_gene_universe": True,
                "same_labels_by_gene": True,
            }
        )

    pd.DataFrame(compatibility_rows).to_csv(
        fold_dir / "dataset_gene_compatibility.tsv",
        sep="\t", index=False
    )

    return generated


def gene_indices_for_order(genes_array: np.ndarray, requested_genes: list[str]) -> np.ndarray:
    mapping = {g: i for i, g in enumerate(genes_array)}
    missing = [g for g in requested_genes if g not in mapping]
    if missing:
        raise RuntimeError(f"Missing requested genes: {missing[:20]}")
    return np.asarray([mapping[g] for g in requested_genes], dtype=int)


# ============================================================================
# Metrics
# ============================================================================

def threshold_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def probability_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
    }


def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return float(thresholds[idx])


def bootstrap_ci(y_true, y_prob, n_boot=1000, ci_level=0.90, seed=20260827):
    if n_boot <= 0:
        return {}

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    aucs = []
    aps = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
        aps.append(average_precision_score(yb, pb))

    alpha = (1.0 - ci_level) / 2.0

    out = {}
    for name, vals in [("auroc", aucs), ("auprc", aps)]:
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            continue
        out[f"{name}_ci_level"] = ci_level
        out[f"{name}_ci_low"] = float(np.quantile(vals, alpha))
        out[f"{name}_ci_high"] = float(np.quantile(vals, 1 - alpha))
        out[f"{name}_bootstrap_n"] = int(len(vals))
    return out


def full_metrics(y_true, y_prob, threshold=THRESHOLD, bootstrap=0, ci_level=0.90):
    out = {}
    out.update(probability_metrics(y_true, y_prob))
    out.update(threshold_metrics(y_true, y_prob, threshold))
    out.update(bootstrap_ci(y_true, y_prob, bootstrap, ci_level))
    return out


# ============================================================================
# Plotting
# ============================================================================

def save_training_history(history_df: pd.DataFrame, path: Path, epochs: int):
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax2 = ax1.twinx()

    ax1.plot(history_df["epoch"], history_df["loss"], label="Loss")
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
    ax2.plot(history_df["epoch"], history_df["accuracy"], label="Accuracy")
    ax2.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation Accuracy")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_xlim(1, max(epochs, 2))

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc(y_true, y_prob, path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUROC = {score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pr(y_true, y_prob, path: Path, title: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    score = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"AUPRC = {score:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion(y_true, y_prob, path: Path, title: str, threshold=THRESHOLD):
    y_pred = (np.asarray(y_prob) > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1], labels=["NG", "CD"])
    ax.set_yticks([0, 1], labels=["NG", "CD"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metric_matrix(matrix: pd.DataFrame, path: Path, title: str):
    arr = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(matrix.columns)),
                                    max(4, 0.7 * len(matrix.index))))
    im = ax.imshow(arr, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Test target")
    ax.set_ylabel("Training source")
    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)



# ============================================================================
# Source-level unified cross-patient ROC / PR plots
# EXACTLY replicates the user's original plotting layout/style.
# ============================================================================

PRETTY_DATASET_LABELS = {
    "Lung_I": "Lung I",
    "Lung_II": "Lung II",
    "Liver_I": "Liver I",
    "Liver_II": "Liver II",
    "Colorectal_T2N2": "Colorecta I",
    "Colorectal_T3N3": "Colorecta II",
    "Gastric_2000639": "Gastric I",
    "Gastric_2000721": "Gastric II",
}

# Exact color sequence from the original plotting cell.
UNIFIED_CURVE_COLORS = [
    "cornflowerblue",
    "darkgreen",
    "limegreen",
    "darkred",
    "orange",
    "gold",
    "navajowhite",
]


def pretty_dataset_label(dataset_id: str) -> str:
    # Preserve the original formatting behavior as closely as possible.
    if dataset_id in PRETTY_DATASET_LABELS:
        return PRETTY_DATASET_LABELS[dataset_id]
    if "_" in dataset_id:
        a, b = dataset_id.split("_", 1)
        return "{:8s} {:5s}".format(a, b)
    return dataset_id


def save_unified_cross_patient_plots(
    source: str,
    targets: list[str],
    source_out: Path,
):
    """
    Reproduce the original unified cross-patient ROC/AUPRC plotting cell.

    Data source:
      ONLY the fresh, fold-aligned predictions_all_folds.tsv files generated
      by the current run for this source. No legacy Cross_patient/New files.

    Layout/style intentionally matches the user's original code:
      - LaTeX text rendering enabled
      - serif font
      - figsize=(9, 3.8)
      - 1x2 GridSpec, wspace=.3, hspace=2.
      - linewidth=0.7
      - navy dashed ROC diagonal
      - exact axis limits/labels
      - exact legend placement bbox_to_anchor=(2.65, .5)
      - exact label formatting
      - exact color order
      - PNG 300 dpi + PDF
    """
    # Exact plotting rcParams from the original cell.
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    plot_datasets = [t for t in targets if t != source]
    csvs = []
    missing = []

    # Build the same [dataset, label, score] structure as the original cell,
    # but from this run's fresh fold-aligned TSVs.
    for ds in plot_datasets:
        f = (
            source_out / "cross_patient" / ds /
            "predictions_all_folds.tsv"
        )

        if f.exists():
            df = pd.read_csv(f, sep="\t")

            needed = {"y_true", "y_prob"}
            if not needed.issubset(df.columns):
                raise ValueError(f"{f} missing one of {needed}")

            # Refuse to plot any file that fails the leakage flag.
            if "source_train_seen" in df.columns:
                if df["source_train_seen"].astype(bool).any():
                    raise RuntimeError(
                        f"Leakage flag present in fresh predictions: {source}->{ds}"
                    )

            tmp = pd.DataFrame({
                "dataset": ds,
                "label": df["y_true"].to_numpy(dtype=int),
                "score": df["y_prob"].to_numpy(dtype=float),
            })
            csvs.append(tmp)
        else:
            missing.append(str(f))

    if missing:
        print("⚠️ Missing fresh prediction TSVs (will skip in plot):", flush=True)
        for m in missing:
            print("   ", m, flush=True)

    if not csvs:
        raise RuntimeError(
            "No fresh cross-patient prediction TSVs found; nothing to plot."
        )

    df_all = (
        pd.concat(csvs, ignore_index=True)
        .dropna(subset=["dataset", "label", "score"])
    )

    unified_dir = source_out / "cross_patient" / "unified"
    unified_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact fresh plotting table used for this figure.
    df_all.to_csv(
        unified_dir / "cross_patient_predictions_all_targets.csv",
        index=False,
    )

    # Same sanity check as the original plotting cell.
    print("\nSanity check from loaded fresh prediction files:", flush=True)
    sanity_rows = []

    for ds in plot_datasets:
        g = df_all[df_all["dataset"] == ds]
        if g.empty:
            continue

        y = g["label"].to_numpy(dtype=int)
        p = g["score"].to_numpy(dtype=float)

        auroc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)

        print(
            f"  {ds:20s}  AUROC={auroc:.4f}  "
            f"AUPRC={auprc:.4f}  N={len(g)}",
            flush=True,
        )

        sanity_rows.append({
            "source": source,
            "target": ds,
            "AUROC": auroc,
            "AUPRC_average_precision": auprc,
            "N": len(g),
        })

    pd.DataFrame(sanity_rows).to_csv(
        unified_dir / "cross_patient_curve_metrics.csv",
        index=False,
    )

    # Exact order/color behavior from the original cell.
    plot_order = plot_datasets
    color_list = [
        "cornflowerblue",
        "darkgreen",
        "limegreen",
        "darkred",
        "orange",
        "gold",
        "navajowhite",
    ]

    curves = []

    for i, ds in enumerate(plot_order):
        g = df_all[df_all["dataset"] == ds]
        if g.empty:
            continue

        y = g["label"].to_numpy(dtype=int)
        p = g["score"].to_numpy(dtype=float)

        if y.min() == y.max():
            continue

        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)

        curves.append({
            "name": pretty_dataset_label(ds),
            "fpr": fpr,
            "tpr": tpr,
            "rec": rec,
            "prec": prec,
            "roc_auc": skl_auc(fpr, tpr),
            # Intentionally preserve the original plotting-cell definition.
            "pr_auc": skl_auc(rec, prec),
            "color": color_list[i % len(color_list)],
        })

    # ----------------------------------------------------------------------
    # EXACT original example layout/style
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(9, 3.8))
    grid = fig.add_gridspec(1, 2, wspace=.3, hspace=2.)
    lw = 0.7

    # ROC (left)
    ax = fig.add_subplot(grid[0, 0])
    ax.plot(
        [0, 1], [0, 1],
        color="navy",
        lw=lw,
        linestyle="--",
    )

    for c in curves:
        label = "{:10s}(AUC ={:.2f})".format(
            c["name"],
            c["roc_auc"],
        )
        ax.plot(
            c["fpr"],
            c["tpr"],
            lw=lw,
            label=label,
            color=c["color"],
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    # PR (right)
    ax = fig.add_subplot(grid[0, 1])

    for c in curves:
        label = "{:10s}( AUC = {:.2f} - AUPRC ={:.2f})".format(
            c["name"],
            c["roc_auc"],
            c["pr_auc"],
        )
        ax.plot(
            c["rec"],
            c["prec"],
            lw=lw,
            label=label,
            color=c["color"],
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # Exact legend placement from the original cell.
    ax.legend(
        loc="right",
        bbox_to_anchor=(2.65, .5),
        ncol=1,
        fancybox=True,
        shadow=True,
        fontsize=14,
    )

    out_stem = (
        unified_dir /
        f"{source}_Cross-patient_ROC_AUPRC_5kbp_epi_N"
    )

    # Keep plotting failure isolated from the completed model analysis:
    # LaTeX is required for exact rendering. If unavailable, predictions and
    # metrics remain saved and this exact plotting function can be rerun later.
    try:
        fig.savefig(
            str(out_stem) + ".png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            str(out_stem) + ".pdf",
            bbox_inches="tight",
        )
        print("\nSaved:", str(out_stem) + ".png", flush=True)
        print("Saved:", str(out_stem) + ".pdf", flush=True)
    except Exception as exc:
        print(
            "\n[plot warning] Exact LaTeX figure could not be rendered on "
            f"this machine: {exc!r}",
            flush=True,
        )
        print(
            "[plot warning] Fresh plotting CSVs were saved; rerun the same "
            "plotting function on a machine with LaTeX installed.",
            flush=True,
        )
    finally:
        plt.close(fig)


# ============================================================================
# Prediction output helpers
# ============================================================================

def prediction_frame(source, target, fold, genes, y_true, y_prob, evaluation_type):
    return pd.DataFrame(
        {
            "source": source,
            "target": target,
            "evaluation_type": evaluation_type,
            "fold": int(fold),
            "gene": np.asarray(genes, dtype=str),
            "y_true": np.asarray(y_true, dtype=int),
            "y_prob": np.asarray(y_prob, dtype=float),
            "y_pred_0.5": (np.asarray(y_prob) > THRESHOLD).astype(int),
            "source_train_seen": False,
        }
    )


def aggregate_prediction_files(files: list[Path], expected_n: int):
    frames = [pd.read_csv(p, sep="\t") for p in files]
    out = pd.concat(frames, ignore_index=True)

    if len(out) != expected_n:
        raise RuntimeError(
            f"Aggregated predictions contain {len(out)} rows; expected {expected_n}"
        )
    if out["gene"].duplicated().any():
        dup = out.loc[out["gene"].duplicated(keep=False), "gene"].tolist()
        raise RuntimeError(f"Gene predicted more than once: {dup[:20]}")
    if out["source_train_seen"].astype(bool).any():
        raise RuntimeError("Leakage audit failed: source_train_seen=True exists")

    return out


# ============================================================================
# One gene set
# ============================================================================

def run_gene_set(
    gene_set: str,
    expected_genes: int,
    expected_cd: int,
    expected_ng: int,
    ds_cfg: pd.DataFrame,
    sources: list[str],
    targets: list[str],
    preprocess_root: Path,
    output_root: Path,
    epochs: int,
    batch_size: int,
    bootstrap: int,
    ci_level: float,
    check_only: bool,
    force: bool,
):
    gene_out = output_root / gene_set
    gene_out.mkdir(parents=True, exist_ok=True)

    manifest = create_or_validate_fold_manifest(
        gene_set=gene_set,
        ds_cfg=ds_cfg,
        preprocess_root=preprocess_root,
        gene_out=gene_out,
        expected_genes=expected_genes,
        expected_cd=expected_cd,
        expected_ng=expected_ng,
    )

    if check_only:
        print(
            f"[CHECK] {gene_set}: {expected_genes} genes; "
            f"sources={sources}; targets={targets}; PASS",
            flush=True,
        )
        return

    ds_lookup = {
        row.dataset_id: row
        for row in ds_cfg.itertuples(index=False)
    }

    for source in sources:
        if source not in ds_lookup:
            raise ValueError(source)

        source_row = ds_lookup[source]
        source_paths = dataset_paths(Path("."), preprocess_root, gene_set, source_row)
        X_source, y_source, genes_source = load_dataset(source_paths, mmap=True)
        source_label_map = dict(zip(genes_source, y_source))

        source_out = gene_out / "sources" / source
        source_out.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 90, flush=True)
        print(f"GENE SET: {gene_set} | TRAIN SOURCE: {source}", flush=True)
        print("=" * 90, flush=True)

        # --------------------------------------------------------------------
        # Train five folds and make within/cross predictions
        # --------------------------------------------------------------------
        for fold in range(1, N_SPLITS + 1):
            fold_out = source_out / "folds" / f"fold_{fold:02d}"
            fold_out.mkdir(parents=True, exist_ok=True)
            success = fold_out / "_SUCCESS"

            if success.exists() and not force:
                print(f"[skip] {gene_set}/{source}/fold_{fold:02d}", flush=True)
                continue

            if success.exists() and force:
                success.unlink()

            train_manifest = manifest.loc[manifest["test_fold"].ne(fold)].copy()
            heldout_manifest = manifest.loc[manifest["test_fold"].eq(fold)].copy()

            train_genes = train_manifest["gene"].tolist()
            heldout_genes = heldout_manifest["gene"].tolist()

            overlap = set(train_genes) & set(heldout_genes)
            if overlap:
                raise RuntimeError(
                    f"{gene_set}/{source}/fold{fold}: source leakage {sorted(overlap)[:20]}"
                )

            train_idx = gene_indices_for_order(genes_source, train_genes)
            val_idx = gene_indices_for_order(genes_source, heldout_genes)

            # Explicitly verify labels after mapping.
            if not np.array_equal(
                y_source[train_idx],
                train_manifest["label"].to_numpy(dtype=int),
            ):
                raise RuntimeError("Source training labels do not match fold manifest")

            if not np.array_equal(
                y_source[val_idx],
                heldout_manifest["label"].to_numpy(dtype=int),
            ):
                raise RuntimeError("Source held-out labels do not match fold manifest")

            # Save fold gene manifests adjacent to the model.
            train_manifest.to_csv(
                fold_out / "train_genes.tsv", sep="\t", index=False
            )
            heldout_manifest.to_csv(
                fold_out / "heldout_genes.tsv", sep="\t", index=False
            )

            leakage_fold = pd.DataFrame(
                [{
                    "gene_set": gene_set,
                    "source": source,
                    "fold": fold,
                    "n_train": len(train_genes),
                    "n_heldout": len(heldout_genes),
                    "source_train_heldout_overlap": len(overlap),
                }]
            )
            leakage_fold.to_csv(
                fold_out / "leakage_audit.tsv", sep="\t", index=False
            )

            print(
                f"[train] {gene_set} | {source} | fold {fold}/5 | "
                f"train={len(train_idx)} heldout={len(val_idx)}",
                flush=True,
            )

            # Materialize only this fold in float32.
            xtrn = np.asarray(X_source[train_idx], dtype=np.float32)
            ytrn = np.asarray(y_source[train_idx], dtype=np.int32)
            xval = np.asarray(X_source[val_idx], dtype=np.float32)
            yval = np.asarray(y_source[val_idx], dtype=np.int32)

            tf.keras.backend.clear_session()
            model = Model_OriGENE(xtrn.shape[1:])

            checkpoint = fold_out / "best_training_loss.weights.h5"
            callback = ModelCheckpoint(
                filepath=str(checkpoint),
                monitor="loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
            )

            history = model.fit(
                [xtrn],
                ytrn,
                validation_data=([xval], yval),
                epochs=epochs,
                shuffle=False,
                batch_size=batch_size,
                callbacks=[callback],
                verbose=2,
            )

            # History / original-style learning curve.
            history_df = pd.DataFrame(history.history)
            history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
            history_df.to_csv(
                fold_out / "training_history.tsv", sep="\t", index=False
            )
            save_training_history(
                history_df,
                fold_out / "training_history.png",
                epochs=epochs,
            )

            # --------------------------------------------------------------
            # WITHIN-PATIENT: original code predicts with final in-memory model
            # --------------------------------------------------------------
            within_prob = model.predict([xval], batch_size=batch_size, verbose=0).flatten()

            within_df = prediction_frame(
                source=source,
                target=source,
                fold=fold,
                genes=heldout_genes,
                y_true=yval,
                y_prob=within_prob,
                evaluation_type="within_patient_final_epoch",
            )
            within_df.to_csv(
                fold_out / f"within_{source}_predictions.tsv",
                sep="\t", index=False
            )

            fold_metrics = {
                "gene_set": gene_set,
                "source": source,
                "target": source,
                "evaluation_type": "within_patient_final_epoch",
                "fold": fold,
            }
            fold_metrics.update(full_metrics(yval, within_prob, THRESHOLD, 0, ci_level))
            pd.DataFrame([fold_metrics]).to_csv(
                fold_out / f"within_{source}_metrics.tsv",
                sep="\t", index=False
            )

            save_roc(
                yval, within_prob,
                fold_out / f"within_{source}_ROC.png",
                f"{source} → {source} | Fold {fold}"
            )
            save_confusion(
                yval, within_prob,
                fold_out / f"within_{source}_confusion.png",
                f"{source} → {source} | Fold {fold}",
                THRESHOLD
            )

            # Preserve final-epoch model separately for traceability.
            model.save(fold_out / "final_epoch_model.keras")

            # --------------------------------------------------------------
            # CROSS-PATIENT: original code loads the saved fold checkpoint
            # --------------------------------------------------------------
            if not checkpoint.exists():
                raise RuntimeError(f"Checkpoint missing after training: {checkpoint}")

            model.load_weights(checkpoint)

            cross_root = fold_out / "cross_patient"
            cross_root.mkdir(parents=True, exist_ok=True)

            for target in targets:
                if target == source:
                    continue

                target_row = ds_lookup[target]
                target_paths = dataset_paths(
                    Path("."), preprocess_root, gene_set, target_row
                )
                X_target, y_target, genes_target = load_dataset(target_paths, mmap=True)

                target_idx = gene_indices_for_order(genes_target, heldout_genes)
                target_test_genes = genes_target[target_idx].tolist()

                # Hard leakage / identity checks.
                if set(target_test_genes) != set(heldout_genes):
                    raise RuntimeError(
                        f"{source}->{target}/fold{fold}: target genes are not the "
                        "source held-out gene set"
                    )
                cross_overlap = set(train_genes) & set(target_test_genes)
                if cross_overlap:
                    raise RuntimeError(
                        f"{source}->{target}/fold{fold}: cross-patient gene leakage: "
                        f"{sorted(cross_overlap)[:20]}"
                    )

                y_target_fold = np.asarray(y_target[target_idx], dtype=np.int32)
                if not np.array_equal(
                    y_target_fold,
                    heldout_manifest["label"].to_numpy(dtype=int),
                ):
                    raise RuntimeError(
                        f"{source}->{target}/fold{fold}: target labels disagree with manifest"
                    )

                x_target_fold = np.asarray(X_target[target_idx], dtype=np.float32)
                target_prob = model.predict(
                    [x_target_fold],
                    batch_size=batch_size,
                    verbose=0,
                ).flatten()

                target_dir = cross_root / target
                target_dir.mkdir(parents=True, exist_ok=True)

                pred_df = prediction_frame(
                    source=source,
                    target=target,
                    fold=fold,
                    genes=target_test_genes,
                    y_true=y_target_fold,
                    y_prob=target_prob,
                    evaluation_type="cross_patient_best_training_loss_checkpoint",
                )
                pred_df.to_csv(
                    target_dir / "predictions.tsv",
                    sep="\t", index=False
                )

                cross_audit = pd.DataFrame(
                    [{
                        "gene_set": gene_set,
                        "source": source,
                        "target": target,
                        "fold": fold,
                        "n_source_train_genes": len(train_genes),
                        "n_target_test_genes": len(target_test_genes),
                        "source_train_target_test_overlap": len(cross_overlap),
                        "target_gene_set_equals_source_heldout": (
                            set(target_test_genes) == set(heldout_genes)
                        ),
                    }]
                )
                cross_audit.to_csv(
                    target_dir / "leakage_audit.tsv",
                    sep="\t", index=False
                )

                m = {
                    "gene_set": gene_set,
                    "source": source,
                    "target": target,
                    "evaluation_type": "cross_patient_best_training_loss_checkpoint",
                    "fold": fold,
                }
                m.update(full_metrics(y_target_fold, target_prob, THRESHOLD, 0, ci_level))
                pd.DataFrame([m]).to_csv(
                    target_dir / "metrics.tsv",
                    sep="\t", index=False
                )

                del X_target, x_target_fold

            success.write_text("OK\n")

            # Free fold-level arrays/model.
            del xtrn, xval, ytrn, yval, model
            tf.keras.backend.clear_session()

        # --------------------------------------------------------------------
        # Aggregate the five folds for this source
        # --------------------------------------------------------------------
        aggregate_source(
            gene_set=gene_set,
            source=source,
            targets=targets,
            source_out=source_out,
            expected_n=expected_genes,
            bootstrap=bootstrap,
            ci_level=ci_level,
        )

        (source_out / "_SUCCESS").write_text("OK\n")

    # ------------------------------------------------------------------------
    # Cross-source matrices and summary
    # ------------------------------------------------------------------------
    build_gene_set_summary(
        gene_set=gene_set,
        gene_out=gene_out,
        sources=sources,
        targets=targets,
    )


def aggregate_source(
    gene_set,
    source,
    targets,
    source_out,
    expected_n,
    bootstrap,
    ci_level,
):
    # Within patient
    within_files = [
        source_out / "folds" / f"fold_{fold:02d}" / f"within_{source}_predictions.tsv"
        for fold in range(1, N_SPLITS + 1)
    ]
    within = aggregate_prediction_files(within_files, expected_n)

    within_out = source_out / "within_patient"
    within_out.mkdir(parents=True, exist_ok=True)
    within.to_csv(
        within_out / "predictions_all_folds.tsv",
        sep="\t", index=False
    )

    original_fold_metrics = pd.concat(
        [
            pd.read_csv(
                source_out / "folds" / f"fold_{fold:02d}" /
                f"within_{source}_metrics.tsv",
                sep="\t"
            )
            for fold in range(1, N_SPLITS + 1)
        ],
        ignore_index=True,
    )
    original_fold_metrics.to_csv(
        within_out / "metrics_per_fold.tsv",
        sep="\t", index=False
    )

    avg_cols = [
        "accuracy", "sensitivity_recall", "specificity",
        "precision", "f1", "auroc", "auprc"
    ]
    avg_row = {
        "gene_set": gene_set,
        "source": source,
        "target": source,
        "evaluation_type": "within_patient_original_style_mean_of_5_folds",
    }
    for c in avg_cols:
        avg_row[c] = float(original_fold_metrics[c].mean())
    pd.DataFrame([avg_row]).to_csv(
        within_out / "original_style_mean_fold_metrics.tsv",
        sep="\t", index=False
    )

    pooled = {
        "gene_set": gene_set,
        "source": source,
        "target": source,
        "evaluation_type": "within_patient_pooled_oof",
    }
    pooled.update(
        full_metrics(
            within["y_true"], within["y_prob"],
            THRESHOLD, bootstrap, ci_level
        )
    )
    pd.DataFrame([pooled]).to_csv(
        within_out / "pooled_metrics_threshold0.5.tsv",
        sep="\t", index=False
    )

    yt = youden_threshold(within["y_true"], within["y_prob"])
    youden = {
        "gene_set": gene_set,
        "source": source,
        "target": source,
        "evaluation_type": "within_patient_pooled_oof_youden",
    }
    youden.update(full_metrics(within["y_true"], within["y_prob"], yt, 0, ci_level))
    pd.DataFrame([youden]).to_csv(
        within_out / "pooled_metrics_youden.tsv",
        sep="\t", index=False
    )

    figs = within_out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    save_roc(
        within["y_true"], within["y_prob"],
        figs / "ROC_all_5_folds.png",
        f"{source} → {source} | pooled 5-fold OOF"
    )
    save_pr(
        within["y_true"], within["y_prob"],
        figs / "PR_all_5_folds.png",
        f"{source} → {source} | pooled 5-fold OOF"
    )
    save_confusion(
        within["y_true"], within["y_prob"],
        figs / "confusion_threshold0.5.png",
        f"{source} → {source} | threshold 0.5",
        THRESHOLD
    )

    # Cross-patient targets
    cross_source_root = source_out / "cross_patient"
    cross_source_root.mkdir(parents=True, exist_ok=True)

    for target in targets:
        if target == source:
            continue

        files = [
            source_out / "folds" / f"fold_{fold:02d}" /
            "cross_patient" / target / "predictions.tsv"
            for fold in range(1, N_SPLITS + 1)
        ]
        pred = aggregate_prediction_files(files, expected_n)

        target_out = cross_source_root / target
        target_out.mkdir(parents=True, exist_ok=True)

        pred.to_csv(
            target_out / "predictions_all_folds.tsv",
            sep="\t", index=False
        )

        per_fold = pd.concat(
            [
                pd.read_csv(
                    source_out / "folds" / f"fold_{fold:02d}" /
                    "cross_patient" / target / "metrics.tsv",
                    sep="\t"
                )
                for fold in range(1, N_SPLITS + 1)
            ],
            ignore_index=True,
        )
        per_fold.to_csv(
            target_out / "metrics_per_fold.tsv",
            sep="\t", index=False
        )

        avg = {
            "gene_set": gene_set,
            "source": source,
            "target": target,
            "evaluation_type": "cross_patient_mean_of_5_folds",
        }
        for c in avg_cols:
            avg[c] = float(per_fold[c].mean())
        pd.DataFrame([avg]).to_csv(
            target_out / "mean_fold_metrics.tsv",
            sep="\t", index=False
        )

        pooled = {
            "gene_set": gene_set,
            "source": source,
            "target": target,
            "evaluation_type": "cross_patient_pooled_fold_aligned",
        }
        pooled.update(
            full_metrics(
                pred["y_true"], pred["y_prob"],
                THRESHOLD, bootstrap, ci_level
            )
        )
        pd.DataFrame([pooled]).to_csv(
            target_out / "pooled_metrics_threshold0.5.tsv",
            sep="\t", index=False
        )

        yt = youden_threshold(pred["y_true"], pred["y_prob"])
        youden = {
            "gene_set": gene_set,
            "source": source,
            "target": target,
            "evaluation_type": "cross_patient_pooled_fold_aligned_youden",
        }
        youden.update(full_metrics(pred["y_true"], pred["y_prob"], yt, 0, ci_level))
        pd.DataFrame([youden]).to_csv(
            target_out / "pooled_metrics_youden.tsv",
            sep="\t", index=False
        )

        audit = pd.concat(
            [
                pd.read_csv(
                    source_out / "folds" / f"fold_{fold:02d}" /
                    "cross_patient" / target / "leakage_audit.tsv",
                    sep="\t"
                )
                for fold in range(1, N_SPLITS + 1)
            ],
            ignore_index=True,
        )
        audit.to_csv(
            target_out / "leakage_audit_all_folds.tsv",
            sep="\t", index=False
        )

        if not (audit["source_train_target_test_overlap"] == 0).all():
            raise RuntimeError(f"Leakage audit failed for {source}->{target}")
        if not audit["target_gene_set_equals_source_heldout"].astype(bool).all():
            raise RuntimeError(f"Held-out identity audit failed for {source}->{target}")

        figs = target_out / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        save_roc(
            pred["y_true"], pred["y_prob"],
            figs / "ROC_all_5_folds.png",
            f"{source} → {target} | fold-aligned"
        )
        save_pr(
            pred["y_true"], pred["y_prob"],
            figs / "PR_all_5_folds.png",
            f"{source} → {target} | fold-aligned"
        )
        save_confusion(
            pred["y_true"], pred["y_prob"],
            figs / "confusion_threshold0.5.png",
            f"{source} → {target} | threshold 0.5",
            THRESHOLD
        )



    save_unified_cross_patient_plots(
        source=source,
        targets=targets,
        source_out=source_out,
    )

def build_gene_set_summary(gene_set, gene_out, sources, targets):
    summary_dir = gene_out / "summary"
    matrix_dir = summary_dir / "matrices"
    figure_dir = summary_dir / "figures"

    matrix_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for source in sources:
        source_out = gene_out / "sources" / source

        # Diagonal within-patient result.
        within_path = source_out / "within_patient" / "pooled_metrics_threshold0.5.tsv"
        if within_path.exists():
            rows.append(pd.read_csv(within_path, sep="\t").iloc[0].to_dict())

        for target in targets:
            if target == source:
                continue
            p = (
                source_out / "cross_patient" / target /
                "pooled_metrics_threshold0.5.tsv"
            )
            if p.exists():
                rows.append(pd.read_csv(p, sep="\t").iloc[0].to_dict())

    metrics = pd.DataFrame(rows)
    metrics.to_csv(summary_dir / "metrics_long.tsv", sep="\t", index=False)

    leakage_rows = []
    for source in sources:
        source_fold_audit = (
            gene_out / "fold_assignments" / "fold_leakage_audit.tsv"
        )
        if source_fold_audit.exists():
            x = pd.read_csv(source_fold_audit, sep="\t")
            x["source"] = source
            leakage_rows.append(x)

        for target in targets:
            if target == source:
                continue
            p = (
                gene_out / "sources" / source / "cross_patient" / target /
                "leakage_audit_all_folds.tsv"
            )
            if p.exists():
                leakage_rows.append(pd.read_csv(p, sep="\t"))

    if leakage_rows:
        pd.concat(leakage_rows, ignore_index=True).to_csv(
            summary_dir / "leakage_audit_all.tsv",
            sep="\t", index=False
        )

    for metric in ["auroc", "auprc", "accuracy", "precision",
                   "sensitivity_recall", "specificity", "f1"]:
        if metric not in metrics.columns:
            continue

        matrix = (
            metrics.pivot(index="source", columns="target", values=metric)
            .reindex(index=sources, columns=targets)
        )
        matrix.to_csv(matrix_dir / f"{metric}_matrix.tsv", sep="\t")

        save_metric_matrix(
            matrix,
            figure_dir / f"{metric}_matrix.png",
            f"{gene_set}: {metric.upper()} source → target"
        )


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    preprocess_root = resolve_under(project_root, args.preprocess_root)
    output_root = resolve_under(project_root, args.output_root)

    gene_cfg, ds_cfg = read_configs()

    gene_sets = choose_ids(gene_cfg, "set_id", args.gene_sets)
    sources = choose_ids(ds_cfg, "dataset_id", args.sources, "source_enabled")
    targets = choose_ids(ds_cfg, "dataset_id", args.targets, "target_enabled")

    if not sources:
        raise ValueError("No source datasets selected")
    if not targets:
        raise ValueError("No target datasets selected")

    output_root.mkdir(parents=True, exist_ok=True)

    run_info = {
        "project_root": str(project_root),
        "preprocess_root": str(preprocess_root),
        "output_root": str(output_root),
        "gene_set_config": GENE_SET_CONFIG,
        "dataset_config": DATASET_CONFIG,
        "gene_sets": gene_sets,
        "sources": sources,
        "targets": targets,
        "n_splits": N_SPLITS,
        "fold_random_state": FOLD_RANDOM_STATE,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "fit_shuffle": False,
        "checkpoint_monitor": "loss",
        "checkpoint_save_best_only": True,
        "within_prediction_model": "final_epoch_in_memory",
        "cross_prediction_model": "best_training_loss_checkpoint",
        "bootstrap": args.bootstrap,
        "ci_level": args.ci_level,
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
    }

    with (output_root / "run_manifest.json").open("w") as handle:
        json.dump(run_info, handle, indent=2)

    print("=" * 100, flush=True)
    print("OriGENE REVIEWER AUTOMATED TRAINING", flush=True)
    print("=" * 100, flush=True)
    print("Gene sets:", gene_sets, flush=True)
    print("Sources  :", sources, flush=True)
    print("Targets  :", targets, flush=True)
    print("Epochs   :", args.epochs, flush=True)
    print("Batch    :", args.batch_size, flush=True)
    print("GPUs     :", run_info["gpu_devices"], flush=True)
    print("Output   :", output_root, flush=True)

    t0 = time.time()

    for gene_set in gene_sets:
        row = gene_cfg.loc[gene_cfg["set_id"].eq(gene_set)].iloc[0]

        run_gene_set(
            gene_set=gene_set,
            expected_genes=int(row.expected_genes),
            expected_cd=int(row.expected_cd),
            expected_ng=int(row.expected_ng),
            ds_cfg=ds_cfg,
            sources=sources,
            targets=targets,
            preprocess_root=preprocess_root,
            output_root=output_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            bootstrap=args.bootstrap,
            ci_level=args.ci_level,
            check_only=args.check_only,
            force=args.force,
        )

    if args.check_only:
        print("\nCHECK-ONLY VALIDATION: PASS", flush=True)
        return

    elapsed = time.time() - t0
    (output_root / "_SUCCESS").write_text("OK\n")

    print("\n" + "=" * 100, flush=True)
    print("ALL REQUESTED OriGENE TRAINING/EVALUATION COMPLETE", flush=True)
    print("=" * 100, flush=True)
    print(f"Elapsed hours: {elapsed / 3600:.2f}", flush=True)
    print(f"Output: {output_root}", flush=True)


if __name__ == "__main__":
    main()
