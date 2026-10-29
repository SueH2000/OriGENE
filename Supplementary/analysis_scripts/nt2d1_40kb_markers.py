"""nt2d1 40kb markers. See Supplementary/README.md."""
import os

def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc as skl_auc,
    roc_auc_score,
)

# =====================================================================
# FIXED PUBLICATION GEOMETRY
#
# IMPORTANT:
# Use EXACTLY these values again for Figures 3, 4, 5, and 6.
#
# These values preserve approximately the same physical ROC/PR axis
# dimensions as the original 9 x 3.8 inch figures, while adding a
# dedicated right-side region for the legend.
# =====================================================================

FIGSIZE = (13.6, 3.8)

# Preserve the SAME physical ROC/PR plotting region
# as the previous 11.6-inch version.
PLOT_LEFT = 0.083
PLOT_RIGHT = 0.595

PLOT_BOTTOM = 0.11
PLOT_TOP = 0.88
PLOT_WSPACE = 0.30

LINEWIDTH = 0.8

LEGEND_RIGHT = 0.985
LEGEND_CENTER_Y = 0.50


# =====================================================================
# Paths
#
# Notebook CWD:
#
# ALL prediction files used below come from the NEW Reviewer-4 run.
# =====================================================================

RESULT_ROOT = Path(os.environ.get("ORIGENE_PIPELINE_ROOT", "epigenetic_snakemake_pipline")).expanduser().resolve() / "results/08_reviewer4_fig6_40kbp_singlemarkers"

OUTPUT_ROOT = Path("results/supplementary/nt2d1_40kb_markers")


# =====================================================================
# Figure 6 datasets: NEW 40 kbp single-marker comparison
#
# IMPORTANT:
# All six curves below are from the new Reviewer-4 batch.
#
# Marker order and colors match the corresponding markers in the
# existing 5 kbp Figure 6.
# =====================================================================

DATASETS_40K = [
    (
        "NT2-D1 H3K4me3",
        RESULT_ROOT / "H3K4me3" / "OOF_predictions.csv",
        "limegreen",
    ),
    (
        "NT2-D1 H3K9ac",
        RESULT_ROOT / "H3K9ac" / "OOF_predictions.csv",
        "darkred",
    ),
    (
        "NT2-D1 H3K36me3",
        RESULT_ROOT / "H3K36me3" / "OOF_predictions.csv",
        "orange",
    ),
    (
        "NT2-D1 H3K9me3",
        RESULT_ROOT / "H3K9me3" / "OOF_predictions.csv",
        "gold",
    ),
    (
        "NT2-D1 H3K27me3",
        RESULT_ROOT / "H3K27me3" / "OOF_predictions.csv",
        "navajowhite",
    ),
    (
        "NT2-D1 H3K4me1",
        RESULT_ROOT / "H3K4me1" / "OOF_predictions.csv",
        "khaki",
    ),
]


# =====================================================================
# Loader
# =====================================================================

def load_new_oof_predictions(
    name,
    csv_path,
):

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing NEW Reviewer-4 OOF file:\n"
            f"{csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    # -----------------------------------------------------------------
    # Expected columns from the new Reviewer-4 batch runner
    # -----------------------------------------------------------------

    if {
        "True_Label",
        "Predicted_Probability",
    }.issubset(df.columns):

        y_true = (
            df["True_Label"]
            .to_numpy()
            .astype(int)
            .squeeze()
        )

        y_pred = (
            df["Predicted_Probability"]
            .to_numpy()
            .astype(float)
            .squeeze()
        )

    elif {
        "y_true",
        "y_pred",
    }.issubset(df.columns):

        y_true = (
            df["y_true"]
            .to_numpy()
            .astype(int)
            .squeeze()
        )

        y_pred = (
            df["y_pred"]
            .to_numpy()
            .astype(float)
            .squeeze()
        )

    else:

        raise ValueError(
            f"{name}: prediction columns not found.\n"
            f"Columns present:\n"
            f"{list(df.columns)}"
        )

    # -----------------------------------------------------------------
    # Strict checks
    # -----------------------------------------------------------------

    if len(y_true) != 1454:
        raise ValueError(
            f"{name}: expected 1454 OOF predictions, "
            f"found {len(y_true)}."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"{name}: length mismatch "
            f"{len(y_true)} vs {len(y_pred)}"
        )

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            f"{name}: y_true does not contain two classes."
        )

    n_positive = int(
        np.sum(y_true == 1)
    )

    n_negative = int(
        np.sum(y_true == 0)
    )

    if n_positive != 727 or n_negative != 727:
        raise ValueError(
            f"{name}: unexpected class counts. "
            f"CD={n_positive}, NG={n_negative}"
        )

    return (
        y_true,
        y_pred,
        df,
    )


# =====================================================================
# Plot function
# =====================================================================

def plot_40kb_single_marker_roc_pr(
    datasets=DATASETS_40K,
    outdir=OUTPUT_ROOT,
    stem="Figure6_OriGENE_40kbp_single_marker",
):

    curves = []
    metric_rows = []

    print("\n40 kbp single-marker OriGENE results")
    print("NEW Reviewer-4 batch only")
    print("-" * 80)

    # -----------------------------------------------------------------
    # Load predictions and calculate metrics
    # -----------------------------------------------------------------

    for name, csv_path, color in datasets:

        y_true, y_pred, df = load_new_oof_predictions(
            name=name,
            csv_path=csv_path,
        )

        # ROC
        fpr, tpr, _ = roc_curve(
            y_true,
            y_pred,
        )

        # PR
        precision, recall, _ = precision_recall_curve(
            y_true,
            y_pred,
        )

        # Metrics
        auroc = roc_auc_score(
            y_true,
            y_pred,
        )

        # Same convention used in the original OriGENE figures/tables
        auprc = skl_auc(
            recall,
            precision,
        )

        curves.append({
            "name": name,
            "fpr": fpr,
            "tpr": tpr,
            "rec": recall,
            "prec": precision,
            "auroc": auroc,
            "auprc": auprc,
            "color": color,
            "n": len(y_true),
        })

        metric_rows.append({
            "Dataset": name,
            "AUROC": auroc,
            "AUPRC": auprc,
            "N": len(y_true),
            "CD": int(np.sum(y_true == 1)),
            "NG": int(np.sum(y_true == 0)),
            "Source": str(csv_path),
        })

        print(
            f"{name:20s} "
            f"AUROC={auroc:.3f}  "
            f"AUPRC={auprc:.3f}  "
            f"N={len(y_true)}"
        )

        print(
            f"{'':20s} "
            f"source={csv_path}"
        )

    # =================================================================
    # FIXED FIGURE GEOMETRY
    # =================================================================

    fig = plt.figure(
        figsize=FIGSIZE
    )

    grid = fig.add_gridspec(
        1,
        2,
        left=PLOT_LEFT,
        right=PLOT_RIGHT,
        bottom=PLOT_BOTTOM,
        top=PLOT_TOP,
        wspace=PLOT_WSPACE,
        hspace=0.0,
    )

    ax1 = fig.add_subplot(
        grid[0, 0]
    )

    ax2 = fig.add_subplot(
        grid[0, 1]
    )

    # =================================================================
    # ROC panel
    # =================================================================

    ax1.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=LINEWIDTH,
        linestyle="--",
    )

    for c in curves:

        ax1.plot(
            c["fpr"],
            c["tpr"],
            lw=LINEWIDTH,
            color=c["color"],
        )

    ax1.set_xlim(
        0.0,
        1.0,
    )

    ax1.set_ylim(
        0.0,
        1.02,
    )

    ax1.set_xlabel(
        "False Positive Rate"
    )

    ax1.set_ylabel(
        "True Positive Rate"
    )

    # =================================================================
    # Precision-recall panel
    # =================================================================

    legend_handles = []
    legend_labels = []

    for c in curves:

        line, = ax2.plot(
            c["rec"],
            c["prec"],
            lw=LINEWIDTH,
            color=c["color"],
        )

        legend_handles.append(
            line
        )

        # Keep exactly the same legend convention as the 5 kbp figure.
        legend_labels.append(
            f"{c['name']} "
            f"(AUROC={c['auroc']:.2f}; "
            f"AUPRC={c['auprc']:.2f})"
        )

    ax2.set_xlim(
        0.0,
        1.0,
    )

    ax2.set_ylim(
        0.0,
        1.02,
    )

    ax2.set_xlabel(
        "Recall"
    )

    ax2.set_ylabel(
        "Precision"
    )

    # =================================================================
    # Legend
    #
    # IMPORTANT:
    # EXACTLY the same settings as the existing publication figures.
    # =================================================================

    fig.legend(
        legend_handles,
        legend_labels,
    
        loc="center left",
    
        bbox_to_anchor=(
            LEGEND_LEFT,
            LEGEND_CENTER_Y,
        ),
    
        bbox_transform=fig.transFigure,
    
        ncol=1,
        fancybox=True,
        shadow=True,
        fontsize=14,
        labelspacing=0.18,
        borderpad=0.40,
        handlelength=2.0,
        handletextpad=0.60,
    )

    # =================================================================
    # Output
    #
    # DO NOT use bbox_inches="tight".
    # The output canvas must remain exactly FIGSIZE for every figure.
    # =================================================================

    outdir = Path(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem_path = (
        outdir / stem
    )

    # -----------------------------------------------------------------
    # Save provenance / metric table
    # -----------------------------------------------------------------

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df.to_csv(
        str(stem_path) + "_metrics.csv",
        index=False,
    )

    # TIFF — publication-quality raster
    # Additional export omitted; retain canonical PNG.

    # PNG — high-resolution preview / Overleaf
    fig.savefig(
        str(stem_path) + ".png",
        format="png",
        dpi=600,
        facecolor="white",
    )

    # EPS — vector
    # Additional export omitted; retain canonical PNG.

    # PDF — vector
    # Additional export omitted; retain canonical PNG.

    # SVG — vector
    # Additional export omitted; retain canonical PNG.

    plt.show()

    print("\nSaved:")
    print(str(stem_path) + ".png")
    print(str(stem_path) + ".tiff")
    print(str(stem_path) + ".pdf")
    print(str(stem_path) + ".eps")
    print(str(stem_path) + ".svg")
    print(str(stem_path) + "_metrics.csv")

    return (
        fig,
        curves,
        metrics_df,
    )


# =====================================================================
# Run
# =====================================================================

fig, curves, metrics_df = plot_40kb_single_marker_roc_pr()

display(metrics_df)
