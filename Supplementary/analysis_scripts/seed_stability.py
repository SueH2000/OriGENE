"""seed stability. See Supplementary/README.md."""
import os

def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

# =====================================================================
# RANDOM-SEED ROBUSTNESS
# SINGLE-TRACK vs PAIRED NORMAL+TUMOUR
#
# Same visual logic as the existing random-seed figure.
#
# Grey/black = single-track H3K4me3
# Blue       = paired normal + tumour H3K4me3
#
# Small circles = individual training seeds
# Filled circle + horizontal error bar = across-seed mean +/- SD
# Open diamond = original OriGENE run
# =====================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# =====================================================================
# STYLE
# Unified with the other supplementary manuscript figures
# =====================================================================

MM = 1 / 25.4
PANEL_Y = 1.035

mpl.rcParams.update({

    "text.usetex": False,

    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",

    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.2,

    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "savefig.dpi": 600,
})


def clean_axes(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        direction="out",
        width=0.8,
        length=3,
    )

    ax.grid(False)


def add_panel_label(
    ax,
    label,
):

    ax.text(
        -0.17,
        PANEL_Y,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


# =====================================================================
# PATHS
# =====================================================================

PROJECT = Path(os.environ.get("ORIGENE_PIPELINE_ROOT", "epigenetic_snakemake_pipline")).expanduser().resolve()

SINGLE_EVAL = (
    PROJECT
    / "results/07_model_evaluation/reviewer_singletrack"
)

PAIRED_EVAL = (
    PROJECT
    / "results/07_model_evaluation/"
      "reviewer_original_ng/original_ng"
)

SINGLE_SEED_ROOT = (
    SINGLE_EVAL
    / "training_seed_robustness/original_ng"
)

PAIRED_SEED_ROOT = (
    PAIRED_EVAL
    / "training_seed_robustness_paired"
)

OUTDIR = Path("results/supplementary/seed_stability")

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# DATASET DEFINITIONS
# =====================================================================

DATASETS = [
    "Lung I",
    "Lung II",
    "Liver I",
    "Liver II",
]

SINGLE_IDS = {
    "Lung I": "Lung_I_Normal",
    "Lung II": "Lung_II_Normal",
    "Liver I": "Liver_I_Normal",
    "Liver II": "Liver_II_Normal",
}

PAIRED_IDS = {
    "Lung I": "Lung_I",
    "Lung II": "Lung_II",
    "Liver I": "Liver_I",
    "Liver II": "Liver_II",
}


# =====================================================================
# COLORS
# =====================================================================

# Existing single-track visual language
SINGLE_SEED_COLOR = "#BDBDBD"
SINGLE_MAIN_COLOR = "#111111"

# New paired condition
PAIRED_SEED_COLOR = "#9EC5E5"
PAIRED_MAIN_COLOR = "#2B6F9E"


# =====================================================================
# LOAD REPEATED-SEED RESULTS
# =====================================================================

def load_seed_metrics(root):

    p = root / "repeated_training_seed_metrics.csv"

    if not p.exists():
        raise FileNotFoundError(
            f"Missing repeated-seed metrics:\n{p}"
        )

    df = pd.read_csv(p)

    required = {
        "dataset",
        "training_seed",
        "AUROC",
        "AUPRC",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{p}\nMissing columns: {sorted(missing)}"
        )

    return df


single_seed = load_seed_metrics(
    SINGLE_SEED_ROOT
)

paired_seed = load_seed_metrics(
    PAIRED_SEED_ROOT
)


# =====================================================================
# STRICT FIVE-SEED AUDIT
# =====================================================================

EXPECTED_SEEDS = {
    4,
    17,
    42,
    73,
    101,
}


def audit_seed_results(
    df,
    id_map,
    label,
):

    for display_name in DATASETS:

        dataset_id = id_map[display_name]

        tmp = df.loc[
            df["dataset"].eq(dataset_id)
        ].copy()

        seeds = set(
            tmp["training_seed"]
            .astype(int)
            .tolist()
        )

        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"{label} / {display_name}: "
                f"expected seeds {sorted(EXPECTED_SEEDS)}, "
                f"found {sorted(seeds)}"
            )

        if len(tmp) != 5:
            raise RuntimeError(
                f"{label} / {display_name}: "
                f"expected 5 seed rows, found {len(tmp)}"
            )


audit_seed_results(
    single_seed,
    SINGLE_IDS,
    "single-track",
)

audit_seed_results(
    paired_seed,
    PAIRED_IDS,
    "paired",
)


# =====================================================================
# LOAD ORIGINAL ORIGENE RESULTS
#
# Recompute AUROC/AP from the original pooled OOF predictions.
# This uses the same metric definition as repeated-seed script:
# AUPRC = sklearn average_precision_score.
# =====================================================================

def load_original_prediction_metrics(
    path,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing original prediction file:\n{path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
    )

    required = {
        "y_true",
        "y_prob",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}\n"
            f"Missing columns: {sorted(missing)}"
        )

    # Keep final-epoch within-patient result if the file contains
    # more than one evaluation type.
    if "evaluation_type" in df.columns:

        mask = (
            df["evaluation_type"]
            .astype(str)
            .eq("within_patient_final_epoch")
        )

        if mask.any():
            df = df.loc[mask].copy()

    # For within-patient OOF predictions each gene should occur once.
    if "gene" in df.columns:

        if df["gene"].duplicated().any():

            raise RuntimeError(
                f"Duplicated genes in:\n{path}"
            )

    y_true = (
        df["y_true"]
        .to_numpy(dtype=int)
    )

    y_prob = (
        df["y_prob"]
        .to_numpy(dtype=float)
    )

    if len(y_true) != 1454:

        raise RuntimeError(
            f"{path}\n"
            f"Expected 1454 OOF rows, found {len(y_true)}"
        )

    return {
        "AUROC": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
        "AUPRC": float(
            average_precision_score(
                y_true,
                y_prob,
            )
        ),
    }


single_original = {}
paired_original = {}


for display_name in DATASETS:

    single_id = SINGLE_IDS[
        display_name
    ]

    paired_id = PAIRED_IDS[
        display_name
    ]

    single_path = (
        SINGLE_EVAL
        / "40000bp/normal/original_ng/sources"
        / single_id
        / "within_patient"
        / "predictions_all_folds.tsv"
    )

    paired_path = (
        PAIRED_EVAL
        / "sources"
        / paired_id
        / "within_patient"
        / "predictions_all_folds.tsv"
    )

    single_original[
        display_name
    ] = load_original_prediction_metrics(
        single_path
    )

    paired_original[
        display_name
    ] = load_original_prediction_metrics(
        paired_path
    )


# =====================================================================
# PREPARE SUMMARY
# =====================================================================

rows = []

for display_name in DATASETS:

    for condition, df, id_map in [
        (
            "Single-track",
            single_seed,
            SINGLE_IDS,
        ),
        (
            "Paired",
            paired_seed,
            PAIRED_IDS,
        ),
    ]:

        dataset_id = id_map[
            display_name
        ]

        tmp = (
            df.loc[
                df["dataset"].eq(
                    dataset_id
                )
            ]
            .sort_values(
                "training_seed"
            )
            .copy()
        )

        rows.append({
            "dataset": display_name,
            "condition": condition,

            "AUROC_mean":
                tmp["AUROC"].mean(),

            "AUROC_SD":
                tmp["AUROC"].std(
                    ddof=1
                ),

            "AUPRC_mean":
                tmp["AUPRC"].mean(),

            "AUPRC_SD":
                tmp["AUPRC"].std(
                    ddof=1
                ),
        })


summary = pd.DataFrame(
    rows
)


print("\nSeed robustness summary")
print("=" * 100)
print(
    summary.to_string(
        index=False
    )
)


# =====================================================================
# FIGURE
# =====================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        180 * MM,
        80 * MM,
    ),
)

ax_auc, ax_pr = axes


# base y positions
y_base = np.arange(
    len(DATASETS)
)

# Paired / single-track separation
OFFSET = 0.105

y_single = (
    y_base - OFFSET
)

y_paired = (
    y_base + OFFSET
)


# =====================================================================
# PANEL FUNCTION
# =====================================================================

def plot_metric_panel(
    ax,
    metric,
    panel_label,
):

    # --------------------------------------------------------------
    # Four donors
    # --------------------------------------------------------------

    for i, display_name in enumerate(DATASETS):

        single_id = SINGLE_IDS[
            display_name
        ]

        paired_id = PAIRED_IDS[
            display_name
        ]


        # ==========================================================
        # SINGLE TRACK
        # ==========================================================

        st = (
            single_seed.loc[
                single_seed[
                    "dataset"
                ].eq(single_id)
            ]
            .sort_values(
                "training_seed"
            )
        )

        st_values = (
            st[metric]
            .to_numpy(float)
        )

        st_mean = (
            st_values.mean()
        )

        st_sd = (
            st_values.std(
                ddof=1
            )
        )


        # Individual seeds
        ax.scatter(
            st_values,
            np.full(
                len(st_values),
                y_single[i],
            ),
            s=28,
            facecolor=SINGLE_SEED_COLOR,
            edgecolor="#888888",
            linewidth=0.6,
            zorder=2,
        )


        # Across-seed mean +/- SD
        ax.errorbar(
            st_mean,
            y_single[i],

            xerr=st_sd,

            fmt="o",

            markersize=5.5,

            color=SINGLE_MAIN_COLOR,
            ecolor=SINGLE_MAIN_COLOR,

            elinewidth=1.15,
            capsize=0,

            zorder=4,
        )


        # Original OriGENE
        ax.scatter(
            single_original[
                display_name
            ][metric],

            y_single[i],

            marker="D",
            s=42,

            facecolor="white",
            edgecolor=SINGLE_MAIN_COLOR,

            linewidth=1.1,
            zorder=5,
        )


        # ==========================================================
        # PAIRED
        # ==========================================================

        pr = (
            paired_seed.loc[
                paired_seed[
                    "dataset"
                ].eq(paired_id)
            ]
            .sort_values(
                "training_seed"
            )
        )

        pr_values = (
            pr[metric]
            .to_numpy(float)
        )

        pr_mean = (
            pr_values.mean()
        )

        pr_sd = (
            pr_values.std(
                ddof=1
            )
        )


        # Individual seeds
        ax.scatter(
            pr_values,
            np.full(
                len(pr_values),
                y_paired[i],
            ),
            s=28,
            facecolor=PAIRED_SEED_COLOR,
            edgecolor=PAIRED_MAIN_COLOR,
            linewidth=0.6,
            zorder=2,
        )


        # Across-seed mean +/- SD
        ax.errorbar(
            pr_mean,
            y_paired[i],

            xerr=pr_sd,

            fmt="o",

            markersize=5.5,

            color=PAIRED_MAIN_COLOR,
            ecolor=PAIRED_MAIN_COLOR,

            elinewidth=1.15,
            capsize=0,

            zorder=4,
        )


        # Original paired OriGENE
        ax.scatter(
            paired_original[
                display_name
            ][metric],

            y_paired[i],

            marker="D",
            s=42,

            facecolor="white",
            edgecolor=PAIRED_MAIN_COLOR,

            linewidth=1.2,
            zorder=5,
        )


    # --------------------------------------------------------------
    # Same general geometry as current slide
    # --------------------------------------------------------------

    ax.set_yticks(
        y_base
    )

    ax.set_yticklabels(
        DATASETS
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        metric
    )

    ax.set_title(
        metric,
        loc="left",
        y=PANEL_Y,
        pad=0,
    )

    clean_axes(
        ax
    )

    add_panel_label(
        ax,
        panel_label,
    )


# =====================================================================
# DRAW PANELS
# =====================================================================

plot_metric_panel(
    ax_auc,
    "AUROC",
    "(a)",
)

plot_metric_panel(
    ax_pr,
    "AUPRC",
    "(b)",
)


# Same range as current random-seed figure.
XMIN = 0.70
XMAX = 0.82

for ax in axes:

    ax.set_xlim(
        XMIN,
        XMAX,
    )

    ax.set_xticks(
        np.arange(
            0.70,
            0.821,
            0.02,
        )
    )


# Only left panel needs donor labels
ax_pr.set_yticklabels(
    []
)


# =====================================================================
# RANGE AUDIT
# =====================================================================

all_auc = np.concatenate([
    single_seed["AUROC"].to_numpy(float),
    paired_seed["AUROC"].to_numpy(float),

    np.asarray([
        single_original[x]["AUROC"]
        for x in DATASETS
    ]),

    np.asarray([
        paired_original[x]["AUROC"]
        for x in DATASETS
    ]),
])

all_pr = np.concatenate([
    single_seed["AUPRC"].to_numpy(float),
    paired_seed["AUPRC"].to_numpy(float),

    np.asarray([
        single_original[x]["AUPRC"]
        for x in DATASETS
    ]),

    np.asarray([
        paired_original[x]["AUPRC"]
        for x in DATASETS
    ]),
])


if (
    all_auc.min() < XMIN
    or all_auc.max() > XMAX
    or all_pr.min() < XMIN
    or all_pr.max() > XMAX
):
    print(
        "\nWARNING: at least one value lies outside "
        f"the fixed {XMIN:.2f}-{XMAX:.2f} plotting range."
    )


# =====================================================================
# LEGEND
#
# Color = input condition
# Shape = statistic
# =====================================================================

legend_handles = [

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=5.5,
        markerfacecolor=SINGLE_MAIN_COLOR,
        markeredgecolor=SINGLE_MAIN_COLOR,
        label="Single-track",
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=5.5,
        markerfacecolor=PAIRED_MAIN_COLOR,
        markeredgecolor=PAIRED_MAIN_COLOR,
        label="Paired normal+tumour",
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=4.8,
        markerfacecolor="#C5C5C5",
        markeredgecolor="#888888",
        label="Individual training seed",
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="-",
        markersize=4.8,
        color="black",
        label=r"Across-seed mean $\pm$ SD",
    ),

    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="None",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor="black",
        label="Original OriGENE",
    ),
]


# ---------------------------------------------------------------------
# Layout first, then center the legend relative to the actual two-panel
# plotting region rather than the full canvas.
# ---------------------------------------------------------------------

PLOT_LEFT = 0.11
PLOT_RIGHT = 0.985
PLOT_TOP = 0.88
PLOT_BOTTOM = 0.28
PLOT_WSPACE = 0.30

fig.subplots_adjust(
    left=PLOT_LEFT,
    right=PLOT_RIGHT,
    top=PLOT_TOP,
    bottom=PLOT_BOTTOM,
    wspace=PLOT_WSPACE,
)

LEGEND_CENTER_X = (
    axes[0].get_position().x0
    + axes[1].get_position().x1
) / 2


legend = fig.legend(
    handles=legend_handles,

    loc="lower center",

    bbox_to_anchor=(
        LEGEND_CENTER_X,
        0.025,
    ),

    bbox_transform=fig.transFigure,

    ncol=5,

    frameon=True,
    fancybox=False,
    framealpha=1.0,
    facecolor="white",
    edgecolor="0.75",

    handletextpad=0.45,
    columnspacing=1.05,
    borderpad=0.45,
)

legend.get_frame().set_linewidth(
    0.6
)


# =====================================================================
# SAVE
# =====================================================================

stem = (
    OUTDIR
    / "random_seed_single_vs_paired"
)

fig.savefig(
    str(stem) + ".png",
    format="png",
    dpi=600,
    facecolor="white",
)

# Additional export omitted; retain canonical PNG.

# Additional export omitted; retain canonical PNG.

# Additional export omitted; retain canonical PNG.

plt.show()


# =====================================================================
# PRINT SD RANGE FOR PPT / RESPONSE
# =====================================================================

single_summary = (
    summary.loc[
        summary["condition"].eq(
            "Single-track"
        )
    ]
)

paired_summary = (
    summary.loc[
        summary["condition"].eq(
            "Paired"
        )
    ]
)


print("\n" + "=" * 100)
print("SD RANGES")
print("=" * 100)

print(
    "Single-track AUROC SD:",
    f"{single_summary['AUROC_SD'].min():.4f}"
    "–"
    f"{single_summary['AUROC_SD'].max():.4f}",
)

print(
    "Single-track AUPRC SD:",
    f"{single_summary['AUPRC_SD'].min():.4f}"
    "–"
    f"{single_summary['AUPRC_SD'].max():.4f}",
)

print(
    "Paired AUROC SD:",
    f"{paired_summary['AUROC_SD'].min():.4f}"
    "–"
    f"{paired_summary['AUROC_SD'].max():.4f}",
)

print(
    "Paired AUPRC SD:",
    f"{paired_summary['AUPRC_SD'].min():.4f}"
    "–"
    f"{paired_summary['AUPRC_SD'].max():.4f}",
)

print("\nSaved:")
print(str(stem) + ".png")
print(str(stem) + ".pdf")
print(str(stem) + ".svg")
print(str(stem) + ".eps")
