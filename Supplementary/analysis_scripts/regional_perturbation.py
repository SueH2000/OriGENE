"""regional perturbation. See Supplementary/README.md."""
import os

def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

# =============================================================================
# OriGENE Slide 11 revision
# V2 common-cohort regional sensitivity
#
# Key changes from the old Slide 11:
#   1. Uses biological_interpretability_v2
#   2. Uses region_equal_length_common_summary.csv
#   3. Same eligible genes are used across all five regions within each
#      gene-set / class / dataset combination
#   4. All six panels use the SAME y-axis
#   5. Saves publication-quality PNG/PDF/EPS
#   6. Also saves a diagnostic plot of input perturbation magnitude
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT = Path(os.environ.get("ORIGENE_PIPELINE_ROOT", "epigenetic_snakemake_pipline")).expanduser().resolve()

INPUT_ROOT = (
    PROJECT
    / "results/07_model_evaluation/reviewer_singletrack/"
      "biological_interpretability_v2/region_equal_length"
)

OUTPUT_DIR = Path("results/supplementary/regional_perturbation")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

GENE_SETS = {
    "original_ng": "Original NG",
    "length_set1": "Length-matched set 1",
    "length_set2": "Length-matched set 2",
}

SOURCES = [
    "Lung_I_Normal",
    "Lung_II_Normal",
    "Liver_I_Normal",
    "Liver_II_Normal",
]

SOURCE_LABELS = {
    "Lung_I_Normal": "Lung I",
    "Lung_II_Normal": "Lung II",
    "Liver_I_Normal": "Liver I",
    "Liver_II_Normal": "Liver II",
}

# Same visual convention as the earlier figure:
# I = solid, II = dashed
SOURCE_STYLE = {
    "Lung_I_Normal": {
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "Lung_II_Normal": {
        "color": "#6baed6",
        "linestyle": "--",
    },
    "Liver_I_Normal": {
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    "Liver_II_Normal": {
        "color": "#fdae6b",
        "linestyle": "--",
    },
}

# V2 exact region names
REGIONS = [
    "promoter",
    "early_body",
    "proximal_body",
    "distal_body",
    "padding",
]

REGION_LABELS = {
    "promoter": "TSS",
    "early_body": "Early",
    "proximal_body": "Proximal",
    "distal_body": "Distal",
    "padding": "Padding",
}

CLASSES = [
    "CD",
    "NG",
]

INPUT_FILE = "region_equal_length_common_summary.csv"


# =============================================================================
# 3. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.2,

    "axes.titlesize": 10.5,
    "axes.labelsize": 10.5,

    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,

    "legend.fontsize": 8.3,

    "axes.linewidth": 0.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SPINE_COLOR = "#444444"
GRID_COLOR = "#E8E8E8"

MEDIAN_COLOR = "black"
MEDIAN_LINEWIDTH = 2.2
SOURCE_LINEWIDTH = 1.25


# =============================================================================
# 4. LOAD V2 COMMON-COHORT DATA
# =============================================================================

records = []

for gene_set in GENE_SETS:

    for source in SOURCES:

        f = (
            INPUT_ROOT
            / gene_set
            / source
            / INPUT_FILE
        )

        if not f.exists():
            raise FileNotFoundError(
                f"Missing V2 summary:\n{f}"
            )

        df = pd.read_csv(f)

        required = {
            "gene_set",
            "dataset",
            "class",
            "region",
            "n_genes",
            "median_abs_delta_logit",
            "mean_abs_delta_logit",
            "median_abs_input_change",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{f}\n"
                f"Missing columns: {sorted(missing)}"
            )

        # ---------------------------------------------------------
        # Expected: 2 classes x 5 regions = 10 rows
        # ---------------------------------------------------------

        if len(df) != 10:
            raise ValueError(
                f"{f}\n"
                f"Expected 10 rows, found {len(df)}"
            )

        # ---------------------------------------------------------
        # Validate region names
        # ---------------------------------------------------------

        observed_regions = set(
            df["region"].astype(str)
        )

        if observed_regions != set(REGIONS):
            raise ValueError(
                f"{f}\n"
                f"Unexpected regions:\n"
                f"{sorted(observed_regions)}"
            )

        # ---------------------------------------------------------
        # Critical V2 audit:
        # same eligible genes across all regions within each class
        # ---------------------------------------------------------

        for cls in CLASSES:

            temp = df.loc[
                df["class"].astype(str).eq(cls)
            ]

            n_values = (
                temp["n_genes"]
                .dropna()
                .astype(int)
                .unique()
            )

            if len(n_values) != 1:
                raise ValueError(
                    "\nCOMMON-COHORT CHECK FAILED\n"
                    f"Gene set: {gene_set}\n"
                    f"Source:   {source}\n"
                    f"Class:    {cls}\n"
                    f"n_genes across regions: {n_values}\n"
                )

        df = df.copy()

        df["gene_set_id"] = gene_set
        df["source_id"] = source
        df["source_label"] = SOURCE_LABELS[source]

        records.append(df)


data = pd.concat(
    records,
    ignore_index=True,
)


# =============================================================================
# 5. ORDER CATEGORICAL VARIABLES
# =============================================================================

data["region"] = pd.Categorical(
    data["region"],
    categories=REGIONS,
    ordered=True,
)

data["class"] = pd.Categorical(
    data["class"],
    categories=CLASSES,
    ordered=True,
)

data = data.sort_values(
    [
        "class",
        "gene_set_id",
        "source_id",
        "region",
    ]
).reset_index(drop=True)


# =============================================================================
# 6. AUDIT TABLE
# =============================================================================

audit = (
    data
    .groupby(
        [
            "gene_set_id",
            "source_id",
            "class",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        n_regions=("region", "nunique"),
        n_genes_min=("n_genes", "min"),
        n_genes_max=("n_genes", "max"),
    )
)

audit["common_cohort_pass"] = (
    (audit["n_regions"] == 5)
    &
    (
        audit["n_genes_min"]
        ==
        audit["n_genes_max"]
    )
)

print(
    "\n================ COMMON-COHORT AUDIT ================\n"
)

display(audit)

if not audit["common_cohort_pass"].all():
    raise RuntimeError(
        "At least one V2 dataset failed the common-cohort audit."
    )


# =============================================================================
# 7. ACROSS-SOURCE MEDIANS
# =============================================================================

across_source = (
    data
    .groupby(
        [
            "gene_set_id",
            "class",
            "region",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        across_source_median_delta=(
            "median_abs_delta_logit",
            "median",
        ),
        across_source_median_input_change=(
            "median_abs_input_change",
            "median",
        ),
        n_genes_min=("n_genes", "min"),
        n_genes_max=("n_genes", "max"),
    )
)

print(
    "\n================ ACROSS-SOURCE MEDIANS ================\n"
)

display(across_source)


# =============================================================================
# 8. SAVE EXACT DATA USED FOR FIGURE
# =============================================================================

plot_data_file = (
    OUTPUT_DIR
    / "slide11_v2_common_cohort_plot_data.csv"
)

data.to_csv(
    plot_data_file,
    index=False,
)

across_source.to_csv(
    OUTPUT_DIR
    / "slide11_v2_across_source_medians.csv",
    index=False,
)

audit.to_csv(
    OUTPUT_DIR
    / "slide11_v2_common_cohort_audit.csv",
    index=False,
)


# =============================================================================
# 9. DETERMINE ONE GLOBAL Y-AXIS FOR ALL SIX PANELS
# =============================================================================

all_delta_values = np.concatenate([
    data[
        "median_abs_delta_logit"
    ].to_numpy(float),

    across_source[
        "across_source_median_delta"
    ].to_numpy(float),
])

finite_delta = all_delta_values[
    np.isfinite(all_delta_values)
]

raw_max = finite_delta.max()

# Add ~8% headroom.
ymax = raw_max * 1.08

# Round upward to a clean 0.1 interval.
ymax = np.ceil(
    ymax / 0.1
) * 0.1

print(
    f"\nGlobal |Δlogit| y-axis: 0 to {ymax:.1f}"
)


# =============================================================================
# 10. HELPER FUNCTION
# =============================================================================

def style_axis(ax):

    for side in [
        "left",
        "bottom",
        "top",
        "right",
    ]:
        ax.spines[side].set_color(
            SPINE_COLOR
        )

        ax.spines[side].set_linewidth(
            0.8
        )

    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.6,
        zorder=0,
    )

    ax.tick_params(
        axis="both",
        direction="out",
        length=3.5,
        width=0.8,
        color=SPINE_COLOR,
    )


# =============================================================================
# 11. MAIN FIGURE:
#     RAW REGIONAL SENSITIVITY WITH ONE COMMON Y-AXIS
# =============================================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(7.1, 4.75),
    sharex=True,
    sharey=True,
)

x = np.arange(
    len(REGIONS)
)

for row_idx, cls in enumerate(CLASSES):

    for col_idx, gene_set in enumerate(GENE_SETS):

        ax = axes[
            row_idx,
            col_idx,
        ]

        # ---------------------------------------------------------
        # Four source-specific curves
        # ---------------------------------------------------------

        for source in SOURCES:

            temp = data.loc[
                (
                    data["gene_set_id"].eq(
                        gene_set
                    )
                )
                &
                (
                    data["source_id"].eq(
                        source
                    )
                )
                &
                (
                    data["class"].astype(str).eq(
                        cls
                    )
                )
            ].copy()

            temp = (
                temp
                .set_index("region")
                .reindex(REGIONS)
            )

            y = temp[
                "median_abs_delta_logit"
            ].to_numpy(float)

            style = SOURCE_STYLE[source]

            ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=SOURCE_LINEWIDTH,
                marker=None,
                zorder=2,
            )

        # ---------------------------------------------------------
        # Across-source median
        # ---------------------------------------------------------

        med = across_source.loc[
            (
                across_source[
                    "gene_set_id"
                ].eq(gene_set)
            )
            &
            (
                across_source[
                    "class"
                ].astype(str)
                .eq(cls)
            )
        ].copy()

        med = (
            med
            .set_index("region")
            .reindex(REGIONS)
        )

        y_med = med[
            "across_source_median_delta"
        ].to_numpy(float)

        ax.plot(
            x,
            y_med,
            color=MEDIAN_COLOR,
            linewidth=MEDIAN_LINEWIDTH,
            marker="o",
            markersize=3.8,
            markerfacecolor=MEDIAN_COLOR,
            markeredgecolor=MEDIAN_COLOR,
            zorder=4,
        )

        # ---------------------------------------------------------
        # Axis formatting
        # ---------------------------------------------------------

        ax.set_ylim(
            0,
            ymax,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            [
                REGION_LABELS[r]
                for r in REGIONS
            ],
            rotation=0,
        )

        style_axis(
            ax
        )

        # Column titles only on upper row
        if row_idx == 0:
            ax.set_title(
                GENE_SETS[
                    gene_set
                ],
                pad=8,
            )

        # Y labels only in first column
        if col_idx == 0:
            ax.set_ylabel(
                f"{cls}\nMedian per-gene |Δlogit|"
            )

        # ---------------------------------------------------------
        # Show common eligible N in each panel
        # ---------------------------------------------------------

        n_values = (
            data.loc[
                (
                    data[
                        "gene_set_id"
                    ].eq(gene_set)
                )
                &
                (
                    data[
                        "class"
                    ].astype(str)
                    .eq(cls)
                ),
                "n_genes",
            ]
            .astype(int)
            .unique()
        )

        # N can differ across source only if preprocessing differs.
        # For this figure, print a range if necessary.
        if len(n_values) == 1:
            n_text = f"N = {n_values[0]}"
        else:
            n_text = (
                f"N = {n_values.min()}–"
                f"{n_values.max()}"
            )

        ax.text(
            0.98,
            0.94,
            n_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#666666",
        )


# =============================================================================
# 12. PANEL LABELS
# =============================================================================

axes[0, 0].text(
    -0.15,
    1.08,
    r"$\mathbf{(a)}$",
    transform=axes[0, 0].transAxes,
    fontsize=10.5,
    va="bottom",
    ha="left",
)

axes[1, 0].text(
    -0.15,
    1.08,
    r"$\mathbf{(b)}$",
    transform=axes[1, 0].transAxes,
    fontsize=10.5,
    va="bottom",
    ha="left",
)


# =============================================================================
# 13. SHARED LEGEND
# =============================================================================

legend_handles = []

for source in SOURCES:

    style = SOURCE_STYLE[source]

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            label=SOURCE_LABELS[source],
        )
    )

legend_handles.append(
    Line2D(
        [0],
        [0],
        color=MEDIAN_COLOR,
        linewidth=MEDIAN_LINEWIDTH,
        marker="o",
        markersize=4,
        label="Across-source median",
    )
)

legend = fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(
        0.5,
        -0.02,
    ),
    ncol=5,
    frameon=True,
    fancybox=False,
    edgecolor="#A0A0A0",
    facecolor="white",
    framealpha=1.0,
    handlelength=2.3,
    columnspacing=1.4,
)

legend.get_frame().set_linewidth(
    0.7
)


# =============================================================================
# 14. LAYOUT
# =============================================================================

fig.subplots_adjust(
    left=0.10,
    right=0.985,
    top=0.90,
    bottom=0.20,
    wspace=0.18,
    hspace=0.28,
)


# =============================================================================
# 15. SAVE MAIN FIGURE
# =============================================================================

main_base = (
    OUTPUT_DIR
    / "slide11_v2_common_cohort_shared_y"
)

fig.savefig(
    main_base.with_suffix(".png"),
    dpi=600,
    bbox_inches="tight",
)

# Additional export omitted; retain canonical PNG.

# Additional export omitted; retain canonical PNG.

plt.show()
plt.close(fig)


# =============================================================================
