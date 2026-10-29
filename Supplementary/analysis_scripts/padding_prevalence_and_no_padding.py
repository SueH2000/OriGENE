"""padding prevalence and no padding. See Supplementary/README.md."""
import os

def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# OriGENE reviewer controls — unified two-panel figure
#
# (a) Terminal-padding prevalence by class
# (b) No-padding subset vs original OriGENE AUROC
#
# UNIFIED VISUAL GRAMMAR:
#   - categorical x-axis in BOTH panels
#   - paired coloured points in BOTH panels
#   - thin neutral connector within each category
#   - matching marker size / line weight / legend framing
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT = Path(os.environ.get("ORIGENE_PIPELINE_ROOT", "epigenetic_snakemake_pipline")).expanduser().resolve()

RESULT_ROOT = (
    PROJECT
    / "results/07_model_evaluation/"
      "reviewer_singletrack"
)

EXACT_ROOT = Path(os.environ.get("ORIGENE_PADDING_ROOT", "results/exact_padding_3sets")).expanduser().resolve()

VALIDATION_ROOT = (
    EXACT_ROOT
    / "validation"
)

NO_PADDING_FILE = (
    EXACT_ROOT
    / "no_padding_heldout"
    / "exact_no_padding_heldout_dataset_metrics.tsv"
)

FIGURE_DIR = Path("results/supplementary/padding_prevalence_and_no_padding")

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. STUDY DESIGN
# ============================================================

GENE_SETS = [
    "original_ng",
    "length_set1",
    "length_set2",
]

GENE_SET_LABELS = {
    "original_ng":
        "Randomly\nselected",

    "length_set1":
        "Length-matched\n1",

    "length_set2":
        "Length-matched\n2",
}

SOURCES = [
    "Lung_I_Normal",
    "Lung_II_Normal",
    "Liver_I_Normal",
    "Liver_II_Normal",
]

SOURCE_LABELS = {
    "Lung_I_Normal":
        "Lung I",

    "Lung_II_Normal":
        "Lung II",

    "Liver_I_Normal":
        "Liver I",

    "Liver_II_Normal":
        "Liver II",
}

NO_PADDING_GENE_SET = "original_ng"


# ============================================================
# 3. PALETTE / GEOMETRY
#
# The same two-colour hierarchy is used in both panels:
#   blue   = primary/reference series
#   orange = comparison series
# ============================================================

BLUE = "#4F81BD"
ORANGE = "#E58B45"

BLUE_LIGHT = "#8FB1D5"
ORANGE_LIGHT = "#EDB17D"

CONNECTOR_GRAY = "#B8B8B8"
CHANCE_GRAY = "#A0A0A0"

MM = 1 / 25.4

FIG_W_MM = 180
FIG_H_MM = 68

PANEL_Y = 1.035

POINT_SIZE = 34
POINT_EDGE_WIDTH = 0.8
CONNECTOR_WIDTH = 0.85


# ============================================================
# 4. PUBLICATION STYLE
# ============================================================

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
    "legend.fontsize": 7.0,

    "axes.linewidth": 0.8,

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


# ============================================================
# 5. HELPERS
# ============================================================

def clean_axes(
    ax,
):

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_linewidth(
        0.8
    )

    ax.spines[
        "bottom"
    ].set_linewidth(
        0.8
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        width=0.8,
        length=3.0,
    )


def panel_label(
    ax,
    text,
):

    ax.text(
        -0.16,
        PANEL_Y,
        text,

        transform=
            ax.transAxes,

        fontsize=
            8.5,

        fontweight=
            "bold",

        ha=
            "left",

        va=
            "bottom",

        clip_on=
            False,
    )


def save_all(
    fig,
    stem,
):

    paths = {
        "png":
            FIGURE_DIR
            / f"{stem}_600dpi.png",

        "pdf":
            FIGURE_DIR
            / f"{stem}.pdf",

        "eps":
            FIGURE_DIR
            / f"{stem}.eps",

        "svg":
            FIGURE_DIR
            / f"{stem}.svg",
    }


    # Fixed canvas:
    # do NOT use bbox_inches="tight".

    fig.savefig(
        paths["png"],
        format="png",
        dpi=600,
        facecolor="white",
    )

    # Additional export omitted; retain canonical PNG.

    # Additional export omitted; retain canonical PNG.

    # Additional export omitted; retain canonical PNG.


    print(
        "\nSaved:"
    )

    for path in [paths["png"]]:
        print(
            " ",
            path,
        )


# ============================================================
# 6. LOAD TERMINAL-PADDING PREVALENCE
# ============================================================

prevalence_rows = []


for gene_set in GENE_SETS:

    for source in SOURCES:

        validation_file = (
            VALIDATION_ROOT
            / gene_set
            / f"{source}_exact_padding_validation.tsv"
        )


        if not validation_file.exists():

            raise FileNotFoundError(
                f"Missing validation file:\n"
                f"{validation_file}"
            )


        tmp = pd.read_csv(
            validation_file,
            sep="\t",
        )


        required = {
            "gene",
            "label",
            "padding_bp",
        }

        missing = (
            required
            - set(
                tmp.columns
            )
        )


        if missing:

            raise RuntimeError(
                f"{validation_file}\n"
                f"Missing columns: "
                f"{sorted(missing)}"
            )


        for label, class_name in [
            (
                1,
                "CD",
            ),
            (
                0,
                "NG",
            ),
        ]:

            sub = (
                tmp.loc[
                    tmp[
                        "label"
                    ].eq(
                        label
                    )
                ]
            )


            if sub.empty:

                raise RuntimeError(
                    f"{validation_file}\n"
                    f"No rows for label={label}."
                )


            prevalence_rows.append({

                "gene_set":
                    gene_set,

                "source":
                    source,

                "class":
                    class_name,

                "padding_prevalence":
                    float(
                        sub[
                            "padding_bp"
                        ]
                        .gt(
                            0
                        )
                        .mean()
                    ),
            })


prevalence_df = pd.DataFrame(
    prevalence_rows
)


prevalence_summary = (
    prevalence_df
    .groupby(
        [
            "gene_set",
            "class",
        ],
        observed=True,
    )
    .agg(
        prevalence=(
            "padding_prevalence",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# 7. LOAD NO-PADDING AUROC RESULTS
# ============================================================

if not NO_PADDING_FILE.exists():

    raise FileNotFoundError(
        "No-padding result file not found:\n"
        f"{NO_PADDING_FILE}"
    )


no_padding_df = pd.read_csv(
    NO_PADDING_FILE,
    sep="\t",
)


required_no_padding_columns = {
    "gene_set",
    "source",
    "full_AUROC",
    "no_padding_AUROC",
}

missing = (
    required_no_padding_columns
    - set(
        no_padding_df.columns
    )
)


if missing:

    raise RuntimeError(
        "No-padding result table is missing columns:\n"
        f"{sorted(missing)}"
    )


panel_b_df = (
    no_padding_df.loc[
        no_padding_df[
            "gene_set"
        ].eq(
            NO_PADDING_GENE_SET
        ),
        [
            "source",
            "full_AUROC",
            "no_padding_AUROC",
        ],
    ]
    .copy()
)


expected_sources = set(
    SOURCES
)

actual_sources = set(
    panel_b_df[
        "source"
    ]
)


if actual_sources != expected_sources:

    raise RuntimeError(
        "Unexpected sources in panel (b).\n"
        f"Expected: {sorted(expected_sources)}\n"
        f"Found: {sorted(actual_sources)}"
    )


# ============================================================
# 8. PRINT AUDIT TABLES
# ============================================================

print(
    "\nTerminal-padding prevalence:"
)

display(
    prevalence_summary.round(
        4
    )
)


print(
    "\nNo-padding subset vs original OriGENE AUROC "
    f"({NO_PADDING_GENE_SET}):"
)

display(
    panel_b_df.round(
        4
    )
)


# ============================================================
# 9. FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        FIG_W_MM * MM,
        FIG_H_MM * MM,
    )
)


gs = fig.add_gridspec(
    1,
    2,

    left=
        0.085,

    right=
        0.985,

    top=
        0.90,

    bottom=
        0.31,

    wspace=
        0.34,
)


ax_a = fig.add_subplot(
    gs[
        0,
        0,
    ]
)

ax_b = fig.add_subplot(
    gs[
        0,
        1,
    ]
)


# ============================================================
# 10. PANEL (a) — TERMINAL-PADDING PREVALENCE
#
# Paired-point presentation, matching panel (b).
# ============================================================

x_a = np.arange(
    len(
        GENE_SETS
    )
)

dx_a = 0.09


cd_values = np.asarray(
    [
        100.0
        *
        float(
            prevalence_summary.loc[
                prevalence_summary[
                    "gene_set"
                ].eq(
                    gene_set
                )
                &
                prevalence_summary[
                    "class"
                ].eq(
                    "CD"
                ),
                "prevalence",
            ].iloc[
                0
            ]
        )
        for gene_set
        in GENE_SETS
    ],
    dtype=float,
)


ng_values = np.asarray(
    [
        100.0
        *
        float(
            prevalence_summary.loc[
                prevalence_summary[
                    "gene_set"
                ].eq(
                    gene_set
                )
                &
                prevalence_summary[
                    "class"
                ].eq(
                    "NG"
                ),
                "prevalence",
            ].iloc[
                0
            ]
        )
        for gene_set
        in GENE_SETS
    ],
    dtype=float,
)


for i in range(
    len(
        GENE_SETS
    )
):

    ax_a.plot(
        [
            x_a[i]
            -
            dx_a,

            x_a[i]
            +
            dx_a,
        ],
        [
            cd_values[i],
            ng_values[i],
        ],

        color=
            CONNECTOR_GRAY,

        linewidth=
            CONNECTOR_WIDTH,

        zorder=
            1,
    )


ax_a.scatter(
    x_a
    -
    dx_a,

    cd_values,

    s=
        POINT_SIZE,

    marker=
        "o",

    facecolor=
        BLUE_LIGHT,

    edgecolor=
        BLUE,

    linewidth=
        POINT_EDGE_WIDTH,

    zorder=
        3,
)


ax_a.scatter(
    x_a
    +
    dx_a,

    ng_values,

    s=
        POINT_SIZE,

    marker=
        "o",

    facecolor=
        ORANGE_LIGHT,

    edgecolor=
        ORANGE,

    linewidth=
        POINT_EDGE_WIDTH,

    zorder=
        3,
)


ax_a.set_xlim(
    -0.45,
    2.45,
)


ax_a.set_ylim(
    0,
    80,
)


ax_a.set_yticks(
    [
        0,
        20,
        40,
        60,
        80,
    ]
)


ax_a.set_ylabel(
    "Genes with terminal padding (%)"
)


ax_a.set_xticks(
    x_a
)


ax_a.set_xticklabels(
    [
        GENE_SET_LABELS[
            gene_set
        ]
        for gene_set
        in GENE_SETS
    ]
)


clean_axes(
    ax_a
)


panel_label(
    ax_a,
    "(a)",
)


# ============================================================
# 11. PANEL (b) — NO-PADDING SUBSET VS ORIGINAL AUROC
#
# Same paired-point grammar:
#   blue   = original OriGENE
#   orange = no-padding subset
# ============================================================

x_b = np.arange(
    len(
        SOURCES
    )
)

dx_b = 0.09


panel_b_lookup = (
    panel_b_df
    .set_index(
        "source"
    )
)


original_values = np.asarray(
    [
        float(
            panel_b_lookup.loc[
                source,
                "full_AUROC",
            ]
        )
        for source
        in SOURCES
    ],
    dtype=float,
)


no_padding_values = np.asarray(
    [
        float(
            panel_b_lookup.loc[
                source,
                "no_padding_AUROC",
            ]
        )
        for source
        in SOURCES
    ],
    dtype=float,
)


for i in range(
    len(
        SOURCES
    )
):

    ax_b.plot(
        [
            x_b[i]
            -
            dx_b,

            x_b[i]
            +
            dx_b,
        ],
        [
            original_values[i],
            no_padding_values[i],
        ],

        color=
            CONNECTOR_GRAY,

        linewidth=
            CONNECTOR_WIDTH,

        zorder=
            1,
    )


ax_b.scatter(
    x_b
    -
    dx_b,

    original_values,

    s=
        POINT_SIZE,

    marker=
        "o",

    facecolor=
        BLUE_LIGHT,

    edgecolor=
        BLUE,

    linewidth=
        POINT_EDGE_WIDTH,

    zorder=
        3,
)


ax_b.scatter(
    x_b
    +
    dx_b,

    no_padding_values,

    s=
        POINT_SIZE,

    marker=
        "o",

    facecolor=
        ORANGE_LIGHT,

    edgecolor=
        ORANGE,

    linewidth=
        POINT_EDGE_WIDTH,

    zorder=
        3,
)


ax_b.axhline(
    0.5,

    color=
        CHANCE_GRAY,

    linestyle=
        "--",

    linewidth=
        0.60,

    zorder=
        0,
)


ax_b.set_xlim(
    -0.45,
    3.45,
)


ax_b.set_ylim(
    0.50,
    0.82,
)


ax_b.set_yticks(
    np.arange(
        0.50,
        0.821,
        0.05,
    )
)


ax_b.set_ylabel(
    "AUROC"
)


ax_b.set_xticks(
    x_b
)


ax_b.set_xticklabels(
    [
        SOURCE_LABELS[
            source
        ]
        for source
        in SOURCES
    ]
)


clean_axes(
    ax_b
)


panel_label(
    ax_b,
    "(b)",
)


# ============================================================
# 12. MATCHED PANEL-SPECIFIC LEGENDS
#
# Both legends use the same position, framing, marker size,
# and visual hierarchy.  Sufficient bottom margin is reserved,
# so they do NOT overlap the x-axis labels.
# ============================================================

panel_a_center = (
    ax_a.get_position().x0
    +
    ax_a.get_position().x1
) / 2


panel_b_center = (
    ax_b.get_position().x0
    +
    ax_b.get_position().x1
) / 2


legend_a_handles = [

    Line2D(
        [0],
        [0],

        marker=
            "o",

        linestyle=
            "None",

        markerfacecolor=
            BLUE_LIGHT,

        markeredgecolor=
            BLUE,

        markeredgewidth=
            POINT_EDGE_WIDTH,

        markersize=
            5.2,

        label=
            "CD",
    ),

    Line2D(
        [0],
        [0],

        marker=
            "o",

        linestyle=
            "None",

        markerfacecolor=
            ORANGE_LIGHT,

        markeredgecolor=
            ORANGE,

        markeredgewidth=
            POINT_EDGE_WIDTH,

        markersize=
            5.2,

        label=
            "NG",
    ),
]


legend_b_handles = [

    Line2D(
        [0],
        [0],

        marker=
            "o",

        linestyle=
            "None",

        markerfacecolor=
            BLUE_LIGHT,

        markeredgecolor=
            BLUE,

        markeredgewidth=
            POINT_EDGE_WIDTH,

        markersize=
            5.2,

        label=
            "Original OriGENE",
    ),

    Line2D(
        [0],
        [0],

        marker=
            "o",

        linestyle=
            "None",

        markerfacecolor=
            ORANGE_LIGHT,

        markeredgecolor=
            ORANGE,

        markeredgewidth=
            POINT_EDGE_WIDTH,

        markersize=
            5.2,

        label=
            "No-padding subset",
    ),
]


common_legend_kwargs = dict(

    loc=
        "lower center",

    bbox_transform=
        fig.transFigure,

    ncol=
        2,

    frameon=
        True,

    fancybox=
        False,

    framealpha=
        1.0,

    facecolor=
        "white",

    edgecolor=
        "0.75",

    borderpad=
        0.40,

    handletextpad=
        0.35,

    columnspacing=
        1.0,

    fontsize=
        7.0,
)


legend_a = fig.legend(
    handles=
        legend_a_handles,

    bbox_to_anchor=(
        panel_a_center,
        0.055,
    ),

    **common_legend_kwargs
)


legend_b = fig.legend(
    handles=
        legend_b_handles,

    bbox_to_anchor=(
        panel_b_center,
        0.055,
    ),

    **common_legend_kwargs
)


legend_a.get_frame().set_linewidth(
    0.5
)

legend_b.get_frame().set_linewidth(
    0.5
)


# ============================================================
# 13. GEOMETRY / CLIPPING AUDIT
# ============================================================

fig.canvas.draw()

renderer = (
    fig.canvas
    .get_renderer()
)

fig_bbox = (
    fig.bbox
)

has_clipping = False


for name, ax in [
    (
        "(a)",
        ax_a,
    ),
    (
        "(b)",
        ax_b,
    ),
]:

    bbox = (
        ax.get_tightbbox(
            renderer
        )
    )


    if bbox.x0 < fig_bbox.x0:
        print(
            f"WARNING: panel {name} exceeds LEFT edge."
        )
        has_clipping = True


    if bbox.y0 < fig_bbox.y0:
        print(
            f"WARNING: panel {name} exceeds BOTTOM edge."
        )
        has_clipping = True


    if bbox.x1 > fig_bbox.x1:
        print(
            f"WARNING: panel {name} exceeds RIGHT edge."
        )
        has_clipping = True


    if bbox.y1 > fig_bbox.y1:
        print(
            f"WARNING: panel {name} exceeds TOP edge."
        )
        has_clipping = True


for name, legend in [
    (
        "(a) legend",
        legend_a,
    ),
    (
        "(b) legend",
        legend_b,
    ),
]:

    bbox = (
        legend.get_window_extent(
            renderer=
                renderer
        )
    )


    if bbox.x0 < fig_bbox.x0:
        print(
            f"WARNING: {name} exceeds LEFT edge."
        )
        has_clipping = True


    if bbox.y0 < fig_bbox.y0:
        print(
            f"WARNING: {name} exceeds BOTTOM edge."
        )
        has_clipping = True


    if bbox.x1 > fig_bbox.x1:
        print(
            f"WARNING: {name} exceeds RIGHT edge."
        )
        has_clipping = True


    if bbox.y1 > fig_bbox.y1:
        print(
            f"WARNING: {name} exceeds TOP edge."
        )
        has_clipping = True


if not has_clipping:

    print(
        "\nPASS: all axes and legends are inside the fixed canvas."
    )


# ============================================================
# 14. SAVE
# ============================================================

save_all(
    fig,
    "FigS14_padding_controls_UNIFIED_paired_points",
)


plt.show()
