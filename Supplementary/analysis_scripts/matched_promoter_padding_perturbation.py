"""matched promoter padding perturbation. See Supplementary/README.md."""
import os

def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

# ============================================================
# Supplementary Figure S2
# Matched promoter vs padding perturbation
#
# Panel a:
#   500-bp matched perturbation
#   replacement_scheme = empirical_low_signal
#
# Panel b:
#   Multi-scale matched perturbation
#
# Datasets:
#   Lung I, Lung II, Liver I, Liver II
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. Paths
# ============================================================

BASE = Path(os.environ.get("ORIGENE_PIPELINE_ROOT", "epigenetic_snakemake_pipline")).expanduser().resolve() / "results/07_model_evaluation/reviewer_singletrack"

OLD_ROOT = (
    BASE
    / "interpretability"
    / "original_ng"
    / "all_sources"
)

V3_ROOT = (
    BASE
    / "interpretability"
    / "original_ng"
    / "v3_three_methods"
)

OUT = Path("results/supplementary/matched_promoter_padding_perturbation")

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Dataset definitions
# ============================================================

DATASETS = [
    "Lung_I_Normal",
    "Lung_II_Normal",
    "Liver_I_Normal",
    "Liver_II_Normal",
]

LABELS = {
    "Lung_I_Normal": "Lung I",
    "Lung_II_Normal": "Lung II",
    "Liver_I_Normal": "Liver I",
    "Liver_II_Normal": "Liver II",
}

# matplotlib places y=0 at the bottom.
# Reverse so displayed order is:
# Lung I, Lung II, Liver I, Liver II
ORDER_PANEL_A = DATASETS[::-1]


# ============================================================
# 3. Global figure style
#    Unified with the other supplementary manuscript figures
# ============================================================

MM = 1 / 25.4

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


# ============================================================
# 4. Dataset colors / markers
# ============================================================

color_map = {
    "Lung_I_Normal": "#1f77b4",
    "Lung_II_Normal": "#6baed6",
    "Liver_I_Normal": "#ff7f0e",
    "Liver_II_Normal": "#fdae6b",
}

marker_map = {
    "Lung_I_Normal": "o",
    "Lung_II_Normal": "o",
    "Liver_I_Normal": "^",
    "Liver_II_Normal": "^",
}


# ============================================================
# 5. Shared helpers
# ============================================================

PANEL_Y = 1.035

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, length=3)
    ax.grid(False)

def add_panel_label(ax, label):
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

LEGEND_STYLE = dict(
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    facecolor="white",
    edgecolor="0.75",
    fontsize=7.2,
    borderpad=0.55,
)


# ============================================================
# 6. Load matched 500-bp perturbation
# ============================================================

def load_matched_500bp():

    rows = []

    for ds in DATASETS:

        f = (
            OLD_ROOT
            / ds
            / "matched_window_counterfactual_summary.csv"
        )

        if not f.exists():
            raise FileNotFoundError(
                f"Missing 500-bp perturbation file:\n{f}"
            )

        d = pd.read_csv(f)

        required_columns = {
            "replacement_scheme",
            "perturbation",
            "window_bp",
            "median_abs_delta_logit",
        }

        missing = required_columns - set(d.columns)

        if missing:
            raise KeyError(
                f"\nUnexpected schema in:\n{f}\n\n"
                f"Missing columns: {sorted(missing)}\n"
                f"Available columns: {d.columns.tolist()}"
            )

        # ----------------------------------------------------
        # Use empirical_low_signal.
        # ----------------------------------------------------

        dd = d.loc[
            d["replacement_scheme"].eq(
                "empirical_low_signal"
            )
            &
            pd.to_numeric(
                d["window_bp"],
                errors="coerce"
            ).eq(500)
        ].copy()

        # ----------------------------------------------------
        # Exact perturbation definitions
        # ----------------------------------------------------

        promoter = dd.loc[
            dd["perturbation"].eq(
                "promoter_TSS_centered"
            )
        ]

        padding = dd.loc[
            dd["perturbation"].eq(
                "padding_only_first_window"
            )
        ]

        if len(promoter) != 1 or len(padding) != 1:

            print(
                "\nAvailable empirical_low_signal "
                f"500-bp rows for {ds}:"
            )

            if not dd.empty:
                print(
                    dd[
                        [
                            "replacement_scheme",
                            "perturbation",
                            "window_bp",
                            "median_abs_delta_logit",
                        ]
                    ].to_string(index=False)
                )

            raise RuntimeError(
                f"\nExpected exactly one "
                f"'promoter_TSS_centered' row and one "
                f"'padding_only_first_window' row for {ds}."
            )

        rows.append({
            "dataset": ds,

            "promoter": float(
                promoter.iloc[0][
                    "median_abs_delta_logit"
                ]
            ),

            "padding": float(
                padding.iloc[0][
                    "median_abs_delta_logit"
                ]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. Load multi-scale perturbation
# ============================================================

def load_multiscale():

    frames = []

    required_columns = {
        "window_bp",
        "median_promoter_abs_delta_logit",
        "median_padding_abs_delta_logit",
    }

    for ds in DATASETS:

        f = (
            V3_ROOT
            / ds
            / "multiscale_matched_paired_summary.csv"
        )

        if not f.exists():
            raise FileNotFoundError(
                f"Missing multi-scale perturbation file:\n{f}"
            )

        d = pd.read_csv(f)

        missing = required_columns - set(d.columns)

        if missing:
            raise KeyError(
                f"\nUnexpected schema in:\n{f}\n\n"
                f"Missing columns: {sorted(missing)}\n"
                f"Available columns: {d.columns.tolist()}"
            )

        d = d.copy()

        d["dataset"] = ds

        d["window_bp"] = pd.to_numeric(
            d["window_bp"],
            errors="raise"
        )

        d[
            "median_promoter_abs_delta_logit"
        ] = pd.to_numeric(
            d[
                "median_promoter_abs_delta_logit"
            ],
            errors="raise"
        )

        d[
            "median_padding_abs_delta_logit"
        ] = pd.to_numeric(
            d[
                "median_padding_abs_delta_logit"
            ],
            errors="raise"
        )

        frames.append(d)

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# 8. Load data
# ============================================================

matched = load_matched_500bp()
multi = load_multiscale()


# ============================================================
# 9. Verify audited 500-bp values
# ============================================================

EXPECTED_500 = {

    "Lung_I_Normal": {
        "promoter": 0.164835,
        "padding": 0.031088,
    },

    "Lung_II_Normal": {
        "promoter": 0.330607,
        "padding": 0.039070,
    },

    "Liver_I_Normal": {
        "promoter": 0.192741,
        "padding": 0.029924,
    },

    "Liver_II_Normal": {
        "promoter": 0.153783,
        "padding": 0.038601,
    },
}


for ds in DATASETS:

    r = matched.loc[
        matched["dataset"].eq(ds)
    ].iloc[0]

    assert np.isclose(
        r["promoter"],
        EXPECTED_500[ds]["promoter"],
        atol=1e-6,
    )

    assert np.isclose(
        r["padding"],
        EXPECTED_500[ds]["padding"],
        atol=1e-6,
    )


print("=" * 72)
print("500-bp matched perturbation")
print("replacement_scheme = empirical_low_signal")
print("=" * 72)

print(
    matched.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n500-bp values verified successfully.")


# ============================================================
# 10. Create figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        180 * MM,
        80 * MM,
    )
)


# ============================================================
# PANEL a
# 500-bp matched perturbation
# ============================================================

ax = axes[0]


for yi, ds in enumerate(
    ORDER_PANEL_A
):

    r = matched.loc[
        matched["dataset"].eq(ds)
    ].iloc[0]

    x_padding = float(
        r["padding"]
    )

    x_promoter = float(
        r["promoter"]
    )

    # Connector
    ax.plot(
        [
            x_padding,
            x_promoter,
        ],
        [
            yi,
            yi,
        ],
        color="0.75",
        lw=1.0,
        zorder=1,
    )

    # Padding-only
    ax.scatter(
        x_padding,
        yi,
        s=38,
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=3,
    )

    # TSS-centered
    ax.scatter(
        x_promoter,
        yi,
        s=38,
        facecolors="black",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )


# ------------------------------------------------------------
# Panel a axes
# ------------------------------------------------------------

ax.set_yticks(
    np.arange(
        len(ORDER_PANEL_A)
    )
)

ax.set_yticklabels(
    [
        LABELS[d]
        for d in ORDER_PANEL_A
    ]
)

ax.set_xlabel(
    r"$|\Delta \mathrm{logit}|$"
)


xmin_a = min(
    0.0,
    float(
        matched[
            [
                "padding",
                "promoter",
            ]
        ]
        .min()
        .min()
    ) - 0.01
)

xmax_a = (
    float(
        matched[
            [
                "padding",
                "promoter",
            ]
        ]
        .max()
        .max()
    )
    + 0.025
)

ax.set_xlim(
    xmin_a,
    xmax_a
)


# ------------------------------------------------------------
# Panel a title / label
# ------------------------------------------------------------

ax.set_title(
    "500-bp matched perturbation",
    loc="left",
    x=0.02,
    y=PANEL_Y,
    pad=0,
)

clean_axes(ax)

add_panel_label(
    ax,
    "(a)"
)


# ------------------------------------------------------------
# Panel a legend
# ------------------------------------------------------------

handles_a = [

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        color="black",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=5.5,
        label="TSS-centered",
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=5.5,
        label="Padding-only",
    ),
]


leg_a = ax.legend(
    handles=handles_a,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.27,
    ),
    ncol=2,
    handlelength=1.0,
    columnspacing=1.6,
    **LEGEND_STYLE,
)

leg_a.get_frame().set_linewidth(
    0.6
)


# ============================================================
# PANEL b
# Multi-scale matched perturbation
# ============================================================

ax = axes[1]


for ds in DATASETS:

    dd = (
        multi.loc[
            multi["dataset"].eq(ds)
        ]
        .sort_values(
            "window_bp"
        )
    )

    # TSS-centered
    ax.plot(
        dd["window_bp"],
        dd[
            "median_promoter_abs_delta_logit"
        ],
        color=color_map[ds],
        linestyle="-",
        marker=marker_map[ds],
        markerfacecolor=color_map[ds],
        markeredgecolor=color_map[ds],
        markeredgewidth=0.8,
        ms=4.0,
        lw=1.25,
        zorder=3,
    )

    # Padding-only
    ax.plot(
        dd["window_bp"],
        dd[
            "median_padding_abs_delta_logit"
        ],
        color="0.68",
        linestyle="--",
        marker=marker_map[ds],
        markerfacecolor="white",
        markeredgecolor="0.55",
        markeredgewidth=0.9,
        ms=4.0,
        lw=1.0,
        zorder=2,
    )


# ============================================================
# Across-dataset medians
# ============================================================

cross = (
    multi
    .groupby(
        "window_bp",
        as_index=False,
    )
    .agg(
        promoter_median=(
            "median_promoter_abs_delta_logit",
            "median",
        ),

        padding_median=(
            "median_padding_abs_delta_logit",
            "median",
        ),
    )
    .sort_values(
        "window_bp"
    )
)


# TSS-centered median
ax.plot(
    cross["window_bp"],
    cross["promoter_median"],
    color="black",
    linestyle="-",
    marker="o",
    markerfacecolor="black",
    markeredgecolor="black",
    markeredgewidth=0.8,
    ms=4.5,
    lw=2.0,
    zorder=5,
)


# Padding-only median
ax.plot(
    cross["window_bp"],
    cross["padding_median"],
    color="0.35",
    linestyle="--",
    marker="o",
    markerfacecolor="white",
    markeredgecolor="0.35",
    markeredgewidth=1.0,
    ms=4.5,
    lw=1.7,
    zorder=4,
)


# ============================================================
# Panel b axes
# ============================================================

ax.set_xscale(
    "log",
    base=2,
)

xticks = sorted(
    multi[
        "window_bp"
    ]
    .dropna()
    .unique()
)

ax.set_xticks(
    xticks
)

ax.set_xticklabels(
    [
        str(int(x))
        for x in xticks
    ]
)

ax.set_xlabel(
    "Perturbation window size (bp)"
)

ax.set_ylabel(
    r"$|\Delta \mathrm{logit}|$"
)


# ------------------------------------------------------------
# Panel b title / label
# ------------------------------------------------------------

ax.set_title(
    "Multi-scale matched perturbation",
    loc="left",
    x=0.02,
    y=PANEL_Y,
    pad=0,
)

clean_axes(ax)

add_panel_label(
    ax,
    "(b)"
)


# ============================================================
# Panel b legend
# ============================================================

handles_b = [

    Line2D(
        [0],
        [0],
        color=color_map[
            "Lung_I_Normal"
        ],
        lw=1.4,
        linestyle="-",
        marker="o",
        markerfacecolor=color_map[
            "Lung_I_Normal"
        ],
        markeredgecolor=color_map[
            "Lung_I_Normal"
        ],
        markersize=4,
        label="Lung I",
    ),

    Line2D(
        [0],
        [0],
        color=color_map[
            "Lung_II_Normal"
        ],
        lw=1.4,
        linestyle="-",
        marker="o",
        markerfacecolor=color_map[
            "Lung_II_Normal"
        ],
        markeredgecolor=color_map[
            "Lung_II_Normal"
        ],
        markersize=4,
        label="Lung II",
    ),

    Line2D(
        [0],
        [0],
        color=color_map[
            "Liver_I_Normal"
        ],
        lw=1.4,
        linestyle="-",
        marker="^",
        markerfacecolor=color_map[
            "Liver_I_Normal"
        ],
        markeredgecolor=color_map[
            "Liver_I_Normal"
        ],
        markersize=4,
        label="Liver I",
    ),

    Line2D(
        [0],
        [0],
        color=color_map[
            "Liver_II_Normal"
        ],
        lw=1.4,
        linestyle="-",
        marker="^",
        markerfacecolor=color_map[
            "Liver_II_Normal"
        ],
        markeredgecolor=color_map[
            "Liver_II_Normal"
        ],
        markersize=4,
        label="Liver II",
    ),

    Line2D(
        [0],
        [0],
        color="black",
        lw=2.0,
        linestyle="-",
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=4.5,
        label="TSS-centered median",
    ),

    Line2D(
        [0],
        [0],
        color="0.35",
        lw=1.7,
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="0.35",
        markersize=4.5,
        label="Padding-only median",
    ),
]


leg_b = ax.legend(
    handles=handles_b,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.27,
    ),
    ncol=3,
    handlelength=1.8,
    columnspacing=1.0,
    **LEGEND_STYLE,
)

leg_b.get_frame().set_linewidth(
    0.6
)


# ============================================================
# 11. Layout
# ============================================================

fig.subplots_adjust(
    left=0.12,
    right=0.99,
    top=0.82,
    bottom=0.40,
    wspace=0.46,
)


# ============================================================
# 12. Save publication outputs
# ============================================================

STEM = (
    "FigS2_matched_perturbation_Lung_Liver"
)

PNG_FILE = (
    OUT
    / f"{STEM}_600dpi.png"
)

SVG_FILE = (
    OUT
    / f"{STEM}.svg"
)

PDF_FILE = (
    OUT
    / f"{STEM}.pdf"
)

EPS_FILE = (
    OUT
    / f"{STEM}.eps"
)


# 600-dpi PNG
fig.savefig(
    PNG_FILE,
    format="png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
    facecolor="white",
)

# SVG
# Additional export omitted; retain canonical PNG.

# PDF
# Additional export omitted; retain canonical PNG.

# EPS
# Additional export omitted; retain canonical PNG.


print("\nSaved:")
print(PNG_FILE)
print(SVG_FILE)
print(PDF_FILE)
print(EPS_FILE)


plt.show()
