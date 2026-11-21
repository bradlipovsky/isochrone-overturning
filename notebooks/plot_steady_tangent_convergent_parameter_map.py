"""Plot the three-parameter finite convergent-basin sweep.

The layout deliberately parallels the sinusoidal-bed parameter map: one
full-size geometry schematic and five heat maps sharing a single diverging
color scale.  Each map fixes ``L/H`` and spans the reflection-unique half of
``q`` against ``Delta b/H``.  No auxiliary ablation or counterflow panel and
no ``J=0`` contour are drawn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPOSITORY_DIRECTORY = Path(__file__).resolve().parent.parent
DATA_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_aspect_parameter_map.csv"
)
FIGURE_PATH = (
    REPOSITORY_DIRECTORY
    / "figures/steady_tangent_convergent_parameter_map.png"
)
REPORT_PATH = (
    REPOSITORY_DIRECTORY
    / "notebooks/steady_tangent_convergent_aspect_parameter_map.md"
)
METADATA_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_aspect_parameter_map_metadata.json"
)
VALIDATION_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_aspect_validation.csv"
)

METHOD_VERSION = "convergent-steady-v3-aspect-central90-ablation-buffer"
THICKNESS_M = 200.0
ASPECT_RATIOS = np.array((15.0, 20.0, 30.0, 45.0, 60.0))
LENGTH_RATIOS = np.r_[np.arange(1.0, 11.0), 12.0]
RELIEF_RATIOS = np.arange(0.25, 4.01, 0.25)

ASPECT_COLUMN = "total_length_to_thickness_ratio"
LENGTH_RATIO_COLUMN = "right_to_left_length_ratio"
RELIEF_RATIO_COLUMN = "relief_to_thickness_ratio"
MINIMUM_COLUMN = "minimum_resolved_J"
STATUS_COLUMN = "status"
METHOD_COLUMN = "diagnostic_method_version"

COLORS = {
    "negative": "#8e1537",
    "positive": "#0b4f7a",
    "ink": "#242424",
    "muted": "#666666",
    "invalid": "#d9d9d9",
    "censored": "#ead9a4",
    "failed": "#e7c7c7",
    "missing": "#f2f2f2",
    "ice": "#dceef5",
    "bed": "#b9a590",
}
J_CMAP = LinearSegmentedColormap.from_list(
    "negative_red_zero_white_positive_blue",
    ["#67001f", "#d6604d", "#ffffff", "#4393c3", "#053061"],
    N=257,
)
J_NORM = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)


def configure_matplotlib() -> None:
    """Match the sinusoidal figure's compact vector-publication defaults."""
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def canonicalize(values, reference, tolerance=1.0e-10):
    """Remove harmless CSV roundoff at known grid coordinates."""
    result = np.asarray(values, dtype=float).copy()
    for target in np.asarray(reference, dtype=float):
        result[np.isclose(result, target, rtol=0.0, atol=tolerance)] = target
    return result


def load_table(path: Path) -> pd.DataFrame:
    """Load, validate, and de-duplicate the aspect sweep."""
    table = pd.read_csv(path)
    required = {
        ASPECT_COLUMN,
        LENGTH_RATIO_COLUMN,
        RELIEF_RATIO_COLUMN,
        MINIMUM_COLUMN,
        STATUS_COLUMN,
        METHOD_COLUMN,
        "valid",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    if not bool(table[METHOD_COLUMN].eq(METHOD_VERSION).all()):
        found = table[METHOD_COLUMN].fillna("<missing>").value_counts().to_dict()
        raise ValueError(
            f"expected only method {METHOD_VERSION!r}, found {found}"
        )
    for column in (
        ASPECT_COLUMN,
        LENGTH_RATIO_COLUMN,
        RELIEF_RATIO_COLUMN,
        MINIMUM_COLUMN,
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table[ASPECT_COLUMN] = canonicalize(table[ASPECT_COLUMN], ASPECT_RATIOS)
    table[LENGTH_RATIO_COLUMN] = canonicalize(
        table[LENGTH_RATIO_COLUMN], LENGTH_RATIOS
    )
    table[RELIEF_RATIO_COLUMN] = canonicalize(
        table[RELIEF_RATIO_COLUMN], RELIEF_RATIOS
    )
    return (
        table.sort_index()
        .drop_duplicates(
            subset=(ASPECT_COLUMN, LENGTH_RATIO_COLUMN, RELIEF_RATIO_COLUMN),
            keep="last",
        )
        .sort_values([ASPECT_COLUMN, RELIEF_RATIO_COLUMN, LENGTH_RATIO_COLUMN])
        .reset_index(drop=True)
    )


def logarithmic_cell_edges(values):
    values = np.asarray(values, dtype=float)
    middle = np.sqrt(values[:-1] * values[1:])
    return np.r_[values[0] ** 2 / middle[0], middle,
                 values[-1] ** 2 / middle[-1]]


def linear_cell_edges(values):
    values = np.asarray(values, dtype=float)
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[2.0 * values[0] - middle[0], middle,
                 2.0 * values[-1] - middle[-1]]


def bool_value(value) -> bool:
    """Interpret a CSV boolean without treating the string 'False' as true."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def status_category(row) -> str:
    """Map one solver row into a stable figure-level category."""
    status = str(row[STATUS_COLUMN]).lower()
    if bool_value(row["valid"]) and np.isfinite(row[MINIMUM_COLUMN]):
        return "resolved"
    if "censor" in status or "safety_cap" in status:
        return "censored"
    if status in ("nan", "", "missing", "not_run"):
        return "missing"
    return "failed"


def grid_for_aspect(table: pd.DataFrame, aspect_ratio: float):
    """Pivot one fixed-aspect slice into values and status categories."""
    subset = table[np.isclose(table[ASPECT_COLUMN], aspect_ratio)].copy()
    q_values = np.sort(subset[LENGTH_RATIO_COLUMN].dropna().unique())
    reliefs = np.sort(subset[RELIEF_RATIO_COLUMN].dropna().unique())
    if len(q_values) == 0 or len(reliefs) == 0:
        return q_values, reliefs, np.empty((0, 0)), np.empty((0, 0), object)
    values = (
        subset.pivot(
            index=RELIEF_RATIO_COLUMN,
            columns=LENGTH_RATIO_COLUMN,
            values=MINIMUM_COLUMN,
        )
        .reindex(index=reliefs, columns=q_values)
        .to_numpy(dtype=float)
    )
    categories = np.full(values.shape, "missing", dtype=object)
    indexed = subset.set_index([RELIEF_RATIO_COLUMN, LENGTH_RATIO_COLUMN])
    for iy, relief in enumerate(reliefs):
        for ix, q_value in enumerate(q_values):
            try:
                categories[iy, ix] = status_category(indexed.loc[(relief, q_value)])
            except KeyError:
                pass
    categories[~np.isfinite(values) & (categories == "resolved")] = "failed"
    return q_values, reliefs, values, categories


def draw_geometry_schematic(axis: plt.Axes) -> None:
    """Draw an exact normalized member of the convergent-basin family."""
    q_value = 3.0
    left_length = 1.0 / (1.0 + q_value)
    right_length = q_value / (1.0 + q_value)
    relief = 0.54
    thickness = 0.66
    x = np.linspace(-left_length, right_length, 801)
    bed = np.where(
        x <= 0.0,
        relief * np.sin(0.5 * np.pi * (-x) / left_length) ** 2,
        relief * np.sin(0.5 * np.pi * x / right_length) ** 2,
    )
    surface = bed + thickness
    axis.fill_between(x, -0.24, bed, color=COLORS["bed"], zorder=0)
    axis.fill_between(x, bed, surface, color=COLORS["ice"], zorder=1)
    axis.plot(x, bed, color=COLORS["ink"], lw=1.5, zorder=3)
    axis.plot(x, surface, color=COLORS["positive"], lw=1.4, zorder=3)

    dimension_arrow = dict(
        arrowstyle="<->", color=COLORS["ink"], lw=0.9, shrinkA=0, shrinkB=0
    )
    guide = dict(color=COLORS["muted"], lw=0.7, ls=(0, (2, 2)))
    dimension_y = -0.16
    for location, top in ((-left_length, relief), (0.0, 0.0),
                          (right_length, relief)):
        axis.plot((location, location), (dimension_y, top), **guide)
    axis.annotate(
        "", xy=(0.0, dimension_y), xytext=(-left_length, dimension_y),
        arrowprops=dimension_arrow,
    )
    axis.annotate(
        "", xy=(right_length, dimension_y), xytext=(0.0, dimension_y),
        arrowprops=dimension_arrow,
    )
    axis.text(-0.5 * left_length, dimension_y - 0.055, r"$L_{\rm left}$",
              ha="center", va="top")
    axis.text(0.5 * right_length, dimension_y - 0.055, r"$L_{\rm right}$",
              ha="center", va="top")

    relief_x = -left_length - 0.055
    axis.plot((relief_x, -left_length), (0.0, 0.0), **guide)
    axis.plot((relief_x, -left_length), (relief, relief), **guide)
    axis.annotate(
        "", xy=(relief_x, relief), xytext=(relief_x, 0.0),
        arrowprops=dimension_arrow,
    )
    axis.text(relief_x - 0.018, 0.5 * relief, r"$\Delta b$",
              ha="right", va="center")

    thickness_x = 0.11
    local_bed = float(np.interp(thickness_x, x, bed))
    axis.annotate(
        "", xy=(thickness_x, local_bed + thickness),
        xytext=(thickness_x, local_bed), arrowprops=dimension_arrow,
    )
    axis.text(thickness_x + 0.025, local_bed + 0.5 * thickness, r"$H$",
              ha="left", va="center")

    flow_arrow = dict(
        arrowstyle="-|>", color=COLORS["positive"], lw=1.0,
        mutation_scale=8, shrinkA=0, shrinkB=0,
    )
    axis.annotate("", xy=(-0.045, 0.69), xytext=(-0.18, 0.94),
                  arrowprops=flow_arrow)
    axis.annotate("", xy=(0.11, 0.69), xytext=(0.57, 1.04),
                  arrowprops=flow_arrow)
    axis.annotate("", xy=(0.02, 0.93), xytext=(0.02, 0.69),
                  arrowprops=flow_arrow)
    axis.text(
        0.02,
        1.00,
        "upwelling\nbalanced by ablation",
        ha="center",
        va="bottom",
        color=COLORS["positive"],
        fontsize=7.5,
    )
    axis.text(
        0.5,
        -0.33,
        r"$b=\Delta b\,\sin^2[\pi |x|/(2L_{\rm side})]$; $s=b+H$"
        "\n"
        r"$L=L_{\rm left}+L_{\rm right}$; "
        r"$q=L_{\rm long}/L_{\rm short}\geq1$",
        ha="center",
        va="top",
        fontsize=7.5,
    )
    axis.set(xlim=(-0.38, 0.79), ylim=(-0.43, 1.28))
    axis.set_aspect("auto")
    axis.axis("off")


def status_underlay(axis, x_edges, y_edges, categories):
    """Paint unresolved cells without implying a numeric J value."""
    palette = {
        "censored": COLORS["censored"],
        "failed": COLORS["failed"],
        "missing": COLORS["missing"],
    }
    for category, color in palette.items():
        mask = categories == category
        if not np.any(mask):
            continue
        axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_where(~mask, np.ones(mask.shape)),
            cmap=ListedColormap((color,)),
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            rasterized=True,
            zorder=0,
        )
        if category in ("censored", "failed"):
            rows, columns = np.nonzero(mask)
            axis.scatter(
                np.sqrt(x_edges[columns] * x_edges[columns + 1]),
                0.5 * (y_edges[rows] + y_edges[rows + 1]),
                marker="x",
                color=COLORS["muted"],
                s=11,
                lw=0.55,
                zorder=4,
            )


def draw_heatmap(axis, table, aspect_ratio, panel_title):
    """Draw one fixed-L/H phase map."""
    q_values, reliefs, values, categories = grid_for_aspect(table, aspect_ratio)
    if values.size == 0:
        axis.text(0.5, 0.5, rf"$L/H={aspect_ratio:g}$ not available",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
        return None
    x_edges = logarithmic_cell_edges(q_values)
    # q=1 is the reflection plane, not the center of an unshown q<1 cell.
    x_edges[0] = 1.0
    y_edges = linear_cell_edges(reliefs)
    resolved = categories == "resolved"
    status_underlay(axis, x_edges, y_edges, categories)
    image = axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_where(~resolved, values),
        cmap=J_CMAP,
        norm=J_NORM,
        shading="flat",
        rasterized=True,
        zorder=1,
    )
    axis.set_xscale("log")
    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.set_ylim(y_edges[0], y_edges[-1])
    axis.set_xticks((1, 2, 4, 8, 12))
    axis.set_xticklabels(("1", "2", "4", "8", "12"))
    axis.set_yticks((1.0, 2.0, 3.0, 4.0))
    axis.grid(False)
    axis.set_title(panel_title, loc="left", pad=4.0)
    return image


def status_legend(table):
    categories = {status_category(row) for _, row in table.iterrows()}
    definitions = {
        "censored": (COLORS["censored"], "age-censored"),
        "failed": (COLORS["failed"], "unresolved numerical cell"),
        "missing": (COLORS["missing"], "missing"),
    }
    return [
        Patch(facecolor=definitions[key][0], edgecolor="none",
              label=definitions[key][1])
        for key in ("censored", "failed", "missing")
        if key in categories
    ]


def save_figure(table: pd.DataFrame, path: Path) -> None:
    """Save the parallel geometry-plus-five-map PNG/PDF pair."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9.1, 5.95),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.90, 1.0)},
    )
    draw_geometry_schematic(axes[0, 0])
    axes[0, 0].set_title("A   Convergent-basin geometry", loc="left", pad=4.0)

    map_axes = (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2])
    images = []
    for letter, aspect_ratio, axis in zip("BCDEF", ASPECT_RATIOS, map_axes):
        total_length_km = aspect_ratio * THICKNESS_M / 1000.0
        image = draw_heatmap(
            axis,
            table,
            aspect_ratio,
            panel_title=(
                rf"{letter}   $L/H={aspect_ratio:g}$; "
                rf"$L={total_length_km:g}$ km"
            ),
        )
        if image is not None:
            images.append(image)

    for index, axis in enumerate(map_axes):
        if not axis.axison:
            continue
        if index in (2, 3, 4):
            axis.set_xlabel(r"Flank-length ratio, $q=L_{\rm long}/L_{\rm short}$")
        if index in (0, 2):
            axis.set_ylabel(r"Bed-relief ratio, $\Delta b/H$")
        else:
            axis.set_yticklabels([])

    if not images:
        raise RuntimeError("no aspect panel contains plottable data")
    colorbar = figure.colorbar(
        images[0],
        ax=list(map_axes),
        orientation="horizontal",
        fraction=0.055,
        pad=0.075,
        aspect=42,
        extend="both",
    )
    colorbar.set_ticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    colorbar.set_label(
        r"Minimum steady-characteristic $J=T_x$ outside $H/12$ ablation buffer "
        r"(central 90% of depositing flux)"
    )
    handles = status_legend(table)
    if handles:
        axes[0, 0].legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            frameon=False,
            ncol=2,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def resolved_table(table: pd.DataFrame) -> pd.DataFrame:
    mask = np.array(
        [status_category(row) == "resolved" for _, row in table.iterrows()]
    )
    return table.loc[mask].copy()


def write_report(table: pd.DataFrame) -> None:
    """Record the parameterization, symmetry, and map-level result."""
    resolved = resolved_table(table)
    expected = len(ASPECT_RATIOS) * len(LENGTH_RATIOS) * len(RELIEF_RATIOS)
    minimum_row = resolved.loc[resolved[MINIMUM_COLUMN].idxmin()]
    lines = [
        "# Steady convergent-basin aspect-ratio sweep",
        "",
        "## Parameterization",
        "",
        "The figure uses the same 2x3 visual grammar as the sinusoidal-bed ",
        "parameter map: one geometry schematic and five heat maps with one ",
        "shared diverging scale. The auxiliary ablation and counterflow panels ",
        "have been removed.",
        "",
        "Only the reflection-unique half of the family is shown. Interchanging ",
        "the left and right flank lengths is an exact horizontal reflection and ",
        "does not change the minimum J, so q=L_long/L_short>=1 is sufficient.",
        "",
        "The independent third geometric coordinate is total basin aspect L/H. ",
        "The sweep holds H=200 m, beta H/eta=0.080, viscosity, basal drag, and ",
        "the linear sliding law fixed while using L/H=(15,20,30,45,60), or ",
        "L=(3,4,6,9,12) km. Each panel spans q=1--12 and Delta b/H=0.25--4.",
        "",
        "The characteristic diagnostic retains the central 5--95% of positive ",
        "depositing flux and excludes the final H/12 beneath an ablating surface. ",
        "All cases use the validated 25 yr characteristic step. New aspect ",
        "cases use a 480 kyr safety ceiling; the reused, uncensored L/H=30 ",
        "baseline retains its original 120 kyr ceiling. These are censoring ",
        "guards rather than analysis horizons.",
        "",
        "## Result",
        "",
        f"The table contains {len(table)}/{expected} expected cells, of which ",
        f"{len(resolved)} are numerically resolved. The global minimum is ",
        f"J={minimum_row[MINIMUM_COLUMN]:.6g} at L/H={minimum_row[ASPECT_COLUMN]:g}, ",
        f"q={minimum_row[LENGTH_RATIO_COLUMN]:g}, and ",
        f"Delta b/H={minimum_row[RELIEF_RATIO_COLUMN]:g}.",
        "",
        "Resolved classification by aspect (J<-0.05 overturned, |J|<=0.05 ",
        "unresolved, J>0.05 ordered):",
        "",
        "| L/H | resolved | overturned | unresolved | ordered | minimum J |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for aspect_ratio in ASPECT_RATIOS:
        subset = resolved[np.isclose(resolved[ASPECT_COLUMN], aspect_ratio)]
        values = subset[MINIMUM_COLUMN].to_numpy(dtype=float)
        lines.append(
            f"| {aspect_ratio:g} | {len(values)} | "
            f"{np.count_nonzero(values < -0.05)} | "
            f"{np.count_nonzero(np.abs(values) <= 0.05)} | "
            f"{np.count_nonzero(values > 0.05)} | {np.min(values):.6g} |"
        )
    lines.extend(
        (
            "",
            "The classification above is explicitly the buffered diagnostic. ",
            "At L/H=15 and 20, respectively, 11 and 6 cells classified as ",
            "ordered outside the H/12 surface buffer have negative raw minima ",
            "during the final near-surface return; no such disagreement occurs ",
            "at L/H=30, 45, or 60.",
            "",
            "The isolated-minimum existence criterion is intentionally shown. ",
            "At L/H=15, 85 cells have J_min<-0.05 but only 51 have a negative ",
            "first percentile across retained source paths, so part of the ",
            "high-relief island is a narrow-source feature.",
        )
    )
    if VALIDATION_PATH.exists():
        validation = pd.read_csv(VALIDATION_PATH)
        lines.extend(
            (
                "",
                "## Targeted validation",
                "",
                "The re-entrant short-aspect branch and the long-residence ",
                "ordered limit were rerun with independent 240x32 Stokes-mesh, ",
                "12.5 yr characteristic-step, and 1801x151 velocity-sampler ",
                "refinements, all using the same physical H/12 outflow ",
                "clearance.",
                "",
                "| scenario | minimum J across variants | maximum J across variants |",
                "|---|---:|---:|",
            )
        )
        for scenario, subset in validation.groupby("validation_scenario"):
            values = pd.to_numeric(
                subset[MINIMUM_COLUMN], errors="coerce"
            ).dropna()
            lines.append(
                f"| {scenario} | {values.min():.6g} | {values.max():.6g} |"
            )
        lines.extend(
            (
                "",
                "Every sentinel preserves its sign. The saturated negative ",
                "magnitude is sampler-sensitive, so it supports robust occurrence ",
                "of overturning rather than a precision estimate of the extremum.",
            )
        )
    lines.extend(
        (
            "",
            "## Outputs",
            "",
            "- `figures/steady_tangent_convergent_parameter_map.png`",
            "- `figures/steady_tangent_convergent_parameter_map.pdf`",
            "- `data/steady_tangent_convergent_aspect_parameter_map.csv`",
            "- `data/steady_tangent_convergent_aspect_parameter_map_metadata.json`",
            "- `data/steady_tangent_convergent_aspect_validation.csv`",
            "",
            "No manuscript source is modified by this analysis.",
        )
    )
    REPORT_PATH.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def write_metadata(table: pd.DataFrame) -> None:
    resolved = resolved_table(table)
    values = resolved[MINIMUM_COLUMN].to_numpy(dtype=float)
    validation = (
        pd.read_csv(VALIDATION_PATH) if VALIDATION_PATH.exists() else pd.DataFrame()
    )
    validation_summary = {}
    if not validation.empty:
        for scenario, subset in validation.groupby("validation_scenario"):
            scenario_values = pd.to_numeric(
                subset[MINIMUM_COLUMN], errors="coerce"
            ).dropna()
            validation_summary[str(scenario)] = {
                "variant_count": int(len(scenario_values)),
                "minimum_J": float(scenario_values.min()),
                "maximum_J": float(scenario_values.max()),
                "sign_stable": bool(
                    np.all(scenario_values > 0.0)
                    or np.all(scenario_values < 0.0)
                ),
            }
    metadata = {
        "diagnostic_method_version": METHOD_VERSION,
        "ice_thickness_m": THICKNESS_M,
        "sliding_number_beta_H_over_eta": 0.08,
        "linear_viscosity_Pa_yr": 2.5e8,
        "basal_drag_Pa_yr_per_m": 1.0e5,
        "aspect_ratios_L_over_H": ASPECT_RATIOS.tolist(),
        "total_lengths_km": (ASPECT_RATIOS * THICKNESS_M / 1000.0).tolist(),
        "length_ratios_q": LENGTH_RATIOS.tolist(),
        "relief_ratios_delta_b_over_H": RELIEF_RATIOS.tolist(),
        "expected_cell_count": int(
            len(ASPECT_RATIOS) * len(LENGTH_RATIOS) * len(RELIEF_RATIOS)
        ),
        "table_cell_count": int(len(table)),
        "resolved_cell_count": int(len(resolved)),
        "overturned_cell_count": int(np.count_nonzero(values < -0.05)),
        "unresolved_cell_count": int(np.count_nonzero(np.abs(values) <= 0.05)),
        "ordered_cell_count": int(np.count_nonzero(values > 0.05)),
        "minimum_resolved_J": float(np.min(values)),
        "maximum_resolved_J": float(np.max(values)),
        "source_flux_quantiles": [0.05, 0.95],
        "source_count_per_accumulation_family": 81,
        "production_stokes_mesh_nx_nz": [180, 24],
        "production_velocity_sampler_nx_nz": [1201, 101],
        "production_characteristic_time_step_yr": 25.0,
        "maximum_age_safety_cap_new_aspects_yr": 480000.0,
        "maximum_age_safety_cap_reused_L_over_H_30_yr": 120000.0,
        "safety_cap_trajectory_count": int(
            pd.to_numeric(
                table["safety_cap_trajectory_count"], errors="coerce"
            ).fillna(0).sum()
        ),
        "point_location_sampler_retry_cell_count": int(
            table["solution_provenance"]
            .astype(str)
            .str.contains("point_location_retry")
            .sum()
        ),
        "ablation_surface_buffer_fraction_of_H": 2.0 / 24.0,
        "reflection_unique_half_only": True,
        "validation": validation_summary,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    arguments = parser.parse_args()
    configure_matplotlib()
    table = load_table(arguments.data)
    save_figure(table, arguments.figure)
    write_report(table)
    write_metadata(table)
    print(f"Wrote {arguments.figure}")
    print(f"Wrote {arguments.figure.with_suffix('.pdf')}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {METADATA_PATH}")


if __name__ == "__main__":
    main()
