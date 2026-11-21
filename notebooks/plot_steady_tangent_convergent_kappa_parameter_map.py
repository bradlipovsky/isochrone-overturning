"""Plot the bed--surface relief-ratio experiment for the convergent basin.

The five phase maps hold mean thickness and total span fixed while varying

    kappa = Delta s / Delta b.

``kappa=1`` is the original parallel-boundary, constant-thickness geometry.
The mean-thickness anchor keeps ice area, ``L/H``, and ``beta H/eta`` fixed:

    b = Delta b f,
    s = H + (1-kappa) Delta b / 2 + kappa Delta b f,

where the two piecewise-sine flanks have ``<f>=1/2``.  Consequently
``<s-b>=H`` for every kappa.  ``kappa=0`` is shown only in the schematic: a
flat free surface gives an exactly hydrostatic, stagnant solution, so there
is no depositing source family on which to define the steady diagnostic.

This file is deliberately plotting-only and does not import Firedrake.
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
    / "data/steady_tangent_convergent_kappa_parameter_map.csv"
)
FIGURE_PATH = (
    REPOSITORY_DIRECTORY
    / "figures/steady_tangent_convergent_kappa_parameter_map.png"
)
REPORT_PATH = (
    REPOSITORY_DIRECTORY / "notebooks/steady_tangent_convergent_kappa_parameter_map.md"
)
METADATA_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_kappa_parameter_map_metadata.json"
)
VALIDATION_PATH = (
    REPOSITORY_DIRECTORY / "data/steady_tangent_convergent_kappa_validation.csv"
)

KAPPA_COLUMN = "surface_to_bed_relief_ratio_kappa"
LENGTH_RATIO_COLUMN = "right_to_left_length_ratio"
RELIEF_RATIO_COLUMN = "relief_to_thickness_ratio"
MINIMUM_COLUMN = "minimum_resolved_J"
STATUS_COLUMN = "status"
METHOD_COLUMN = "diagnostic_method_version"
EXPECTED_METHOD_VERSION = (
    "convergent-steady-v4-kappa-meanH-central90-local-ablation-buffer"
)

MEAN_THICKNESS_M = 200.0
TOTAL_LENGTH_M = 6_000.0
KAPPA_VALUES = np.array((0.10, 0.25, 0.50, 0.75, 1.00))
LENGTH_RATIOS = np.r_[np.arange(1.0, 11.0), 12.0]
RELIEF_RATIOS = np.arange(0.25, 4.01, 0.25)

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
    "kappa_zero": "#6a51a3",
    "kappa_half": "#2b8cbe",
}
J_CMAP = LinearSegmentedColormap.from_list(
    "negative_red_zero_white_positive_blue",
    ("#67001f", "#d6604d", "#ffffff", "#4393c3", "#053061"),
    N=257,
)
J_NORM = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
MASS_BALANCE_CLOSURE_LIMIT = 1.0e-3
ENERGY_RESIDUAL_LIMIT = 1.0e-3
BASAL_NORMAL_SPEED_LIMIT_M_PER_YR = 1.0e-4


def configure_matplotlib() -> None:
    """Match the two existing geometry-plus-five-map figures."""
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


def bool_value(value) -> bool:
    """Interpret CSV booleans without treating the string 'False' as true."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def load_table(path: Path) -> pd.DataFrame:
    """Load and validate one restartable kappa sweep."""
    table = pd.read_csv(path)
    required = {
        KAPPA_COLUMN,
        LENGTH_RATIO_COLUMN,
        RELIEF_RATIO_COLUMN,
        MINIMUM_COLUMN,
        STATUS_COLUMN,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    if METHOD_COLUMN in table:
        methods = table[METHOD_COLUMN].dropna().astype(str).unique()
        if len(methods) != 1 or methods[0] != EXPECTED_METHOD_VERSION:
            raise ValueError(
                f"expected only method {EXPECTED_METHOD_VERSION!r}, "
                f"found {methods.tolist()}"
            )
    for column in (
        KAPPA_COLUMN,
        LENGTH_RATIO_COLUMN,
        RELIEF_RATIO_COLUMN,
        MINIMUM_COLUMN,
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table[KAPPA_COLUMN] = canonicalize(table[KAPPA_COLUMN], KAPPA_VALUES)
    table[LENGTH_RATIO_COLUMN] = canonicalize(
        table[LENGTH_RATIO_COLUMN], LENGTH_RATIOS
    )
    table[RELIEF_RATIO_COLUMN] = canonicalize(
        table[RELIEF_RATIO_COLUMN], RELIEF_RATIOS
    )
    return (
        table.sort_index()
        .drop_duplicates(
            subset=(KAPPA_COLUMN, LENGTH_RATIO_COLUMN, RELIEF_RATIO_COLUMN),
            keep="last",
        )
        .sort_values([KAPPA_COLUMN, RELIEF_RATIO_COLUMN, LENGTH_RATIO_COLUMN])
        .reset_index(drop=True)
    )


def validate_table(table: pd.DataFrame) -> None:
    """Fail closed if a partial or internally inconsistent sweep is plotted."""
    def validate_mechanics(frame: pd.DataFrame, label: str) -> None:
        positive = frame["integrated_positive_mass_balance_m2_per_yr"].abs()
        negative = frame["integrated_negative_mass_balance_m2_per_yr"].abs()
        closure = (
            frame["integrated_positive_mass_balance_m2_per_yr"]
            + frame["integrated_negative_mass_balance_m2_per_yr"]
        ).abs() / (positive + negative).replace(0.0, np.nan)
        if not np.isfinite(closure).all() or closure.max() > (
            MASS_BALANCE_CLOSURE_LIMIT
        ):
            raise ValueError(f"{label} fails mass-balance closure")
        energy = frame["relative_energy_residual"].abs()
        if not np.isfinite(energy).all() or energy.max() > ENERGY_RESIDUAL_LIMIT:
            raise ValueError(f"{label} fails the Stokes energy check")
        basal = frame["rms_basal_normal_speed_m_per_yr"].abs()
        if not np.isfinite(basal).all() or basal.max() > (
            BASAL_NORMAL_SPEED_LIMIT_M_PER_YR
        ):
            raise ValueError(f"{label} fails the basal-normal-speed check")

    expected_coordinates = {
        (float(kappa), float(q_value), float(relief_ratio))
        for kappa in KAPPA_VALUES
        for q_value in LENGTH_RATIOS
        for relief_ratio in RELIEF_RATIOS
    }
    actual_coordinates = {
        (float(kappa), float(q_value), float(relief_ratio))
        for kappa, q_value, relief_ratio in table[
            [KAPPA_COLUMN, LENGTH_RATIO_COLUMN, RELIEF_RATIO_COLUMN]
        ].itertuples(index=False, name=None)
    }
    if actual_coordinates != expected_coordinates or len(table) != len(
        expected_coordinates
    ):
        missing = len(expected_coordinates.difference(actual_coordinates))
        extra = len(actual_coordinates.difference(expected_coordinates))
        raise ValueError(
            "kappa sweep is incomplete or off-grid: "
            f"{len(table)} rows, {missing} missing coordinates, "
            f"{extra} extra coordinates"
        )

    expected_statuses = {"valid", "invalid_nonpositive_thickness"}
    statuses = set(table[STATUS_COLUMN].astype(str))
    if not statuses.issubset(expected_statuses):
        raise ValueError(f"unresolved solver statuses remain: {sorted(statuses)}")

    valid_mask = table[STATUS_COLUMN].astype(str).eq("valid")
    invalid_mask = table[STATUS_COLUMN].astype(str).eq(
        "invalid_nonpositive_thickness"
    )
    if not np.isfinite(table.loc[valid_mask, MINIMUM_COLUMN]).all():
        raise ValueError("a valid cell has non-finite minimum J")
    if not (table.loc[valid_mask, "minimum_thickness_m"] > 0.0).all():
        raise ValueError("a valid cell has nonpositive thickness")
    if not (table.loc[invalid_mask, "minimum_thickness_m"] <= 0.0).all():
        raise ValueError("an analytically invalid cell has positive thickness")
    if table.loc[valid_mask, "safety_cap_trajectory_count"].fillna(0).sum():
        raise ValueError("one or more trajectories reached the safety cap")
    if table.loc[valid_mask, "numerical_failure_trajectory_count"].fillna(
        0
    ).sum():
        raise ValueError("one or more trajectories failed numerically")
    if table.loc[
        valid_mask, "path_counterflow_velocity_threshold_m_per_yr"
    ].isna().any():
        raise ValueError("a valid cell lacks its counterflow threshold")
    if not table.loc[invalid_mask, "solution_provenance"].astype(str).eq(
        "analytical_invalid_nonpositive_thickness"
    ).all():
        raise ValueError("an invalid cell lacks analytical provenance")
    validate_mechanics(table.loc[valid_mask], "production sweep")

    if VALIDATION_PATH.exists():
        validation = pd.read_csv(VALIDATION_PATH)
        if len(validation) != 15:
            raise ValueError(
                f"expected 15 validation variants, found {len(validation)}"
            )
        if not validation[STATUS_COLUMN].astype(str).eq("valid").all():
            raise ValueError("a targeted validation variant is not valid")
        if not np.isfinite(validation[MINIMUM_COLUMN]).all():
            raise ValueError("a targeted validation variant has non-finite J")
        for scenario, group in validation.groupby("scenario"):
            if not (group[MINIMUM_COLUMN] < 0.0).all():
                raise ValueError(
                    f"validation scenario {scenario!r} does not preserve sign"
                )
        for column in (
            "safety_cap_trajectory_count",
            "numerical_failure_trajectory_count",
        ):
            if validation[column].fillna(0).sum():
                raise ValueError(f"targeted validation has nonzero {column}")
        validate_mechanics(validation, "targeted validation")


def logarithmic_cell_edges(values):
    values = np.asarray(values, dtype=float)
    middle = np.sqrt(values[:-1] * values[1:])
    return np.r_[
        values[0] ** 2 / middle[0],
        middle,
        values[-1] ** 2 / middle[-1],
    ]


def linear_cell_edges(values):
    values = np.asarray(values, dtype=float)
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[
        2.0 * values[0] - middle[0],
        middle,
        2.0 * values[-1] - middle[-1],
    ]


def minimum_thickness_ratio(kappa, relief_ratio):
    """Exact h_min/H for the mean-thickness parameterization."""
    return 1.0 - 0.5 * abs(1.0 - kappa) * relief_ratio


def status_category(row) -> str:
    """Map detailed solver output into a stable plotting category."""
    status = str(row.get(STATUS_COLUMN, "missing")).lower()
    valid = bool_value(row.get("valid", np.isfinite(row[MINIMUM_COLUMN])))
    if valid and np.isfinite(row[MINIMUM_COLUMN]):
        return "resolved"
    if "invalid" in status or "nonpositive" in status:
        return "invalid"
    if "censor" in status or "safety_cap" in status:
        return "censored"
    if status in ("nan", "", "missing", "not_run"):
        return "missing"
    return "failed"


def grid_for_kappa(table: pd.DataFrame, kappa: float):
    """Pivot one fixed-kappa slice without interpolating missing cells."""
    subset = table[np.isclose(table[KAPPA_COLUMN], kappa)].copy()
    # Use the production coordinates even when analytically invalid rows were
    # intentionally not sent to the Stokes solver.
    q_values = np.sort(
        np.unique(np.r_[LENGTH_RATIOS, subset[LENGTH_RATIO_COLUMN].dropna()])
    )
    reliefs = np.sort(
        np.unique(np.r_[RELIEF_RATIOS, subset[RELIEF_RATIO_COLUMN].dropna()])
    )
    values = np.full((len(reliefs), len(q_values)), np.nan)
    categories = np.full(values.shape, "missing", dtype=object)
    indexed = subset.set_index([RELIEF_RATIO_COLUMN, LENGTH_RATIO_COLUMN])
    for iy, relief in enumerate(reliefs):
        for ix, q_value in enumerate(q_values):
            if minimum_thickness_ratio(kappa, relief) <= 0.0:
                categories[iy, ix] = "invalid"
                continue
            try:
                row = indexed.loc[(relief, q_value)]
            except KeyError:
                continue
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            values[iy, ix] = float(row[MINIMUM_COLUMN])
            categories[iy, ix] = status_category(row)
    categories[~np.isfinite(values) & (categories == "resolved")] = "failed"
    return q_values, reliefs, values, categories


def flank_fraction(x, left_length, right_length):
    """Dimensionless piecewise-sine relief, with exact mean one half."""
    x = np.asarray(x)
    return np.where(
        x <= 0.0,
        np.sin(0.5 * np.pi * (-x) / left_length) ** 2,
        np.sin(0.5 * np.pi * x / right_length) ** 2,
    )


def draw_geometry_schematic(axis: plt.Axes) -> None:
    """Show how kappa changes the surface while preserving mean thickness."""
    q_value = 3.0
    left_length = 1.0 / (1.0 + q_value)
    right_length = q_value / (1.0 + q_value)
    relief = 0.50
    mean_thickness = 0.66
    x = np.linspace(-left_length, right_length, 1001)
    fraction = flank_fraction(x, left_length, right_length)
    bed = relief * fraction

    def surface(kappa):
        return (
            mean_thickness
            + 0.5 * (1.0 - kappa) * relief
            + kappa * bed
        )

    representative_surface = surface(0.5)
    axis.fill_between(x, -0.20, bed, color=COLORS["bed"], zorder=0)
    axis.fill_between(
        x, bed, representative_surface, color=COLORS["ice"], zorder=1
    )
    axis.plot(x, bed, color=COLORS["ink"], lw=1.5, zorder=4)
    styles = {
        0.0: (COLORS["kappa_zero"], (0, (2, 2))),
        0.5: (COLORS["kappa_half"], (0, (5, 2))),
        1.0: (COLORS["positive"], "-"),
    }
    for kappa, (color, linestyle) in styles.items():
        axis.plot(
            x,
            surface(kappa),
            color=color,
            ls=linestyle,
            lw=1.45,
            zorder=5,
        )
        axis.text(
            right_length - 0.018,
            surface(kappa)[-1],
            rf"$\kappa={kappa:g}$",
            color=color,
            va="center",
            ha="right",
            fontsize=7.3,
        )

    arrow = dict(
        arrowstyle="<->", color=COLORS["ink"], lw=0.85, shrinkA=0, shrinkB=0
    )
    guide = dict(color=COLORS["muted"], lw=0.65, ls=(0, (2, 2)))

    # f=1/2 at the midpoint of either flank.  Every kappa surface therefore
    # intersects here and the local thickness is exactly the domain mean H.
    thickness_x = 0.5 * right_length
    thickness_bed = 0.5 * relief
    thickness_surface = thickness_bed + mean_thickness
    axis.annotate(
        "",
        xy=(thickness_x, thickness_surface),
        xytext=(thickness_x, thickness_bed),
        arrowprops=arrow,
    )
    axis.text(
        thickness_x + 0.025,
        0.5 * (thickness_bed + thickness_surface),
        r"$H=\langle h\rangle$",
        ha="left",
        va="center",
    )

    relief_x = -left_length - 0.045
    axis.plot((relief_x, -left_length), (0.0, 0.0), **guide)
    axis.plot((relief_x, -left_length), (relief, relief), **guide)
    axis.annotate(
        "", xy=(relief_x, relief), xytext=(relief_x, 0.0), arrowprops=arrow
    )
    axis.text(
        relief_x - 0.016,
        0.5 * relief,
        r"$\Delta b$",
        ha="right",
        va="center",
    )

    surface_relief_x = right_length + 0.12
    half_surface = surface(0.5)
    axis.plot(
        (right_length, surface_relief_x),
        (half_surface[0], half_surface[0]),
        **guide,
    )
    axis.plot(
        (right_length, surface_relief_x),
        (half_surface[-1], half_surface[-1]),
        **guide,
    )
    axis.annotate(
        "",
        xy=(surface_relief_x, half_surface[-1]),
        xytext=(surface_relief_x, half_surface[0]),
        arrowprops=arrow,
    )
    axis.text(
        surface_relief_x + 0.018,
        0.5 * (half_surface[0] + half_surface[-1]),
        r"$\Delta s=\kappa\Delta b$",
        ha="left",
        va="center",
    )

    axis.text(
        0.5,
        -0.24,
        r"$b=\Delta b f(x)$;  "
        r"$s=H+(1-\kappa)\Delta b/2+\kappa\Delta b f(x)$"
        "\n"
        rf"$\langle h\rangle=H={MEAN_THICKNESS_M:g}\,\mathrm{{m}}$;  "
        rf"$L/H={TOTAL_LENGTH_M / MEAN_THICKNESS_M:g}$;  "
        r"$h_{\min}/H=1-|1-\kappa|(\Delta b/H)/2$"
        "\n"
        r"$\kappa=0$: horizontal surface, hydrostatic stagnation ($J$ undefined)",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.25,
    )
    axis.set(xlim=(-0.39, 1.08), ylim=(-0.38, 1.27))
    axis.axis("off")


def status_underlay(axis, x_edges, y_edges, categories):
    """Paint unresolved cells without assigning a numerical J value."""
    palette = {
        "invalid": COLORS["invalid"],
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


def draw_heatmap(axis, table, kappa, panel_title):
    """Draw one fixed-kappa phase map."""
    q_values, reliefs, values, categories = grid_for_kappa(table, kappa)
    x_edges = logarithmic_cell_edges(q_values)
    x_edges[0] = 1.0  # q=1 is the exact reflection plane.
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

    invalid_rows = np.all(categories == "invalid", axis=1)
    if np.any(invalid_rows):
        first_invalid_index = int(np.flatnonzero(invalid_rows)[0])
        invalid_band_bottom = y_edges[first_invalid_index]
        two_line_label = y_edges[-1] - invalid_band_bottom >= 0.5
        label_y = (
            0.5 * (invalid_band_bottom + y_edges[-1])
            if two_line_label
            else reliefs[first_invalid_index] - 0.04
        )
        axis.text(
            np.sqrt(x_edges[0] * x_edges[-1]),
            label_y,
            (
                "nonpositive\nice thickness"
                if two_line_label
                else r"invalid: $h_{\min}\leq0$"
            ),
            ha="center",
            va="center",
            color=COLORS["muted"],
            fontsize=(7.2 if two_line_label else 6.7),
            zorder=5,
        )
    return image


def status_legend(table):
    """Return only non-geometric status categories present in the table."""
    categories = {status_category(row) for _, row in table.iterrows()}
    definitions = {
        "censored": (COLORS["censored"], "age-censored"),
        "failed": (COLORS["failed"], "unresolved numerical cell"),
        "missing": (COLORS["missing"], "missing"),
    }
    return [
        Patch(
            facecolor=definitions[key][0],
            edgecolor="none",
            label=definitions[key][1],
        )
        for key in ("censored", "failed", "missing")
        if key in categories
    ]


def save_figure(table: pd.DataFrame, path: Path, show=False) -> None:
    """Save the geometry-plus-five-kappa-map PNG/PDF pair."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9.1, 5.95),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.90, 1.0)},
    )
    draw_geometry_schematic(axes[0, 0])
    axes[0, 0].set_title("A   Bed--surface coupling", loc="left", pad=4.0)

    map_axes = (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2])
    images = []
    for letter, kappa, axis in zip("BCDEF", KAPPA_VALUES, map_axes):
        images.append(
            draw_heatmap(
                axis,
                table,
                kappa,
                panel_title=rf"{letter}   $\kappa={kappa:.2f}$",
            )
        )

    for index, axis in enumerate(map_axes):
        if index in (2, 3, 4):
            axis.set_xlabel(
                r"Flank-length ratio, $q=L_{\rm long}/L_{\rm short}$"
            )
        if index in (0, 2):
            axis.set_ylabel(r"Bed-relief ratio, $\Delta b/H$")
        else:
            axis.set_yticklabels([])

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
        r"Minimum steady-characteristic $J=T_x$ outside the local two-cell "
        r"ablation buffer (central 90% of depositing flux)"
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
    if show:
        plt.show()
    plt.close(figure)


def valid_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return mechanically valid cells with a finite primary diagnostic."""
    valid = table[STATUS_COLUMN].astype(str).eq("valid")
    return table.loc[valid & np.isfinite(table[MINIMUM_COLUMN])].copy()


def classify(values: pd.Series) -> pd.Series:
    """Apply the a-priori resolved/unresolved thresholds."""
    return pd.Series(
        np.where(
            values < -0.05,
            "overturned",
            np.where(values > 0.05, "ordered", "unresolved"),
        ),
        index=values.index,
    )


def validation_ranges():
    """Load compact sign-stability ranges when validation is available."""
    if not VALIDATION_PATH.exists():
        return pd.DataFrame()
    validation = pd.read_csv(VALIDATION_PATH)
    if validation.empty:
        return validation
    return (
        validation.groupby("scenario")[MINIMUM_COLUMN]
        .agg(["min", "max", "count"])
        .reset_index()
    )


def write_report(table: pd.DataFrame) -> None:
    """Write a concise, data-derived interpretation beside the scripts."""
    valid = valid_rows(table)
    valid["classification"] = classify(valid[MINIMUM_COLUMN])
    rows = []
    for kappa in KAPPA_VALUES:
        subset = valid[np.isclose(valid[KAPPA_COLUMN], kappa)]
        counts = subset["classification"].value_counts()
        rows.append(
            {
                "kappa": kappa,
                "valid": len(subset),
                "overturned": int(counts.get("overturned", 0)),
                "broad_overturned": int(
                    np.count_nonzero(
                        subset[
                            "resolved_per_source_minimum_J_1st_percentile"
                        ]
                        < -0.05
                    )
                ),
                "unresolved": int(counts.get("unresolved", 0)),
                "ordered": int(counts.get("ordered", 0)),
                "minimum": float(subset[MINIMUM_COLUMN].min()),
            }
        )
    common = valid[valid[RELIEF_RATIO_COLUMN] <= 2.0].copy()
    common_counts = []
    for kappa in KAPPA_VALUES:
        subset = common[np.isclose(common[KAPPA_COLUMN], kappa)]
        counts = classify(subset[MINIMUM_COLUMN]).value_counts()
        common_counts.append(
            (
                kappa,
                int(counts.get("overturned", 0)),
                int(counts.get("unresolved", 0)),
                int(counts.get("ordered", 0)),
            )
        )
    pivot = common.pivot_table(
        index=(LENGTH_RATIO_COLUMN, RELIEF_RATIO_COLUMN),
        columns=KAPPA_COLUMN,
        values=MINIMUM_COLUMN,
    )
    comparisons = []
    for kappa in KAPPA_VALUES[:-1]:
        paired = pivot[[kappa, 1.0]].dropna()
        comparisons.append(
            (
                kappa,
                int(np.count_nonzero(paired[kappa] < paired[1.0])),
                len(paired),
                float(np.median(paired[kappa] - paired[1.0])),
            )
        )
    monotone_count = int(
        np.count_nonzero(
            np.all(
                np.diff(pivot[KAPPA_VALUES].to_numpy(dtype=float), axis=1)
                >= 0.0,
                axis=1,
            )
        )
    )
    validations = validation_ranges()
    retry_mask = valid["solution_provenance"].astype(str).str.contains(
        "point_location_retry", na=False
    )
    interior_mask = (
        (valid["sampler_sigma_lower"] > 0.002)
        | (valid["sampler_sigma_upper"] < 0.999)
    )
    source_extrapolation_mask = valid["sampler_sigma_upper"] < 0.995
    lines = [
        "# Steady convergent-basin bed--surface coupling sweep",
        "",
        "## Parameterization",
        "",
        "The calculation fixes total length at 6 km and domain-mean vertical "
        "thickness at 200 m, so L/H=30 and beta H/eta=0.08.  For the "
        "piecewise-sine flank shape f with exact mean 1/2,",
        "",
        "\\[b=\\Delta b f,\\qquad "
        "s=H+(1-\\kappa)\\Delta b/2+\\kappa\\Delta b f.\\]",
        "",
        "Thus kappa=Delta s/Delta b=1 is the previous constant-thickness "
        "geometry.  Lower kappa flattens the surface while preserving ice "
        "area: the confluence thickens and the divides thin.  Positive "
        "thickness requires",
        "",
        "\\[h_{\\min}/H=1-|1-\\kappa|(\\Delta b/H)/2>0.\\]",
        "",
        "The kappa=0 endpoint is exactly hydrostatic and stagnant.  It has "
        "no depositing source family, so steady J is undefined; the solved "
        "maps begin at kappa=0.10.",
        "",
        "Every positive-thickness case uses the central 5--95% of depositing "
        "surface flux and a local two-vertical-cell ablation buffer.  The "
        "validated time step scales as 1/kappa.  Low-kappa safety caps were "
        "enlarged as 240 kyr/kappa; the uncensored reused kappa=1 baseline "
        "retains its validated 120 kyr cap.",
        "",
        "## Result",
        "",
        f"The grid has {len(table)} cells: {len(valid)} resolved positive-"
        f"thickness cases and {len(table) - len(valid)} nonpositive-thickness "
        "geometries.  No trajectory reaches the safety cap.",
        "",
        "| kappa | valid | overturned | broad-source overturned | unresolved "
        "| ordered | minimum J |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['kappa']:.2f} | {row['valid']} | {row['overturned']} | "
        f"{row['broad_overturned']} | {row['unresolved']} | "
        f"{row['ordered']} | {row['minimum']:.6g} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            "The common positive-thickness rectangle Delta b/H<=2 contains "
            "the same 88 cells in every slice:",
            "",
            "| kappa | overturned | unresolved | ordered |",
            "|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {kappa:.2f} | {overturned} | {unresolved} | {ordered} |"
        for kappa, overturned, unresolved, ordered in common_counts
    )
    lines.extend(
        [
            "",
            "On this paired domain, reducing kappa makes J smaller in nearly "
            "every cell relative to kappa=1:",
            "",
            "| kappa | cells with J below kappa=1 | paired cells | median "
            "Delta J |",
            "|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {kappa:.2f} | {smaller} | {count} | {delta:.6g} |"
        for kappa, smaller, count, delta in comparisons
    )
    lines.extend(
        [
            "",
            f"This is not a strict monotonic law: {monotone_count}/88 paired "
            "cells are monotone across all five sampled kappa values.  The "
            "comparison to kappa=1 is nevertheless systematic.",
            "",
            "For 0.10<=kappa<=1, flattening the surface therefore does not "
            "remove the folding mechanism in this controlled family.  "
            "Dimensional speed and required mass balance weaken approximately "
            "with kappa, while residence ages grow approximately as 1/kappa.  "
            "Those two scalings largely cancel in accumulated steady strain; "
            "the J shift instead reflects the changing normalized strain and "
            "streamline geometry, the flatter initial surface tangent and the "
            "divide-to-confluence thickness redistribution.  This experiment "
            "does not separate those effects, and the one-parameter kappa "
            "coupling still forces the bed and surface extrema to coincide "
            "on both flanks.",
            "",
            "The full-map counts are not directly comparable because the "
            "low-kappa slices lose high-relief cells to nonpositive thickness. "
            "The paired Delta b/H<=2 comparison is the controlled result.  "
            "Cells nearest the positivity boundary have only 12.5--20 m "
            "minimum divide thickness; their negative signs validate, but "
            "they are a thin-divide limit.  The plotted isolated minimum asks "
            "whether any retained source overturns; the broad-source counts "
            "show that exact onset remains sensitive to source width.",
            "",
            "## Targeted validation",
            "",
        ]
    )
    if validations.empty:
        lines.append("Validation output is not present.")
    else:
        lines.extend(
            [
                "| scenario | minimum J across variants | maximum J across "
                "variants | variants |",
                "|---|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {row.scenario} | {row['min']:.6g} | {row['max']:.6g} | "
            f"{int(row['count'])} |"
            for _, row in validations.iterrows()
        )
        lines.extend(
            [
                "",
                "All 15 validation variants pass the mechanical diagnostics "
                "and preserve the sign, including the low-kappa onset, the "
                "thin positive-thickness edge and the cells requiring a more "
                "interior velocity sampler.",
                "",
                f"Point-location retries were required in {int(retry_mask.sum())} "
                f"of {len(valid)} resolved cells; {int(interior_mask.sum())} use "
                "interior sampler bounds.  Most offsets are slight; the widest "
                f"bounds are sigma={valid['sampler_sigma_lower'].max():g} to "
                f"{valid['sampler_sigma_upper'].min():g}.  "
                f"{int(source_extrapolation_mask.sum())} cells have an upper "
                "sampler bound below the sigma=0.995 source level and therefore "
                "extrapolate there; their negative signs are more robust than "
                "their most saturated minimum-J magnitudes.  Mesh and sampler "
                "variants preserve the signs in the targeted retry scenarios.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `figures/steady_tangent_convergent_kappa_parameter_map.png`",
            "- `figures/steady_tangent_convergent_kappa_parameter_map.pdf`",
            "- `data/steady_tangent_convergent_kappa_parameter_map.csv`",
            "- `data/steady_tangent_convergent_kappa_validation.csv`",
            "",
            "No manuscript source is modified by this analysis.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def write_metadata(table: pd.DataFrame) -> None:
    """Record the fixed controls and completeness checks as JSON."""
    valid = valid_rows(table)
    closure = (
        valid["integrated_positive_mass_balance_m2_per_yr"]
        + valid["integrated_negative_mass_balance_m2_per_yr"]
    ).abs()
    scale = (
        valid["integrated_positive_mass_balance_m2_per_yr"].abs()
        + valid["integrated_negative_mass_balance_m2_per_yr"].abs()
    )
    retry_mask = valid["solution_provenance"].astype(str).str.contains(
        "point_location_retry", na=False
    )
    interior_mask = (
        (valid["sampler_sigma_lower"] > 0.002)
        | (valid["sampler_sigma_upper"] < 0.999)
    )
    source_extrapolation_mask = valid["sampler_sigma_upper"] < 0.995
    metadata = {
        "method_version": EXPECTED_METHOD_VERSION,
        "mean_thickness_m": MEAN_THICKNESS_M,
        "total_length_m": TOTAL_LENGTH_M,
        "total_length_to_mean_thickness_ratio": (
            TOTAL_LENGTH_M / MEAN_THICKNESS_M
        ),
        "kappa_values": KAPPA_VALUES.tolist(),
        "length_ratios": LENGTH_RATIOS.tolist(),
        "relief_ratios": RELIEF_RATIOS.tolist(),
        "expected_cells": int(
            len(KAPPA_VALUES) * len(LENGTH_RATIOS) * len(RELIEF_RATIOS)
        ),
        "table_cells": int(len(table)),
        "resolved_positive_thickness_cells": int(len(valid)),
        "nonpositive_thickness_cells": int(len(table) - len(valid)),
        "safety_cap_trajectory_count": int(
            valid["safety_cap_trajectory_count"].fillna(0).sum()
        ),
        "point_location_retry_cells": int(retry_mask.sum()),
        "interior_sampler_bound_cells": int(interior_mask.sum()),
        "source_level_extrapolation_cells": int(
            source_extrapolation_mask.sum()
        ),
        "maximum_sampler_sigma_lower": float(
            valid["sampler_sigma_lower"].max()
        ),
        "minimum_sampler_sigma_upper": float(
            valid["sampler_sigma_upper"].min()
        ),
        "maximum_relative_mass_balance_closure": float(
            (closure / scale.replace(0.0, np.nan)).max()
        ),
        "maximum_absolute_relative_energy_residual": float(
            valid["relative_energy_residual"].abs().max()
        ),
        "maximum_rms_basal_normal_speed_m_per_yr": float(
            valid["rms_basal_normal_speed_m_per_yr"].max()
        ),
        "validation_variants": int(
            len(pd.read_csv(VALIDATION_PATH)) if VALIDATION_PATH.exists() else 0
        ),
        "kappa_zero_limit": (
            "horizontal traction-free surface; hydrostatic zero velocity; "
            "steady surface-origin J undefined"
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=FIGURE_PATH)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    configure_matplotlib()
    table = load_table(arguments.data)
    validate_table(table)
    save_figure(table, arguments.output, show=arguments.show)
    write_report(table)
    write_metadata(table)


if __name__ == "__main__":
    main()
