"""Plot the three-parameter steady-characteristic sinusoidal-bed sweep.

The production table is written by
``steady_characteristic_sinusoidal_parameter_sweep.py``.  This plotting file
does not import Firedrake and can therefore be rerun with an ordinary Python
environment after the sweep has finished.

Two figures are produced:

* ``steady_characteristic_sinusoidal_parameter_map`` combines a precise
  one-wavelength geometry schematic with one ``A/H``--``lambda/H`` map for
  each sampled mean thickness.
* ``steady_characteristic_sinusoidal_limits`` fixes the central production
  thickness, uses row/column sections plus two flow diagnostics to distinguish
  wavelength, flat-bed, and thin-neck limits, and optionally adds the separate
  ``lambda/H = 0.5`` and 24 and ``A/H = 0.90`` and 0.95 audit cases.  A dashed
  curve retains the raw minimum to exit for the selected row.

No ``J=0`` contour is drawn.  Every heat map uses the same diverging scale,
with ``J=0`` white.  Non-resolved cells are shown explicitly rather than
being silently interpolated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


H_COLUMN = "mean ice thickness (m)"
EPSILON_COLUMN = "amplitude / mean thickness"
WAVELENGTH_COLUMN = "wavelength / mean thickness"
MINIMUM_COLUMN = "minimum J"
STATUS_COLUMN = "status"
METHOD_COLUMN = "diagnostic method version"
EXPECTED_METHOD_VERSION = (
    "sinusoidal-steady-v2-exact-surface-ablation-only-buffer"
)

EXPECTED_THICKNESSES = np.array((50.0, 100.0, 160.0, 240.0, 320.0))
EXPECTED_AMPLITUDE_RATIOS = np.array(
    (0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35,
     0.45, 0.55, 0.65, 0.75, 0.85)
)
EXPECTED_WAVELENGTH_RATIOS = np.array(
    (1.0 / 12.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0,
     1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
     10.0, 12.0)
)

COLORS = {
    "negative": "#8e1537",
    "negative_light": "#d6604d",
    "positive_light": "#67a9cf",
    "positive": "#0b4f7a",
    "ink": "#242424",
    "muted": "#666666",
    "grid": "#d5d5d5",
    "invalid": "#d9d9d9",
    "censored": "#ead9a4",
    "failed": "#e7c7c7",
    "missing": "#f2f2f2",
    "screened": "#c9dce8",
    "ice": "#dceef5",
    "bed": "#b9a590",
    "accent": "#6a51a3",
    "exchange": "#bd6b20",
}

J_CMAP = LinearSegmentedColormap.from_list(
    "negative_red_zero_white_positive_blue",
    ["#67001f", "#d6604d", "#ffffff", "#4393c3", "#053061"],
    N=257,
)
J_NORM = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)


def configure_matplotlib():
    """Set compact, editable-vector publication defaults."""
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


def canonicalize_grid_coordinate(values, reference, tolerance=1.0e-10):
    """Remove harmless CSV roundoff while preserving intentional extensions."""
    result = np.asarray(values, dtype=float).copy()
    for target in np.asarray(reference, dtype=float):
        result[np.isclose(result, target, rtol=0.0, atol=tolerance)] = target
    return result


def load_table(path):
    """Load, type-check, and de-duplicate restartable sweep output."""
    table = pd.read_csv(path)
    required = {
        H_COLUMN,
        EPSILON_COLUMN,
        WAVELENGTH_COLUMN,
        MINIMUM_COLUMN,
        STATUS_COLUMN,
        METHOD_COLUMN,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    current_method = table[METHOD_COLUMN].eq(EXPECTED_METHOD_VERSION)
    if not bool(current_method.all()):
        counts = table[METHOD_COLUMN].fillna("<missing>").value_counts()
        raise ValueError(
            f"{path} mixes or lacks diagnostic method versions; expected "
            f"{EXPECTED_METHOD_VERSION!r} in every row, found "
            f"{counts.to_dict()}"
        )
    for column in (H_COLUMN, EPSILON_COLUMN, WAVELENGTH_COLUMN, MINIMUM_COLUMN):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table[H_COLUMN] = canonicalize_grid_coordinate(
        table[H_COLUMN], EXPECTED_THICKNESSES
    )
    table[EPSILON_COLUMN] = canonicalize_grid_coordinate(
        table[EPSILON_COLUMN], EXPECTED_AMPLITUDE_RATIOS
    )
    table[WAVELENGTH_COLUMN] = canonicalize_grid_coordinate(
        table[WAVELENGTH_COLUMN], EXPECTED_WAVELENGTH_RATIOS
    )
    table = (
        table.sort_index()
        .drop_duplicates(
            subset=(H_COLUMN, EPSILON_COLUMN, WAVELENGTH_COLUMN), keep="last"
        )
        .sort_values([H_COLUMN, EPSILON_COLUMN, WAVELENGTH_COLUMN])
        .reset_index(drop=True)
    )
    return table


def resolved_status(status):
    """Return True for numerical successes and the analytic flat-bed row."""
    text = str(status)
    return text == "success" or text.startswith("analytic flat-bed limit")


def status_category(status):
    """Map detailed solver statuses to figure-level categories."""
    text = str(status).lower()
    if resolved_status(status):
        return "resolved"
    if "screened below numerical surface signal" in text:
        return "screened"
    if text.startswith("invalid"):
        return "invalid"
    if "censor" in text or "safety-cap" in text:
        return "censored"
    if text in ("nan", "", "missing"):
        return "missing"
    return "failed"


def check_completeness(table):
    """Report absent production cells; permit intentional grid extensions."""
    expected = {
        (float(h), float(epsilon), float(wavelength))
        for h in EXPECTED_THICKNESSES
        for epsilon in EXPECTED_AMPLITUDE_RATIOS
        for wavelength in EXPECTED_WAVELENGTH_RATIOS
    }
    observed = {
        (float(row[H_COLUMN]), float(row[EPSILON_COLUMN]),
         float(row[WAVELENGTH_COLUMN]))
        for _, row in table.iterrows()
        if np.all(
            np.isfinite(
                (row[H_COLUMN], row[EPSILON_COLUMN], row[WAVELENGTH_COLUMN])
            )
        )
    }
    messages = []
    if expected - observed:
        messages.append(f"{len(expected - observed)} production cells are absent")
    return messages


def linear_cell_edges(values, lower_bound=None, upper_bound=None):
    """Cell edges centered on a monotonically increasing linear coordinate."""
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        half = max(0.05, 0.1 * abs(values[0]))
        result = np.array((values[0] - half, values[0] + half))
    else:
        middle = 0.5 * (values[:-1] + values[1:])
        result = np.r_[2.0 * values[0] - middle[0], middle,
                       2.0 * values[-1] - middle[-1]]
    if lower_bound is not None:
        result[0] = max(float(lower_bound), result[0])
    if upper_bound is not None:
        result[-1] = min(float(upper_bound), result[-1])
    return result


def logarithmic_cell_edges(values):
    """Positive cell edges centered geometrically for a logarithmic axis."""
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("logarithmic coordinate must be positive")
    if len(values) == 1:
        return values[0] * np.array((1.0 / np.sqrt(1.25), np.sqrt(1.25)))
    middle = np.sqrt(values[:-1] * values[1:])
    return np.r_[values[0] ** 2 / middle[0], middle,
                 values[-1] ** 2 / middle[-1]]


def grid_for_thickness(table, mean_thickness):
    """Pivot one fixed-H slice into values and status categories."""
    subset = table[np.isclose(table[H_COLUMN], mean_thickness)].copy()
    wavelengths = np.sort(subset[WAVELENGTH_COLUMN].dropna().unique())
    amplitudes = np.sort(subset[EPSILON_COLUMN].dropna().unique())
    if len(wavelengths) == 0 or len(amplitudes) == 0:
        return wavelengths, amplitudes, np.empty((0, 0)), np.empty((0, 0), object)
    values = (
        subset.pivot(index=EPSILON_COLUMN, columns=WAVELENGTH_COLUMN,
                     values=MINIMUM_COLUMN)
        .reindex(index=amplitudes, columns=wavelengths)
        .to_numpy(dtype=float)
    )
    statuses = (
        subset.pivot(index=EPSILON_COLUMN, columns=WAVELENGTH_COLUMN,
                     values=STATUS_COLUMN)
        .reindex(index=amplitudes, columns=wavelengths)
        .fillna("missing")
        .to_numpy(dtype=object)
    )
    categories = np.vectorize(status_category, otypes=(object,))(statuses)
    categories[~np.isfinite(values) & (categories == "resolved")] = "failed"
    return wavelengths, amplitudes, values, categories


def draw_geometry_schematic(axis, epsilon=0.32):
    """Draw one exact nondimensional sinusoidal model cell."""
    x = np.linspace(0.0, 1.0, 801)
    bed = -1.0 + epsilon * np.cos(2.0 * np.pi * x)
    axis.fill_between(x, bed, 0.0, color=COLORS["ice"], zorder=1)
    # Stop the bed fill above the equation band so labels never compete with
    # the mathematical definition of the geometry.
    axis.fill_between(x, -1.46, bed, color=COLORS["bed"], zorder=0)
    axis.plot(x, bed, color=COLORS["ink"], lw=1.5, zorder=3)
    axis.plot((0.0, 1.0), (0.0, 0.0), color=COLORS["positive"], lw=1.4)
    axis.axhline(-1.0, color=COLORS["muted"], lw=0.8, ls=(0, (2, 2)))

    arrow = dict(arrowstyle="<->", color=COLORS["ink"], lw=0.9,
                 shrinkA=0, shrinkB=0)
    axis.annotate("", xy=(0.0, 0.16), xytext=(1.0, 0.16), arrowprops=arrow)
    axis.text(0.5, 0.19, r"one period, $\lambda$", ha="center", va="bottom")
    # At x/lambda=1/4 the sinusoid crosses its mean, so this arrow is exactly H.
    axis.annotate("", xy=(0.25, 0.0), xytext=(0.25, -1.0), arrowprops=arrow)
    axis.text(0.275, -0.50, r"$H$", ha="left", va="center")
    axis.annotate(
        "", xy=(1.0, -1.0 + epsilon), xytext=(1.0, -1.0),
        arrowprops=arrow,
    )
    axis.text(0.975, -1.0 + 0.5 * epsilon, r"$A$", ha="right", va="center")

    axis.text(0.015, -0.04, "flat surface", ha="left", va="top",
              color=COLORS["positive"])
    axis.text(0.38, -0.96, r"mean bed: $z=-H$", ha="center", va="bottom",
              color=COLORS["muted"])
    axis.text(0.02, -1.0 + epsilon + 0.035, r"$H-A$", ha="left", va="bottom")
    axis.text(0.50, -1.0 - epsilon + 0.10, r"thickness $H+A$",
              ha="center", va="bottom")
    axis.text(
        0.5,
        -1.80,
        r"$b(x)=-H+A\,\cos(2\pi x/\lambda)$"
        "\n"
        r"relief $=2A$; $\max|b_x|=2\pi A/\lambda$",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    axis.text(
        0.5,
        -0.24,
        "ice",
        ha="center",
        va="center",
        color=COLORS["positive"],
        fontsize=8.0,
    )
    axis.set(xlim=(-0.03, 1.03), ylim=(-1.84, 0.31))
    axis.set_aspect(0.52)
    axis.axis("off")


def _status_underlay(axis, x_edges, y_edges, categories):
    """Paint every unresolved grid cell with a stable categorical color."""
    palette = {
        "invalid": COLORS["invalid"],
        "censored": COLORS["censored"],
        "failed": COLORS["failed"],
        "missing": COLORS["missing"],
        "screened": COLORS["screened"],
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
        if category in ("censored", "failed", "screened"):
            rows, columns = np.nonzero(mask)
            axis.scatter(
                np.sqrt(x_edges[columns] * x_edges[columns + 1]),
                0.5 * (y_edges[rows] + y_edges[rows + 1]),
                marker=("." if category == "screened" else "x"),
                color=COLORS["muted"],
                s=(7 if category == "screened" else 11),
                lw=0.55,
                zorder=4,
            )


def draw_heatmap(axis, table, mean_thickness, panel_title=None):
    """Draw one fixed-H map and return its ScalarMappable."""
    wavelengths, amplitudes, values, categories = grid_for_thickness(
        table, mean_thickness
    )
    if values.size == 0:
        axis.text(0.5, 0.5, f"H={mean_thickness:g} m\nnot yet available",
                  transform=axis.transAxes, ha="center", va="center",
                  color=COLORS["muted"])
        axis.set_axis_off()
        return None
    x_edges = logarithmic_cell_edges(wavelengths)
    y_edges = linear_cell_edges(amplitudes, lower_bound=0.0, upper_bound=1.0)
    resolved = categories == "resolved"
    _status_underlay(axis, x_edges, y_edges, categories)
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
    axis.set_ylim(0.0, y_edges[-1])
    axis.set_xticks((1 / 12, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 12))
    axis.set_xticklabels(
        (r"$1/12$", r"$1/8$", r"$1/4$", r"$1/2$", "1", "2", "4", "8", "12"),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_yticks((0.0, 0.25, 0.5, 0.75))
    axis.grid(False)
    if panel_title is not None:
        axis.set_title(panel_title, loc="left", pad=4.0)
    return image


def status_legend(table):
    """Return only categorical keys that occur in the loaded table."""
    categories = {status_category(value) for value in table[STATUS_COLUMN]}
    if np.any(
        table[STATUS_COLUMN].map(resolved_status)
        & ~np.isfinite(table[MINIMUM_COLUMN])
    ):
        categories.add("failed")
    handles = []
    labels = {
        "invalid": (COLORS["invalid"], "geometrically invalid"),
        "censored": (COLORS["censored"], "age-censored"),
        "failed": (COLORS["failed"], "unresolved numerical cell"),
        "missing": (COLORS["missing"], "missing"),
        "screened": (
            COLORS["screened"],
            r"surface signal below numerical leakage ($J$ not evaluated)",
        ),
    }
    for category in ("invalid", "censored", "failed", "screened", "missing"):
        if category in categories:
            color, label = labels[category]
            handles.append(Patch(facecolor=color, edgecolor="none", label=label))
    return handles


def save_main_figure(table, path):
    """Save the geometry plus five fixed-H heat maps."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9.1, 5.95),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.90, 1.0)},
    )
    draw_geometry_schematic(axes[0, 0])
    axes[0, 0].set_title("A   One-wavelength geometry", loc="left", pad=4.0)

    map_axes = (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2])
    images = []
    for letter, mean_thickness, axis in zip(
        "BCDEF", EXPECTED_THICKNESSES, map_axes
    ):
        delta = mean_thickness / 2500.0
        image = draw_heatmap(
            axis,
            table,
            mean_thickness,
            panel_title=(
                rf"{letter}   $H={mean_thickness:g}$ m; "
                rf"$\beta H/\eta={delta:.3f}$"
            ),
        )
        if image is not None:
            images.append(image)

    for index, axis in enumerate(map_axes):
        if axis.axison:
            if index in (2, 3, 4):
                axis.set_xlabel(r"Wavelength, $\lambda/H$")
            if index in (0, 2):
                axis.set_ylabel(r"Bed amplitude, $A/H$")
            if index not in (0, 2):
                axis.set_yticklabels([])

    if not images:
        raise RuntimeError("no fixed-H panel contains plottable data")
    colorbar = figure.colorbar(
        images[0],
        ax=list(map_axes),
        orientation="horizontal",
        fraction=0.055,
        pad=0.075,
        aspect=42,
        extend="min",
    )
    colorbar.set_ticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    colorbar.set_label(
        r"Minimum steady-characteristic $J=T_x$ "
        r"(central 80% of depositing flux)"
    )
    handles = status_legend(table)
    if handles:
        axes[0, 0].legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            frameon=False,
            ncol=2,
        )
    save_figure_pair(figure, path)


def resolved_table(table):
    mask = table[STATUS_COLUMN].map(resolved_status) & np.isfinite(
        table[MINIMUM_COLUMN]
    )
    return table.loc[mask].copy()


def select_sections(table, preferred_thickness=160.0):
    """Choose the lowest-J resolved row at the central production thickness."""
    resolved = resolved_table(table)
    if resolved.empty:
        raise RuntimeError("no resolved minimum J is available")
    thicknesses = np.sort(resolved[H_COLUMN].unique())
    selected_thickness = float(
        thicknesses[np.argmin(np.abs(thicknesses - preferred_thickness))]
    )
    section = resolved[
        np.isclose(resolved[H_COLUMN], selected_thickness)
        & (resolved[EPSILON_COLUMN] > 0.0)
    ]
    if section.empty:
        raise RuntimeError(
            f"no resolved non-flat row is available at H={selected_thickness:g} m"
        )
    index = section[MINIMUM_COLUMN].idxmin()
    row = resolved.loc[index]
    return (
        float(row[H_COLUMN]),
        float(row[EPSILON_COLUMN]),
        float(row[WAVELENGTH_COLUMN]),
        float(row[MINIMUM_COLUMN]),
    )


def nearest_values(values, targets):
    """Choose unique sampled values closest to ordered targets."""
    values = np.sort(np.asarray(values, dtype=float))
    chosen = []
    for target in targets:
        value = float(values[np.argmin(np.abs(values - target))])
        if value not in chosen:
            chosen.append(value)
    return chosen


def _wavelength_audit_values(validation_table, mean_thickness, amplitude):
    """Return distinct short/long extensions, never production-grid cells."""
    empty = pd.DataFrame(columns=(WAVELENGTH_COLUMN, MINIMUM_COLUMN))
    if validation_table is None or validation_table.empty:
        return empty
    required = {
        "scenario",
        METHOD_COLUMN,
        STATUS_COLUMN,
        H_COLUMN,
        EPSILON_COLUMN,
        WAVELENGTH_COLUMN,
        MINIMUM_COLUMN,
    }
    if not required.issubset(validation_table.columns):
        return empty
    audit = validation_table.copy()
    for column in (H_COLUMN, EPSILON_COLUMN, WAVELENGTH_COLUMN, MINIMUM_COLUMN):
        audit[column] = pd.to_numeric(audit[column], errors="coerce")
    audit = audit[
        audit["scenario"].isin(
            ("short-wave extension", "long-wave extension")
        )
        & audit[METHOD_COLUMN].eq(EXPECTED_METHOD_VERSION)
        & audit[STATUS_COLUMN].map(resolved_status)
        & np.isclose(audit[H_COLUMN], mean_thickness)
        & np.isclose(audit[EPSILON_COLUMN], amplitude)
        & np.isfinite(audit[MINIMUM_COLUMN])
    ]
    return audit.sort_values(WAVELENGTH_COLUMN)


def _thin_neck_audit_values(
    validation_table, mean_thickness, wavelength_ratio
):
    """Return the two primary A/H>0.85 extensions on one fixed slice."""
    empty = pd.DataFrame(columns=(EPSILON_COLUMN, MINIMUM_COLUMN))
    if validation_table is None or validation_table.empty:
        return empty
    required = {
        "scenario",
        METHOD_COLUMN,
        STATUS_COLUMN,
        H_COLUMN,
        EPSILON_COLUMN,
        WAVELENGTH_COLUMN,
        MINIMUM_COLUMN,
    }
    if not required.issubset(validation_table.columns):
        return empty
    audit = validation_table.copy()
    for column in (H_COLUMN, EPSILON_COLUMN, WAVELENGTH_COLUMN, MINIMUM_COLUMN):
        audit[column] = pd.to_numeric(audit[column], errors="coerce")
    audit = audit[
        audit["scenario"].isin(
            (
                "thin-neck extension A/H=0.90",
                "thin-neck extension A/H=0.95",
            )
        )
        & audit[METHOD_COLUMN].eq(EXPECTED_METHOD_VERSION)
        & audit[STATUS_COLUMN].map(resolved_status)
        & np.isclose(audit[H_COLUMN], mean_thickness)
        & np.isclose(audit[WAVELENGTH_COLUMN], wavelength_ratio)
        & np.isfinite(audit[MINIMUM_COLUMN])
    ]
    return audit.sort_values(EPSILON_COLUMN)


def _thin_neck_wavelength_audits(validation_table, mean_thickness):
    """Return the finest A/H=0.95 audit at each sampled wavelength."""
    if validation_table is None or validation_table.empty:
        return pd.DataFrame(columns=(WAVELENGTH_COLUMN, MINIMUM_COLUMN))
    required = {
        "scenario",
        STATUS_COLUMN,
        H_COLUMN,
        EPSILON_COLUMN,
        WAVELENGTH_COLUMN,
        MINIMUM_COLUMN,
    }
    if not required.issubset(validation_table.columns):
        return pd.DataFrame(columns=(WAVELENGTH_COLUMN, MINIMUM_COLUMN))
    audit = validation_table.copy()
    for column in (H_COLUMN, EPSILON_COLUMN, WAVELENGTH_COLUMN, MINIMUM_COLUMN):
        audit[column] = pd.to_numeric(audit[column], errors="coerce")
    audit = audit[
        audit["scenario"].astype(str).str.contains("A/H=0.95")
        & audit[STATUS_COLUMN].map(resolved_status)
        & np.isclose(audit[H_COLUMN], mean_thickness)
        & np.isclose(audit[EPSILON_COLUMN], 0.95)
        & np.isfinite(audit[MINIMUM_COLUMN])
    ].copy()
    if audit.empty:
        return audit
    if "mesh cells" in audit:
        audit = audit.sort_values("mesh cells")
    return (
        audit.drop_duplicates(WAVELENGTH_COLUMN, keep="last")
        .sort_values(WAVELENGTH_COLUMN)
    )


def _line_data(table, mean_thickness, amplitude=None, wavelength=None):
    subset = resolved_table(table)
    subset = subset[np.isclose(subset[H_COLUMN], mean_thickness)]
    if amplitude is not None:
        subset = subset[np.isclose(subset[EPSILON_COLUMN], amplitude)]
    if wavelength is not None:
        subset = subset[np.isclose(subset[WAVELENGTH_COLUMN], wavelength)]
    return subset.sort_values(
        WAVELENGTH_COLUMN if amplitude is not None else EPSILON_COLUMN
    )


def draw_wavelength_section(axis, table, mean_thickness, selected_amplitude,
                            selected_wavelength, validation_table=None):
    """Show short/intermediate/long wavelength behavior at fixed H."""
    available = np.sort(
        table.loc[np.isclose(table[H_COLUMN], mean_thickness), EPSILON_COLUMN]
        .dropna()
        .unique()
    )
    amplitudes = nearest_values(
        available[available > 0.0],
        (0.12, 0.35, 0.65, selected_amplitude),
    )
    amplitudes = sorted(amplitudes)
    color_values = plt.get_cmap("viridis")(np.linspace(0.15, 0.84, len(amplitudes)))
    amplitude_colors = {}
    for amplitude, color in zip(amplitudes, color_values):
        amplitude_colors[amplitude] = color
        subset = _line_data(table, mean_thickness, amplitude=amplitude)
        if subset.empty:
            continue
        is_selected = np.isclose(amplitude, selected_amplitude)
        axis.plot(
            subset[WAVELENGTH_COLUMN],
            subset[MINIMUM_COLUMN],
            marker="o",
            ms=3.4 if is_selected else 2.8,
            lw=1.8 if is_selected else 1.0,
            color=COLORS["ink"] if is_selected else color,
            zorder=4 if is_selected else 2,
            label=rf"$A/H={amplitude:g}$" + (" (selected)" if is_selected else ""),
        )
    selected = _line_data(
        table, mean_thickness, amplitude=selected_amplitude
    )
    raw_column = "raw minimum J to exit"
    if raw_column in selected and selected[raw_column].notna().any():
        raw = selected.loc[selected[raw_column].notna()]
        axis.plot(
            raw[WAVELENGTH_COLUMN],
            raw[raw_column],
            color=COLORS["ink"],
            lw=1.0,
            ls=(0, (4, 2)),
            marker="none",
            alpha=0.82,
            zorder=3,
            label="raw to exit (selected row)",
        )

    audit_values = _wavelength_audit_values(
        validation_table, mean_thickness, amplitude=0.65
    )
    if len(audit_values) and any(np.isclose(amplitudes, 0.65)):
        audit_color = amplitude_colors[
            min(amplitude_colors, key=lambda value: abs(value - 0.65))
        ]
        production_audit_row = _line_data(
            table, mean_thickness, amplitude=0.65
        )
        for _, audit in audit_values.iterrows():
            edge_index = (
                production_audit_row[WAVELENGTH_COLUMN] -
                audit[WAVELENGTH_COLUMN]
            ).abs().idxmin()
            edge = production_audit_row.loc[edge_index]
            axis.plot(
                (audit[WAVELENGTH_COLUMN], edge[WAVELENGTH_COLUMN]),
                (audit[MINIMUM_COLUMN], edge[MINIMUM_COLUMN]),
                color=audit_color,
                lw=0.8,
                ls=(0, (2, 2)),
                zorder=1,
            )
        axis.scatter(
            audit_values[WAVELENGTH_COLUMN],
            audit_values[MINIMUM_COLUMN],
            marker="D",
            s=26,
            facecolor="white",
            edgecolor=audit_color,
            linewidth=1.0,
            zorder=5,
            label=r"$A/H=0.65$ regime audits",
        )
    thin_neck_wavelengths = _thin_neck_wavelength_audits(
        validation_table, mean_thickness
    )
    if len(thin_neck_wavelengths):
        axis.plot(
            thin_neck_wavelengths[WAVELENGTH_COLUMN],
            thin_neck_wavelengths[MINIMUM_COLUMN],
            color=COLORS["negative_light"],
            lw=1.0,
            ls=(0, (3, 2)),
            marker="s",
            ms=3.8,
            mfc="white",
            label=r"$A/H=0.95$ thin-neck audits",
            zorder=5,
        )
    axis.axhline(0.0, color=COLORS["muted"], lw=0.7)
    axis.axhline(1.0, color=COLORS["grid"], lw=0.8, ls=(0, (2, 2)))
    axis.axvline(selected_wavelength, color=COLORS["grid"], lw=0.8,
                 ls=(0, (3, 2)), zorder=0)
    if not selected.empty:
        selected_minimum = selected.loc[selected[MINIMUM_COLUMN].idxmin()]
        axis.annotate(
            "intermediate-scale\nmaximum distortion",
            xy=(selected_minimum[WAVELENGTH_COLUMN],
                selected_minimum[MINIMUM_COLUMN]),
            xytext=(10, 14),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "->",
                "color": COLORS["muted"],
                "lw": 0.7,
            },
            ha="left",
            va="bottom",
            fontsize=7.2,
            color=COLORS["muted"],
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.0,
            },
        )
    axis.set_xscale("log")
    if len(audit_values):
        axis.set_xlim(0.07, 26.0)
        axis.set_xticks(
            (1 / 12, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 12, 24)
        )
        axis.set_xticklabels(
            (r"$1/12$", r"$1/8$", r"$1/4$", r"$1/2$", "1", "2", "4", "8", "12", "24")
        )
    else:
        axis.set_xticks((1 / 12, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 12))
        axis.set_xticklabels(
            (r"$1/12$", r"$1/8$", r"$1/4$", r"$1/2$", "1", "2", "4", "8", "12")
        )
    for label in axis.get_xticklabels():
        label.set_rotation(35)
        label.set_horizontalalignment("right")
    all_values = resolved_table(table)
    all_values = all_values[np.isclose(all_values[H_COLUMN], mean_thickness)]
    displayed_values = all_values[MINIMUM_COLUMN].to_numpy(float)
    if raw_column in selected:
        displayed_values = np.r_[
            displayed_values,
            selected[raw_column].dropna().to_numpy(float),
        ]
    if len(audit_values):
        displayed_values = np.r_[
            displayed_values,
            audit_values[MINIMUM_COLUMN].to_numpy(float),
        ]
    if len(thin_neck_wavelengths):
        displayed_values = np.r_[
            displayed_values,
            thin_neck_wavelengths[MINIMUM_COLUMN].to_numpy(float),
        ]
    minimum = float(np.nanmin(displayed_values))
    axis.set_ylim(min(-0.08, minimum - 0.12), 1.04)
    axis.set(
        xlabel=r"Wavelength, $\lambda/H$",
        ylabel=r"Minimum steady $J$",
    )
    axis.text(0.02, 0.04, "toward short-wave\nscreening",
              transform=axis.transAxes, ha="left", va="bottom",
              fontsize=7.5, color=COLORS["muted"])
    axis.text(0.98, 0.04, "toward long-wave,\ngentle-bed flow",
              transform=axis.transAxes, ha="right", va="bottom",
              fontsize=7.5, color=COLORS["muted"])
    axis.legend(frameon=False, ncol=2, loc="lower center",
                bbox_to_anchor=(0.5, 0.12), handlelength=1.5,
                columnspacing=0.7, fontsize=7.0)


def draw_amplitude_section(axis, table, selected_thickness,
                           selected_wavelength, validation_table=None):
    """Show the regular flat-bed and singular thin-neck directions."""
    thicknesses = np.sort(table[H_COLUMN].dropna().unique())
    colors = plt.get_cmap("plasma")(np.linspace(0.10, 0.82, len(thicknesses)))
    for mean_thickness, color in zip(thicknesses, colors):
        subset = _line_data(
            table, mean_thickness, wavelength=selected_wavelength
        )
        if subset.empty:
            continue
        is_selected = np.isclose(mean_thickness, selected_thickness)
        axis.plot(
            subset[EPSILON_COLUMN],
            subset[MINIMUM_COLUMN],
            marker="o",
            ms=3.4 if is_selected else 2.5,
            lw=1.8 if is_selected else 0.9,
            color=COLORS["ink"] if is_selected else color,
            alpha=1.0 if is_selected else 0.72,
            label=rf"$H={mean_thickness:g}$ m" + (" (selected)" if is_selected else ""),
        )
    axis.axhline(0.0, color=COLORS["muted"], lw=0.7)
    axis.axhline(1.0, color=COLORS["grid"], lw=0.8, ls=(0, (2, 2)))
    axis.axvspan(EXPECTED_AMPLITUDE_RATIOS[-1], 1.0,
                 color=COLORS["missing"], zorder=-2)
    thin_neck_audits = _thin_neck_audit_values(
        validation_table, selected_thickness, selected_wavelength
    )
    if len(thin_neck_audits):
        selected_production = _line_data(
            table, selected_thickness, wavelength=selected_wavelength
        )
        selected_production = selected_production[
            selected_production[EPSILON_COLUMN]
            == selected_production[EPSILON_COLUMN].max()
        ]
        extension = pd.concat(
            (
                selected_production[[EPSILON_COLUMN, MINIMUM_COLUMN]],
                thin_neck_audits[[EPSILON_COLUMN, MINIMUM_COLUMN]],
            ),
            ignore_index=True,
        ).sort_values(EPSILON_COLUMN)
        axis.plot(
            extension[EPSILON_COLUMN],
            extension[MINIMUM_COLUMN],
            color=COLORS["ink"],
            lw=0.8,
            ls=(0, (2, 2)),
            zorder=2,
        )
        axis.scatter(
            thin_neck_audits[EPSILON_COLUMN],
            thin_neck_audits[MINIMUM_COLUMN],
            marker="D",
            s=26,
            facecolor="white",
            edgecolor=COLORS["ink"],
            linewidth=1.0,
            zorder=5,
            label="thin-neck audits (outside map)",
        )
    axis.set_xlim(0.0, 1.0)
    values = resolved_table(table)
    values = values[np.isclose(values[WAVELENGTH_COLUMN], selected_wavelength)]
    displayed_values = values[MINIMUM_COLUMN].to_numpy(float)
    if len(thin_neck_audits):
        displayed_values = np.r_[
            displayed_values,
            thin_neck_audits[MINIMUM_COLUMN].to_numpy(float),
        ]
    axis.set_ylim(min(-0.08, float(np.nanmin(displayed_values)) - 0.12), 1.04)
    axis.set(
        xlabel=r"Bed amplitude, $A/H$",
        ylabel=r"Minimum steady $J$",
    )
    axis.text(
        0.02,
        0.04,
        r"flat bed: $J\rightarrow 1$;"
        "\n"
        r"surface exchange $\rightarrow 0$",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    axis.text(
        0.98,
        0.04,
        "thin-neck direction:"
        "\n"
        r"$(H-A)/H\rightarrow 0$",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    axis.legend(frameon=False, ncol=2, loc="lower center",
                bbox_to_anchor=(0.53, 0.14), handlelength=1.5,
                columnspacing=0.8)


def draw_flow_diagnostics(axis, table, mean_thickness, amplitude):
    """Relate the selected J row to counterflow and surface exchange."""
    subset = table[
        np.isclose(table[H_COLUMN], mean_thickness)
        & np.isclose(table[EPSILON_COLUMN], amplitude)
        & (table[STATUS_COLUMN] == "success")
    ].sort_values(WAVELENGTH_COLUMN)
    columns = {
        "minimum sampled u (m/yr)",
        "minimum path u (m/yr)",
        "maximum sampled u (m/yr)",
        "sampled gross surface mass exchange (m2/yr)",
        "wavelength (m)",
    }
    if subset.empty or not columns.issubset(subset.columns):
        axis.text(0.5, 0.5, "flow diagnostics unavailable",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
        return
    u_min = subset["minimum sampled u (m/yr)"].to_numpy(float)
    path_u_min = subset["minimum path u (m/yr)"].to_numpy(float)
    u_max = subset["maximum sampled u (m/yr)"].to_numpy(float)
    speed = np.maximum(np.abs(u_min), np.abs(u_max))
    velocity_ratio = np.divide(
        u_min, speed, out=np.full_like(u_min, np.nan), where=speed > 0.0
    )
    path_velocity_ratio = np.divide(
        path_u_min,
        speed,
        out=np.full_like(path_u_min, np.nan),
        where=speed > 0.0,
    )
    gross = subset["sampled gross surface mass exchange (m2/yr)"].to_numpy(float)
    wavelength = subset["wavelength (m)"].to_numpy(float)
    exchange_ratio = np.divide(
        gross, wavelength * speed,
        out=np.full_like(gross, np.nan), where=(wavelength * speed) > 0.0,
    )
    ratio = subset[WAVELENGTH_COLUMN].to_numpy(float)

    axis.axhspan(-1.0, 0.0, color=COLORS["negative_light"], alpha=0.08)
    axis.axhline(0.0, color=COLORS["muted"], lw=0.7)
    domain_velocity_line = axis.plot(
        ratio,
        velocity_ratio,
        marker="o",
        ms=3.2,
        mfc="white",
        color=COLORS["muted"],
        label=r"domain $u_{\min}/\max|u|$",
    )[0]
    path_velocity_line = axis.plot(
        ratio,
        path_velocity_ratio,
        marker="o",
        ms=3.5,
        color=COLORS["ink"],
        label=r"path $u_{\min}/\max|u|$",
    )[0]
    axis.set_xscale("log")
    axis.set_xticks((1 / 12, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 12))
    axis.set_xticklabels(
        (r"$1/12$", r"$1/8$", r"$1/4$", r"$1/2$", "1", "2", "4", "8", "12"),
        rotation=35,
        ha="right",
    )
    minimum_velocity_ratio = np.nanmin(
        np.r_[velocity_ratio, path_velocity_ratio]
    )
    axis.set_ylim(min(-0.15, minimum_velocity_ratio - 0.08), 1.02)
    axis.set(
        xlabel=r"Wavelength, $\lambda/H$",
        ylabel=r"Minimum horizontal velocity / $\max|u|$",
    )
    axis.text(
        0.02,
        0.05,
        r"counterflow where $u_{\min}<0$",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=COLORS["muted"],
    )

    exchange_axis = axis.twinx()
    exchange_line = exchange_axis.plot(
        ratio,
        exchange_ratio,
        marker="s",
        ms=3.0,
        color=COLORS["exchange"],
        label=r"$\int|a_s|\,dx/(\lambda\max|u|)$",
    )[0]
    exchange_axis.set_ylabel(
        r"Normalized surface mass exchange",
        color=COLORS["exchange"],
    )
    exchange_axis.tick_params(axis="y", colors=COLORS["exchange"])
    exchange_axis.spines["right"].set_color(COLORS["exchange"])
    exchange_axis.set_ylim(bottom=0.0)
    axis.legend(
        handles=(path_velocity_line, domain_velocity_line, exchange_line),
        loc="upper center",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        ncol=1,
        handlelength=1.6,
    )


def save_limits_figure(table, path, validation_table=None):
    """Save central-H map sections and dynamical/asymptotic diagnostics."""
    mean_thickness, amplitude, wavelength, minimum = select_sections(table)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 6.25),
        constrained_layout=True,
    )
    map_axis, wavelength_axis, amplitude_axis, dynamics_axis = axes.flat

    image = draw_heatmap(
        map_axis,
        table,
        mean_thickness,
        panel_title=(
            rf"A   Selected map, $H={mean_thickness:g}$ m "
            rf"($\beta H/\eta={mean_thickness / 2500.0:.3f}$)"
        ),
    )
    map_axis.set_xlabel(r"Wavelength, $\lambda/H$")
    map_axis.set_ylabel(r"Bed amplitude, $A/H$")
    map_axis.axhline(amplitude, color=COLORS["ink"], lw=0.9,
                    ls=(0, (4, 2)))
    map_axis.axvline(wavelength, color=COLORS["ink"], lw=0.9,
                    ls=(0, (4, 2)))
    map_axis.plot(wavelength, amplitude, marker="o", ms=5.0,
                  mfc="none", mec=COLORS["ink"], mew=1.0, zorder=5)
    map_axis.text(
        0.98,
        0.03,
        rf"selected-row minimum $J={minimum:.3f}$",
        transform=map_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=COLORS["ink"],
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 1.5,
        },
    )
    selected_status_handles = status_legend(
        table[np.isclose(table[H_COLUMN], mean_thickness)]
    )
    if selected_status_handles:
        for handle in selected_status_handles:
            if "surface signal" in handle.get_label():
                handle.set_label("dots: surface signal below leakage")
        map_axis.legend(
            handles=selected_status_handles,
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.82,
        )
    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=map_axis,
            orientation="horizontal",
            fraction=0.07,
            pad=0.13,
            aspect=32,
            extend="min",
        )
        colorbar.set_ticks((-1.0, -0.5, 0.0, 0.5, 1.0))
        colorbar.set_label(r"Minimum steady $J$")

    draw_wavelength_section(
        wavelength_axis,
        table,
        mean_thickness,
        amplitude,
        wavelength,
        validation_table=validation_table,
    )
    wavelength_axis.set_title(
        rf"B   Wavelength sections at $H={mean_thickness:g}$ m",
        loc="left",
    )
    draw_amplitude_section(
        amplitude_axis,
        table,
        mean_thickness,
        wavelength,
        validation_table=validation_table,
    )
    amplitude_axis.set_title(
        rf"C   Amplitude section at $\lambda/H={wavelength:g}$",
        loc="left",
    )
    draw_flow_diagnostics(dynamics_axis, table, mean_thickness, amplitude)
    dynamics_axis.set_title(
        rf"D   Flow response at $H={mean_thickness:g}$ m, "
        rf"$A/H={amplitude:g}$",
        loc="left",
    )
    save_figure_pair(figure, path)


def save_figure_pair(figure, png_path):
    """Write a high-resolution PNG and an editable-font PDF."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_sinusoidal_parameter_map.csv",
    )
    parser.add_argument(
        "--main-figure",
        type=Path,
        default=repository
        / "figures"
        / "steady_characteristic_sinusoidal_parameter_map.png",
    )
    parser.add_argument(
        "--limits-figure",
        type=Path,
        default=repository
        / "figures"
        / "steady_characteristic_sinusoidal_limits.png",
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_sinusoidal_validation.csv",
        help=(
            "Optional regime-audit table; if present, lambda/H=0.5 and 24 "
            "are plotted as distinct open audit markers."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Plot available cells while a restartable production sweep is running.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    configure_matplotlib()
    table = load_table(args.csv)
    problems = check_completeness(table)
    if problems and not args.allow_incomplete:
        raise RuntimeError(
            "; ".join(problems)
            + ". Rerun after the sweep finishes, or pass --allow-incomplete."
        )
    save_main_figure(table, args.main_figure)
    validation_table = None
    if args.validation_csv.exists():
        validation_table = pd.read_csv(args.validation_csv)
        if METHOD_COLUMN not in validation_table:
            raise ValueError(
                f"{args.validation_csv} lacks {METHOD_COLUMN!r}"
            )
        if not bool(
            validation_table[METHOD_COLUMN].eq(EXPECTED_METHOD_VERSION).all()
        ):
            raise ValueError(
                f"{args.validation_csv} contains a stale diagnostic method"
            )
    save_limits_figure(
        table,
        args.limits_figure,
        validation_table=validation_table,
    )
    print(f"Wrote {args.main_figure}")
    print(f"Wrote {args.main_figure.with_suffix('.pdf')}")
    print(f"Wrote {args.limits_figure}")
    print(f"Wrote {args.limits_figure.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
