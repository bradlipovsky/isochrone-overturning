"""Plot the steady-characteristic two-harmonic bed sweep.

The Stokes/characteristic sweep is intentionally separate from this file.
This script only reads its CSV table and can therefore be rerun with an
ordinary scientific-Python environment (Firedrake is not imported).

The modeled bed is

    b(x) = -H + A f(theta),
    f(theta) = [cos(theta) + q cos(2 theta + phi)] / max(raw),
    theta = 2 pi x / lambda.

Consequently ``H`` is the mean thickness and ``H-A`` is the minimum ice
thickness for every ``q`` and ``phi``.  The upper two rows show the geometry
and five amplitude--wavelength maps for the selected shape.  The lower row
shows a wavelength section through those maps, the separate shape scan's
resolved ``q``--``phi`` dependence, and the exact characteristic mechanism
that controls the same minimum-J pattern.

No ``J=0`` contour is drawn.  Missing sweep cells remain visibly gray rather
than being interpolated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


H_COLUMN = "mean ice thickness (m)"
EPSILON_COLUMN = "amplitude / mean thickness"
WAVELENGTH_COLUMN = "wavelength / mean thickness"
Q_COLUMN = "second / first harmonic amplitude"
PHI_COLUMN = "relative phase (rad)"
MINIMUM_COLUMN = "minimum J"
STATUS_COLUMN = "status"
RELIEF_COLUMN = "normalized relief"
SCREENING_COLUMN = "screening chi"
MECHANISM_COLUMN = "mechanism C at minimum J"
VELOCITY_RATIO_COLUMN = "u / source u at minimum J"


COLUMN_ALIASES = {
    H_COLUMN: (
        H_COLUMN,
        "H (m)",
        "mean thickness (m)",
        "mean_thickness_m",
    ),
    EPSILON_COLUMN: (
        EPSILON_COLUMN,
        "A / H",
        "A/H",
        "amplitude ratio",
        "amplitude_ratio",
    ),
    WAVELENGTH_COLUMN: (
        WAVELENGTH_COLUMN,
        "lambda / H",
        "lambda/H",
        "wavelength ratio",
        "wavelength_ratio",
    ),
    Q_COLUMN: (
        Q_COLUMN,
        "second harmonic / first harmonic",
        "second harmonic amplitude ratio",
        "A2 / A1",
        "A2/A1",
        "q",
    ),
    PHI_COLUMN: (
        PHI_COLUMN,
        "harmonic phase (rad)",
        "relative harmonic phase (rad)",
        "phi (rad)",
        "phi",
    ),
    MINIMUM_COLUMN: (
        MINIMUM_COLUMN,
        "minimum steady J",
        "min J",
        "min_J",
    ),
    STATUS_COLUMN: (STATUS_COLUMN, "solver status", "case status"),
    RELIEF_COLUMN: (
        RELIEF_COLUMN,
        "bed relief / mean thickness",
        "relief / mean thickness",
        "relief / H",
        "relief/H",
    ),
    SCREENING_COLUMN: (
        SCREENING_COLUMN,
        "surface screening chi",
        "surface-screening chi",
        "screening factor chi",
        "chi",
    ),
    MECHANISM_COLUMN: (
        MECHANISM_COLUMN,
        "mechanism C",
        "characteristic mechanism C",
        "mechanism coefficient C",
        "C",
    ),
    VELOCITY_RATIO_COLUMN: (
        VELOCITY_RATIO_COLUMN,
        "minimum horizontal velocity / flat-bed velocity",
        "minimum u / flat-bed u",
        "minimum u / u0",
        "min u/u0",
        "u_min / u0",
        "minimum path u / u0",
    ),
}

DIMENSIONAL_RELIEF_ALIASES = (
    "bed peak-to-trough relief (m)",
    "peak-to-trough relief (m)",
    "bed relief (m)",
)
DIMENSIONAL_MINIMUM_U_ALIASES = (
    "minimum path u (m/yr)",
    "minimum sampled u (m/yr)",
)


COLORS = {
    "ink": "#252525",
    "muted": "#666666",
    "grid": "#d6d6d6",
    "missing": "#e3e3e3",
    "screened": "#c9dce8",
    "ice": "#dceef5",
    "bed": "#b9a590",
    "surface": "#0b4f7a",
    "negative": "#8e1537",
    "envelope": "#a63603",
    "points": "#6f6f6f",
    "counterflow": "#f3ddd7",
}

J_CMAP = LinearSegmentedColormap.from_list(
    "negative_red_zero_white_positive_blue",
    ("#67001f", "#d6604d", "#ffffff", "#4393c3", "#053061"),
    N=257,
)
J_CMAP.set_bad(COLORS["missing"])


def configure_matplotlib():
    """Use compact, editable-vector defaults consistent with nearby figures."""
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _find_alias(columns, aliases):
    """Return one matching input column, allowing case/space differences."""
    direct = {str(column): column for column in columns}
    folded = {
        " ".join(str(column).strip().lower().split()): column
        for column in columns
    }
    for alias in aliases:
        if alias in direct:
            return direct[alias]
        key = " ".join(alias.strip().lower().split())
        if key in folded:
            return folded[key]
    return None


def load_table(path):
    """Load the sweep, normalize its schema, and retain the latest restart."""
    table = pd.read_csv(path)
    rename = {}
    required = (
        H_COLUMN,
        EPSILON_COLUMN,
        WAVELENGTH_COLUMN,
        Q_COLUMN,
        PHI_COLUMN,
        MINIMUM_COLUMN,
        STATUS_COLUMN,
    )
    for canonical, aliases in COLUMN_ALIASES.items():
        source = _find_alias(table.columns, aliases)
        if source is not None:
            rename[source] = canonical
    table = table.rename(columns=rename)
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")

    numeric = (
        H_COLUMN,
        EPSILON_COLUMN,
        WAVELENGTH_COLUMN,
        Q_COLUMN,
        PHI_COLUMN,
        MINIMUM_COLUMN,
        RELIEF_COLUMN,
        SCREENING_COLUMN,
        MECHANISM_COLUMN,
        VELOCITY_RATIO_COLUMN,
    )
    for column in numeric:
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    # Accept a dimensional relief column without weakening the public schema.
    if RELIEF_COLUMN not in table:
        source = _find_alias(table.columns, DIMENSIONAL_RELIEF_ALIASES)
        if source is not None:
            table[RELIEF_COLUMN] = (
                pd.to_numeric(table[source], errors="coerce") / table[H_COLUMN]
            )

    # Phase is periodic.  Rounding values at 2 pi to zero also removes a
    # common duplicate endpoint from restartable sweeps.
    table[PHI_COLUMN] = np.mod(table[PHI_COLUMN], 2.0 * np.pi)
    table.loc[
        np.isclose(table[PHI_COLUMN], 2.0 * np.pi, atol=1.0e-10), PHI_COLUMN
    ] = 0.0
    table = (
        table.sort_index()
        .drop_duplicates(
            subset=(
                H_COLUMN,
                EPSILON_COLUMN,
                WAVELENGTH_COLUMN,
                Q_COLUMN,
                PHI_COLUMN,
            ),
            keep="last",
        )
        .sort_values(
            [
                H_COLUMN,
                EPSILON_COLUMN,
                WAVELENGTH_COLUMN,
                Q_COLUMN,
                PHI_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )
    return table


def resolved_rows(table):
    """Keep successful numerical cases and any explicit analytic limits."""
    status = table[STATUS_COLUMN].fillna("").astype(str).str.lower()
    good_status = status.str.startswith("success") | status.str.startswith(
        "analytic"
    )
    finite = np.isfinite(table[MINIMUM_COLUMN])
    return table.loc[good_status & finite].copy()


def nearest_thickness(table, requested):
    values = np.sort(table[H_COLUMN].dropna().unique())
    if len(values) == 0:
        raise ValueError("the sweep has no finite mean thickness")
    return float(values[np.argmin(np.abs(values - requested))])


def linear_cell_edges(values, lower_bound=None, upper_bound=None):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        half = max(0.05, 0.10 * max(abs(values[0]), 1.0))
        edges = np.array((values[0] - half, values[0] + half))
    else:
        middle = 0.5 * (values[:-1] + values[1:])
        edges = np.r_[
            2.0 * values[0] - middle[0],
            middle,
            2.0 * values[-1] - middle[-1],
        ]
    if lower_bound is not None:
        edges[0] = max(float(lower_bound), edges[0])
    if upper_bound is not None:
        edges[-1] = min(float(upper_bound), edges[-1])
    return edges


def logarithmic_cell_edges(values):
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("logarithmic coordinates must be positive")
    if len(values) == 1:
        return values[0] * np.array((1.0 / np.sqrt(1.5), np.sqrt(1.5)))
    middle = np.sqrt(values[:-1] * values[1:])
    return np.r_[
        values[0] ** 2 / middle[0],
        middle,
        values[-1] ** 2 / middle[-1],
    ]


def phase_cell_edges(phases):
    """Return periodic edges around phase centers in ``[0, 2 pi)``."""
    phases = np.sort(np.asarray(phases, dtype=float))
    if len(phases) == 1:
        return np.array((-np.pi, np.pi)) + phases[0]
    edges = np.empty(len(phases) + 1)
    wrap_gap = phases[0] + 2.0 * np.pi - phases[-1]
    edges[0] = phases[0] - 0.5 * wrap_gap
    edges[1:-1] = 0.5 * (phases[:-1] + phases[1:])
    edges[-1] = edges[0] + 2.0 * np.pi
    return edges


def pivot_grid(frame, row, column, value, aggregate="min"):
    rows = np.sort(frame[row].dropna().unique())
    columns = np.sort(frame[column].dropna().unique())
    grid = (
        frame.pivot_table(
            index=row,
            columns=column,
            values=value,
            aggfunc=aggregate,
        )
        .reindex(index=rows, columns=columns)
        .to_numpy(dtype=float)
    )
    return rows, columns, grid


def choose_j_norm(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("no resolved minimum J is available")
    lower = min(-1.0, float(finite.min()))
    upper = max(1.0, float(finite.max()))
    return TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)


def normalized_shape(x_over_wavelength, q, phi):
    """Evaluate the exact normalized two-harmonic shape used by the sweep."""
    theta = 2.0 * np.pi * np.asarray(x_over_wavelength)
    raw = np.cos(theta) + q * np.cos(2.0 * theta + phi)
    dense_theta = np.linspace(0.0, 2.0 * np.pi, 20001, endpoint=False)
    dense_raw = np.cos(dense_theta) + q * np.cos(
        2.0 * dense_theta + phi
    )
    maximum = float(dense_raw.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("two-harmonic shape has no finite positive maximum")
    return raw / maximum


def draw_geometry(axis, row):
    epsilon = float(row[EPSILON_COLUMN])
    q = float(row[Q_COLUMN])
    phi = float(row[PHI_COLUMN])
    x = np.linspace(0.0, 1.0, 1601)
    shape = normalized_shape(x, q, phi)
    bed = -1.0 + epsilon * shape
    peak = int(np.argmax(bed))
    relief = epsilon * (float(shape.max()) - float(shape.min()))
    y_bottom = min(-1.55, float(bed.min()) - 0.32)

    axis.fill_between(x, bed, 0.0, color=COLORS["ice"], zorder=1)
    axis.fill_between(x, y_bottom, bed, color=COLORS["bed"], zorder=0)
    axis.plot(x, bed, color=COLORS["ink"], lw=1.55, zorder=3)
    axis.plot((0.0, 1.0), (0.0, 0.0), color=COLORS["surface"], lw=1.4)
    axis.axhline(-1.0, color=COLORS["muted"], lw=0.8, ls=(0, (2, 2)))

    arrow = dict(
        arrowstyle="<->", color=COLORS["ink"], lw=0.85, shrinkA=0, shrinkB=0
    )
    axis.annotate("", xy=(0.0, 0.16), xytext=(1.0, 0.16), arrowprops=arrow)
    axis.text(0.5, 0.19, r"one period, $\lambda$", ha="center", va="bottom")
    axis.annotate("", xy=(0.18, 0.0), xytext=(0.18, -1.0), arrowprops=arrow)
    axis.text(0.205, -0.50, r"$H$", ha="left", va="center")
    axis.annotate(
        "",
        xy=(x[peak], bed[peak]),
        xytext=(x[peak], -1.0),
        arrowprops=arrow,
    )
    axis.text(x[peak] + 0.025, -1.0 + 0.5 * epsilon, r"$A$", va="center")
    axis.text(
        0.5,
        0.025,
        "flat surface",
        color=COLORS["surface"],
        ha="center",
        va="bottom",
    )
    axis.text(0.98, -0.96, r"mean bed, $-H$", color=COLORS["muted"],
              ha="right", va="bottom")
    axis.text(
        0.5,
        y_bottom + 0.025,
        r"$f=[\cos\theta+q\cos(2\theta+\phi)]/\max(\mathrm{raw})$"
        "\n"
        rf"$q={q:.3g}$; $\phi={phi / np.pi:.3g}\pi$; relief$/H={relief:.3g}$",
        ha="center",
        va="bottom",
        fontsize=7.4,
    )
    axis.set(xlim=(-0.03, 1.03), ylim=(y_bottom, 0.31))
    axis.set_aspect(0.48)
    axis.axis("off")


def _wavelength_ticks(values):
    candidates = np.array((1 / 12, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 12))
    labels = (r"$1/12$", r"$1/8$", r"$1/4$", r"$1/2$", "1", "2", "4", "8", "12")
    lower, upper = float(np.min(values)), float(np.max(values))
    keep = (candidates >= lower * 0.999) & (candidates <= upper * 1.001)
    return candidates[keep], np.asarray(labels, object)[keep]


def draw_parameter_map(axis, frame, norm):
    """Draw the selected-shape amplitude--wavelength parameter map."""
    parameter_map = (
        frame.groupby([EPSILON_COLUMN, WAVELENGTH_COLUMN], as_index=False)[
            MINIMUM_COLUMN
        ]
        .min()
    )
    amplitudes, wavelengths, values = pivot_grid(
        parameter_map,
        EPSILON_COLUMN,
        WAVELENGTH_COLUMN,
        MINIMUM_COLUMN,
    )
    x_edges = logarithmic_cell_edges(wavelengths)
    y_edges = linear_cell_edges(amplitudes, lower_bound=0.0, upper_bound=1.0)
    screened_frame = frame.copy()
    screened_frame["screened"] = (
        screened_frame[STATUS_COLUMN]
        .fillna("")
        .astype(str)
        .str.contains("screened below numerical surface signal")
        .astype(float)
    )
    _, _, screened = pivot_grid(
        screened_frame,
        EPSILON_COLUMN,
        WAVELENGTH_COLUMN,
        "screened",
        aggregate="max",
    )
    axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_where(screened < 0.5, screened),
        cmap=ListedColormap((COLORS["screened"],)),
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        rasterized=True,
        zorder=0,
    )
    image = axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(values),
        cmap=J_CMAP,
        norm=norm,
        shading="flat",
        rasterized=True,
        zorder=1,
    )
    screen_rows, screen_columns = np.nonzero(screened >= 0.5)
    if len(screen_rows):
        axis.scatter(
            np.sqrt(
                x_edges[screen_columns] * x_edges[screen_columns + 1]
            ),
            0.5 * (y_edges[screen_rows] + y_edges[screen_rows + 1]),
            marker=".",
            s=7,
            color=COLORS["muted"],
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.set_ylim(y_edges[0], y_edges[-1])
    ticks, labels = _wavelength_ticks(wavelengths)
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
    axis.set_xlabel(r"Fundamental wavelength, $\lambda/H$")
    axis.set_ylabel(r"Bed amplitude, $A/H$")
    axis.grid(False)

    resolved = resolved_rows(frame)
    worst = resolved.loc[resolved[MINIMUM_COLUMN].idxmin()]
    axis.scatter(
        [worst[WAVELENGTH_COLUMN]],
        [worst[EPSILON_COLUMN]],
        marker="o",
        s=24,
        facecolor="none",
        edgecolor=COLORS["ink"],
        lw=0.9,
        zorder=4,
    )
    return image, worst


def status_legend(table):
    """Explain unresolved cells once, outside the heat-map axes."""
    status = table[STATUS_COLUMN].fillna("").astype(str)
    handles = [
        Line2D(
            (),
            (),
            marker="o",
            ms=4.5,
            mfc="none",
            mec=COLORS["ink"],
            mew=0.8,
            ls="none",
            label=r"open circle: lowest resolved $J$",
        )
    ]
    if status.str.contains("screened below numerical surface signal").any():
        handles.append(
            Line2D(
                (),
                (),
                marker=".",
                ms=6.0,
                color=COLORS["muted"],
                ls="none",
                label="dots: numerically unresolved surface signal",
            )
        )
    if status.str.contains("censored", case=False).any():
        handles.append(
            Patch(
                facecolor=COLORS["missing"],
                edgecolor="none",
                label="age-censored numerical cell",
            )
        )
    return handles


def _fill_phase_invariant_q_zero(section, phases):
    """Replicate the q=0 value because phi is then a coordinate gauge."""
    zero = section[np.isclose(section[Q_COLUMN], 0.0)]
    if zero.empty:
        return section
    template = zero.loc[zero[MINIMUM_COLUMN].idxmin()].copy()
    existing = zero[PHI_COLUMN].to_numpy(dtype=float)
    rows = []
    for phase in phases:
        if not np.any(np.isclose(existing, phase, atol=1.0e-10)):
            row = template.copy()
            row[PHI_COLUMN] = phase
            rows.append(row)
    if rows:
        return pd.concat((section, pd.DataFrame(rows)), ignore_index=True)
    return section


def draw_shape_map(axis, frame, worst, norm):
    section = frame[
        np.isclose(frame[EPSILON_COLUMN], worst[EPSILON_COLUMN])
        & np.isclose(frame[WAVELENGTH_COLUMN], worst[WAVELENGTH_COLUMN])
    ].copy()
    nonzero = section.loc[~np.isclose(section[Q_COLUMN], 0.0)]
    phase_source = nonzero if not nonzero.empty else section
    phases = np.sort(phase_source[PHI_COLUMN].dropna().unique())
    section = _fill_phase_invariant_q_zero(section, phases)
    q_values, phases, values = pivot_grid(
        section, Q_COLUMN, PHI_COLUMN, MINIMUM_COLUMN
    )
    x_edges = phase_cell_edges(phases)
    y_edges = linear_cell_edges(q_values, lower_bound=0.0)
    image = axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(values),
        cmap=J_CMAP,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    axis.scatter(
        [worst[PHI_COLUMN]],
        [worst[Q_COLUMN]],
        marker="o",
        s=24,
        facecolor="none",
        edgecolor=COLORS["ink"],
        lw=0.9,
        zorder=4,
    )
    phase_ticks = np.array((0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi))
    keep = (phase_ticks >= x_edges[0]) & (phase_ticks <= x_edges[-1])
    axis.set_xticks(phase_ticks[keep])
    axis.set_xticklabels(np.asarray(("0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"))[keep])
    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.set_ylim(y_edges[0], y_edges[-1])
    axis.set_xlabel(r"Relative phase, $\phi$")
    axis.set_ylabel(r"Harmonic ratio, $q=A_2/A_1$")
    axis.grid(False)
    return image


def nearest_values(values, targets):
    """Choose distinct sampled values nearest a short ordered target list."""
    values = np.sort(np.asarray(values, dtype=float))
    chosen = []
    for target in targets:
        value = float(values[np.argmin(np.abs(values - target))])
        if value not in chosen:
            chosen.append(value)
    return chosen


def draw_wavelength_section(
    axis,
    frame,
    selected_amplitude=0.85,
    validation_table=None,
):
    """Expose the short-, intermediate-, and long-wave limits."""
    amplitudes = nearest_values(
        frame.loc[frame[EPSILON_COLUMN] > 0.0, EPSILON_COLUMN].unique(),
        (0.25, 0.65, selected_amplitude, 0.95),
    )
    colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.86, len(amplitudes))
    )
    amplitude_colors = dict(zip(amplitudes, colors))
    screened_label_used = False
    selected_rows = None
    for amplitude, color in zip(amplitudes, colors):
        rows = frame[np.isclose(frame[EPSILON_COLUMN], amplitude)].copy()
        rows = rows.sort_values(WAVELENGTH_COLUMN)
        status = rows[STATUS_COLUMN].fillna("").astype(str).str.lower()
        resolved = (
            status.str.startswith("success")
            | status.str.startswith("analytic")
        ) & np.isfinite(rows[MINIMUM_COLUMN])
        y = rows[MINIMUM_COLUMN].where(resolved, np.nan)
        is_selected = np.isclose(amplitude, selected_amplitude)
        axis.plot(
            rows[WAVELENGTH_COLUMN],
            y,
            color=COLORS["ink"] if is_selected else color,
            marker="o",
            ms=3.6 if is_selected else 2.8,
            lw=1.8 if is_selected else 1.0,
            label=rf"$A/H={amplitude:g}$" + (
                " (quick scan)" if is_selected else (
                    " (thin neck)" if np.isclose(amplitude, 0.95) else ""
                )
            ),
            zorder=4 if is_selected else 2,
        )
        screened = status.str.contains(
            "screened below numerical surface signal"
        )
        if screened.any():
            axis.scatter(
                rows.loc[screened, WAVELENGTH_COLUMN],
                np.full(int(screened.sum()), 0.995),
                marker="v",
                s=23,
                facecolor="white",
                edgecolor=COLORS["ink"] if is_selected else color,
                linewidth=0.8,
                zorder=5,
                label=(
                    "screened; short-wave limit inferred"
                    if not screened_label_used
                    else None
                ),
            )
            screened_label_used = True
        if is_selected:
            selected_rows = rows.loc[resolved]

    if validation_table is not None and not validation_table.empty:
        validation = validation_table.copy()
        for column in (
            H_COLUMN,
            EPSILON_COLUMN,
            WAVELENGTH_COLUMN,
            Q_COLUMN,
            PHI_COLUMN,
            MINIMUM_COLUMN,
        ):
            if column in validation:
                validation[column] = pd.to_numeric(
                    validation[column], errors="coerce"
                )
        required = {
            H_COLUMN,
            EPSILON_COLUMN,
            WAVELENGTH_COLUMN,
            Q_COLUMN,
            PHI_COLUMN,
            MINIMUM_COLUMN,
            STATUS_COLUMN,
            "validation resolution",
        }
        if required.issubset(validation.columns):
            mean_thickness = float(frame[H_COLUMN].dropna().iloc[0])
            selected_q = float(frame[Q_COLUMN].dropna().iloc[0])
            selected_phi = float(frame[PHI_COLUMN].dropna().iloc[0])
            status = validation[STATUS_COLUMN].fillna("").astype(str).str.lower()
            resolution = (
                validation["validation resolution"]
                .fillna("")
                .astype(str)
                .str.lower()
            )
            same_phase = np.isclose(
                np.mod(validation[PHI_COLUMN], 2.0 * np.pi),
                np.mod(selected_phi, 2.0 * np.pi),
                atol=1.0e-10,
            )
            audits = validation.loc[
                status.str.startswith("success")
                & resolution.eq("refined")
                & np.isclose(validation[H_COLUMN], mean_thickness)
                & np.isclose(validation[Q_COLUMN], selected_q)
                & same_phase
                & np.isfinite(validation[MINIMUM_COLUMN])
            ]
            label_used = False
            for amplitude, color in amplitude_colors.items():
                points = audits.loc[
                    np.isclose(audits[EPSILON_COLUMN], amplitude)
                ]
                if points.empty:
                    continue
                axis.scatter(
                    points[WAVELENGTH_COLUMN],
                    points[MINIMUM_COLUMN],
                    marker="D",
                    s=34,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=7,
                    label="refined audits" if not label_used else None,
                )
                label_used = True

    axis.axhline(0.0, color=COLORS["ink"], lw=0.65, zorder=0)
    axis.axhline(1.0, color=COLORS["grid"], lw=0.8, ls=(0, (2, 2)))
    axis.set_xscale("log")
    ticks, labels = _wavelength_ticks(frame[WAVELENGTH_COLUMN].unique())
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
    axis.set_xlim(
        logarithmic_cell_edges(np.sort(frame[WAVELENGTH_COLUMN].unique()))[[0, -1]]
    )
    minimum = resolved_rows(frame)[MINIMUM_COLUMN].min()
    axis.set_ylim(min(0.35, float(minimum) - 0.08), 1.04)
    axis.set_xlabel(r"Fundamental wavelength, $\lambda/H$")
    axis.set_ylabel(r"Minimum steady $J$")
    axis.grid(color=COLORS["grid"], lw=0.45, alpha=0.75)

    if selected_rows is not None and not selected_rows.empty:
        worst = selected_rows.loc[selected_rows[MINIMUM_COLUMN].idxmin()]
        axis.annotate(
            "largest distortion at\nan intermediate scale",
            xy=(worst[WAVELENGTH_COLUMN], worst[MINIMUM_COLUMN]),
            xytext=(0, 30),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "->",
                "color": COLORS["muted"],
                "lw": 0.7,
            },
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=COLORS["muted"],
        )
    axis.text(
        0.02,
        0.04,
        "short waves:\nvertical screening",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    axis.text(
        0.98,
        0.04,
        "long waves:\ngentle-bed limit",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    axis.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.01),
        handlelength=1.5,
        columnspacing=0.7,
        fontsize=6.9,
    )


def draw_factor_section(axis, frame, selected_amplitude=0.95):
    """Separate orientation change from along-path speed change."""
    rows = resolved_rows(frame)
    rows = rows[
        np.isclose(rows[EPSILON_COLUMN], selected_amplitude)
    ].sort_values(WAVELENGTH_COLUMN)
    required = {MECHANISM_COLUMN, VELOCITY_RATIO_COLUMN}
    if rows.empty or not required.issubset(rows.columns):
        axis.text(
            0.5,
            0.5,
            "factor diagnostics unavailable",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=COLORS["muted"],
        )
        axis.set_axis_off()
        return

    orientation_factor = 1.0 + rows[MECHANISM_COLUMN].to_numpy(dtype=float)
    speed_factor = rows[VELOCITY_RATIO_COLUMN].to_numpy(dtype=float)
    reconstructed_j = orientation_factor * speed_factor

    orientation = axis.plot(
        rows[WAVELENGTH_COLUMN],
        orientation_factor,
        color=COLORS["envelope"],
        marker="o",
        ms=3.2,
        label=r"orientation factor $1+C$",
    )[0]
    speed = axis.plot(
        rows[WAVELENGTH_COLUMN],
        speed_factor,
        color=COLORS["surface"],
        marker="s",
        ms=3.0,
        label=r"speed factor $u/u_{\rm source}$",
    )[0]
    recorded = axis.plot(
        rows[WAVELENGTH_COLUMN],
        rows[MINIMUM_COLUMN],
        color=COLORS["ink"],
        marker="D",
        ms=3.0,
        lw=1.5,
        label=r"recorded minimum $J$",
        zorder=4,
    )[0]
    reconstructed = axis.plot(
        rows[WAVELENGTH_COLUMN],
        reconstructed_j,
        color=COLORS["muted"],
        lw=0.9,
        ls=(0, (2, 2)),
        label=r"$(u/u_{\rm source})(1+C)$",
        zorder=3,
    )[0]
    axis.axhline(0.0, color=COLORS["ink"], lw=0.8)
    axis.axhline(1.0, color=COLORS["grid"], lw=0.8, ls=(0, (2, 2)))
    axis.set_xscale("log")
    ticks, labels = _wavelength_ticks(frame[WAVELENGTH_COLUMN].unique())
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
    axis.set_xlim(
        logarithmic_cell_edges(np.sort(frame[WAVELENGTH_COLUMN].unique()))[[0, -1]]
    )
    factor_values = np.r_[
        orientation_factor,
        speed_factor,
        rows[MINIMUM_COLUMN].to_numpy(dtype=float),
    ]
    axis.set_ylim(
        min(-0.05, float(np.nanmin(factor_values)) - 0.08),
        max(1.25, float(np.nanmax(factor_values)) + 0.08),
    )
    axis.set_xlabel(r"Fundamental wavelength, $\lambda/H$")
    axis.set_ylabel("Exact characteristic factors")
    axis.grid(color=COLORS["grid"], lw=0.45, alpha=0.75)
    axis.text(
        0.98,
        0.04,
        r"for $u>0$, folding requires $1+C<0$",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=COLORS["muted"],
    )

    axis.legend(
        handles=(orientation, speed, recorded, reconstructed),
        loc="upper left",
        frameon=False,
        fontsize=6.8,
        handlelength=1.6,
    )


def _rank_correlation(x, y):
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if len(x_rank) < 3 or np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _binned_lower_envelope(x, y, count=12):
    """Summarize the lower edge without implying a fitted constitutive law."""
    if len(x) < 4:
        return np.asarray(x), np.asarray(y)
    unique = np.unique(x)
    if len(unique) <= 20:
        return unique, np.asarray([np.min(y[np.isclose(x, value)]) for value in unique])
    ranks = pd.Series(x).rank(method="first").to_numpy(dtype=float)
    bins = np.minimum((count * (ranks - 1) / len(ranks)).astype(int), count - 1)
    centers = []
    minima = []
    for index in np.unique(bins):
        take = bins == index
        centers.append(float(np.median(np.asarray(x)[take])))
        minima.append(float(np.min(np.asarray(y)[take])))
    order = np.argsort(centers)
    return np.asarray(centers)[order], np.asarray(minima)[order]


def draw_relationship(
    axis,
    frame,
    x_column,
    x_label,
    *,
    log_x=False,
    counterflow_threshold=None,
):
    if x_column not in frame:
        axis.text(
            0.5,
            0.5,
            f"{x_label}\nnot recorded in sweep",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=COLORS["muted"],
        )
        axis.set_axis_off()
        return
    subset = frame[[x_column, MINIMUM_COLUMN]].replace(
        (np.inf, -np.inf), np.nan
    ).dropna()
    if log_x:
        subset = subset[subset[x_column] > 0.0]
    if subset.empty:
        axis.text(
            0.5,
            0.5,
            f"{x_label}\nnot recorded in sweep",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=COLORS["muted"],
        )
        axis.set_axis_off()
        return

    x = subset[x_column].to_numpy(dtype=float)
    y = subset[MINIMUM_COLUMN].to_numpy(dtype=float)
    if counterflow_threshold is not None and np.min(x) < counterflow_threshold:
        axis.axvspan(
            np.min(x),
            counterflow_threshold,
            color=COLORS["counterflow"],
            lw=0,
            zorder=0,
        )
        axis.text(
            0.02,
            0.04,
            "counterflow",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color=COLORS["muted"],
            fontsize=7.3,
        )
    axis.scatter(
        x,
        y,
        s=8,
        facecolor=COLORS["points"],
        edgecolor="none",
        alpha=0.30,
        rasterized=True,
        zorder=1,
    )
    envelope_x, envelope_y = _binned_lower_envelope(x, y)
    axis.plot(
        envelope_x,
        envelope_y,
        color=COLORS["envelope"],
        marker="o",
        ms=2.8,
        lw=1.2,
        label="binned lower envelope",
        zorder=3,
    )
    worst = int(np.argmin(y))
    axis.scatter(
        [x[worst]],
        [y[worst]],
        s=28,
        facecolor="none",
        edgecolor=COLORS["ink"],
        lw=0.9,
        zorder=4,
    )
    axis.axhline(0.0, color=COLORS["ink"], lw=0.65, zorder=0)
    if counterflow_threshold is not None:
        axis.axvline(counterflow_threshold, color=COLORS["muted"], lw=0.7)
    if log_x:
        axis.set_xscale("log")
    rho = _rank_correlation(x, y)
    if np.isfinite(rho):
        axis.text(
            0.98,
            0.96,
            rf"$\rho_s={rho:.2f}$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=COLORS["muted"],
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(r"Minimum steady $J$")
    axis.grid(color=COLORS["grid"], lw=0.45, alpha=0.75)


def _screening_metric(frame):
    if SCREENING_COLUMN in frame and frame[SCREENING_COLUMN].notna().any():
        values = frame[SCREENING_COLUMN].dropna()
        use_log = bool((values > 0.0).all() and values.max() / values.min() > 40.0)
        return SCREENING_COLUMN, r"Recorded screening measure, $\chi$", use_log
    # kH is the unambiguous nondimensional decay scale of each Fourier mode;
    # it is not silently labeled as the solver's possibly different chi.
    fallback = "computed kH"
    frame[fallback] = 2.0 * np.pi / frame[WAVELENGTH_COLUMN]
    return fallback, r"Fourier screening scale, $kH=2\pi H/\lambda$", True


def _mechanism_metric(frame):
    if MECHANISM_COLUMN in frame and frame[MECHANISM_COLUMN].notna().any():
        return MECHANISM_COLUMN, r"Mechanism at minimum $J$, $C$"
    if RELIEF_COLUMN in frame and frame[RELIEF_COLUMN].notna().any():
        return RELIEF_COLUMN, r"Bed relief$/H$ (fallback; $C$ not recorded)"
    return MECHANISM_COLUMN, r"Mechanism at minimum $J$, $C$"


def _velocity_metric(frame):
    if (
        VELOCITY_RATIO_COLUMN in frame
        and frame[VELOCITY_RATIO_COLUMN].notna().any()
    ):
        return (
            VELOCITY_RATIO_COLUMN,
            r"Velocity at minimum $J$, $u/u_{\mathrm{source}}$",
            0.0,
        )
    source = _find_alias(frame.columns, DIMENSIONAL_MINIMUM_U_ALIASES)
    if source is not None:
        frame["minimum horizontal velocity (m/yr)"] = pd.to_numeric(
            frame[source], errors="coerce"
        )
        return (
            "minimum horizontal velocity (m/yr)",
            r"Minimum horizontal velocity (m yr$^{-1}$; $u_0$ not recorded)",
            0.0,
        )
    return (
        VELOCITY_RATIO_COLUMN,
        r"Velocity at minimum $J$, $u/u_{\mathrm{source}}$",
        0.0,
    )


def save_figure(
    table,
    output_base,
    requested_thickness=160.0,
    dpi=240,
    shape_table=None,
    validation_table=None,
):
    """Save geometry, five fixed-H maps, and three explanatory sections."""
    resolved = resolved_rows(table)
    if resolved.empty:
        raise ValueError("the input contains no resolved finite minimum J")
    thickness = nearest_thickness(resolved, requested_thickness)
    central_resolved = resolved[np.isclose(resolved[H_COLUMN], thickness)].copy()
    central_all = table[np.isclose(table[H_COLUMN], thickness)].copy()
    if central_resolved.empty:
        raise ValueError(f"no resolved cases are available at H={thickness:g} m")

    shape_fixed = pd.DataFrame(columns=central_resolved.columns)
    if shape_table is not None:
        shape_resolved = resolved_rows(shape_table)
        shape_fixed = shape_resolved[
            np.isclose(shape_resolved[H_COLUMN], thickness)
        ].copy()

    color_values = resolved[MINIMUM_COLUMN]
    if not shape_fixed.empty:
        color_values = pd.concat(
            (color_values, shape_fixed[MINIMUM_COLUMN]), ignore_index=True
        )
    norm = choose_j_norm(color_values)
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(11.0, 9.65),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.94, 1.0, 1.06)},
    )

    shape_frame = shape_fixed if not shape_fixed.empty else central_resolved
    shape_worst = shape_frame.loc[shape_frame[MINIMUM_COLUMN].idxmin()]
    selected_q = np.sort(table[Q_COLUMN].dropna().unique())
    selected_phi = np.sort(table[PHI_COLUMN].dropna().unique())
    if len(selected_q) != 1 or len(selected_phi) != 1:
        raise ValueError(
            "selected-shape parameter map must contain exactly one q and phi"
        )
    geometry_rows = central_resolved[
        np.isclose(central_resolved[EPSILON_COLUMN], 0.95)
    ]
    if geometry_rows.empty:
        geometry_rows = central_resolved
    geometry_row = geometry_rows.loc[geometry_rows[MINIMUM_COLUMN].idxmin()]
    draw_geometry(axes[0, 0], geometry_row)
    axes[0, 0].set_title(
        "A   Selected two-harmonic geometry\n"
        rf"illustrated at $A/H={geometry_row[EPSILON_COLUMN]:g}$; "
        rf"$q={selected_q[0]:g}$; $\phi={selected_phi[0] / np.pi:g}\pi$",
        loc="left",
        pad=4.0,
    )

    map_axes = (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2])
    thicknesses = np.sort(table[H_COLUMN].dropna().unique())
    if len(thicknesses) != len(map_axes):
        raise ValueError(
            f"expected five mean-thickness slices, found {len(thicknesses)}"
        )
    images = []
    map_worsts = []
    for letter, mean_thickness, axis in zip("BCDEF", thicknesses, map_axes):
        fixed_all = table[np.isclose(table[H_COLUMN], mean_thickness)].copy()
        image, worst = draw_parameter_map(axis, fixed_all, norm)
        images.append(image)
        map_worsts.append(worst)
        axis.set_title(
            rf"{letter}   $H={mean_thickness:g}$ m; "
            rf"$\beta H/\eta={mean_thickness / 2500.0:.3f}$",
            loc="left",
            pad=4.0,
        )
    for index, axis in enumerate(map_axes):
        if index in (2, 3, 4):
            axis.set_xlabel(r"Fundamental wavelength, $\lambda/H$")
        else:
            axis.set_xlabel("")
            axis.tick_params(labelbottom=False)
        if index not in (0, 2):
            axis.set_ylabel("")
            axis.tick_params(labelleft=False)

    colorbar = figure.colorbar(
        images[0],
        ax=list(map_axes),
        orientation="horizontal",
        fraction=0.047,
        pad=0.055,
        aspect=48,
    )
    colorbar.set_label(
        r"Minimum steady-characteristic $J=T_x$ "
        r"(central 80% of depositing flux)"
    )
    colorbar.set_ticks(
        (norm.vmin, 0.5 * norm.vmin, 0.0, 0.5 * norm.vmax, norm.vmax)
    )
    handles = status_legend(table)
    if handles:
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.006),
            frameon=False,
            fontsize=7.0,
            ncol=len(handles),
        )

    draw_wavelength_section(
        axes[2, 0],
        central_all,
        selected_amplitude=0.85,
        validation_table=validation_table,
    )
    axes[2, 0].set_title(
        rf"G   Broad-shape wavelength sections at $H={thickness:g}$ m",
        loc="left",
        pad=35.0,
    )

    draw_shape_map(axes[2, 1], shape_frame, shape_worst, norm)
    axes[2, 1].set_title(
        "H   Two-harmonic shape scan\n"
        rf"thin neck: $H={thickness:g}$ m; "
        rf"$A/H={shape_worst[EPSILON_COLUMN]:g}$; "
        rf"$\lambda/H={shape_worst[WAVELENGTH_COLUMN]:g}$",
        loc="left",
        pad=28.0,
        fontsize=9.0,
    )

    draw_factor_section(axes[2, 2], central_all, selected_amplitude=0.95)
    axes[2, 2].set_title(
        "I   Why the minimum stays positive\n"
        rf"thin neck: $A/H=0.95$; $q={selected_q[0]:g}$; "
        rf"$\phi={selected_phi[0] / np.pi:g}\pi$",
        loc="left",
        pad=28.0,
    )

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    selected_worst = min(map_worsts, key=lambda row: row[MINIMUM_COLUMN])
    return thickness, selected_worst, shape_worst


def parse_arguments():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=root / "data" / "steady_characteristic_two_harmonic_parameter_map.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "figures" / "steady_characteristic_two_harmonic_parameter_map",
        help="output path without extension (both PDF and PNG are written)",
    )
    parser.add_argument(
        "--shape-csv",
        "--shape-scan",
        dest="shape_csv",
        type=Path,
        help=(
            "optional q--phi shape-scan CSV; when omitted, available sibling "
            "general and thin-neck shape scans are merged"
        ),
    )
    parser.add_argument("--mean-thickness", type=float, default=160.0)
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=root / "data" / "steady_characteristic_two_harmonic_validation.csv",
        help="optional focused-convergence table used for open-diamond audits",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main():
    args = parse_arguments()
    configure_matplotlib()
    table = load_table(args.input)
    if args.shape_csv is not None:
        shape_paths = [args.shape_csv]
    else:
        shape_paths = []
        for name in (
            "steady_characteristic_two_harmonic_shape_scan.csv",
            "steady_characteristic_two_harmonic_thin_neck_shape_scan.csv",
        ):
            candidate = args.input.with_name(name)
            if candidate.exists() and candidate.resolve() != args.input.resolve():
                shape_paths.append(candidate)
    loaded_shape_tables = [load_table(path) for path in shape_paths]
    shape_table = (
        pd.concat(loaded_shape_tables, ignore_index=True)
        if loaded_shape_tables
        else None
    )
    validation_table = None
    if args.validation_csv is not None and args.validation_csv.exists():
        validation_table = pd.read_csv(args.validation_csv)
    thickness, worst, shape_worst = save_figure(
        table,
        args.output,
        requested_thickness=args.mean_thickness,
        dpi=args.dpi,
        shape_table=shape_table,
        validation_table=validation_table,
    )
    print(
        f"Wrote {args.output.with_suffix('.pdf')} and "
        f"{args.output.with_suffix('.png')} at H={thickness:g} m; "
        f"minimum J={worst[MINIMUM_COLUMN]:.6g} at "
        f"A/H={worst[EPSILON_COLUMN]:g}, "
        f"lambda/H={worst[WAVELENGTH_COLUMN]:g}, "
        f"q={worst[Q_COLUMN]:g}, phi/pi={worst[PHI_COLUMN] / np.pi:g}; "
        f"shape-scan minimum J={shape_worst[MINIMUM_COLUMN]:.6g}."
    )


if __name__ == "__main__":
    main()
