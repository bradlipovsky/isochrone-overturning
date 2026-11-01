"""Plot the geometry, flow, and audits for smooth periodic convergence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from steady_characteristic_periodic_convergent import (
    METHOD_VERSION,
    PeriodicConvergentGeometry,
)


BED_COLOR = "#5B4336"
SURFACE_COLOR = "#177A9C"
ICE_COLOR = "#D8ECF3"
FLOW_COLOR = "#2E6075"
NEGATIVE_COLOR = "#B43C45"
POSITIVE_COLOR = "#2B6EA6"
ABLATION_COLOR = "#D06A4E"


def configure_plotting():
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def panel_title(axis, label, title):
    axis.set_title(rf"$\bf{{{label}}}$  {title}", loc="left", pad=7)


def close_periodic(profile):
    closed = pd.concat((profile, profile.iloc[[0]].copy()), ignore_index=True)
    closed.loc[closed.index[-1], "phase x/L"] = 1.0
    return closed


def periodic_zero_crossings(x, values):
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    result = []
    x_closed = np.r_[x, 1.0]
    v_closed = np.r_[values, values[0]]
    for x0, x1, y0, y1 in zip(
        x_closed[:-1], x_closed[1:], v_closed[:-1], v_closed[1:]
    ):
        if y0 == 0.0:
            result.append(float(x0 % 1.0))
        elif np.signbit(y0) != np.signbit(y1):
            fraction = -y0 / (y1 - y0)
            result.append(float((x0 + fraction * (x1 - x0)) % 1.0))
    return sorted(result)


def representative_divide_phase(profile):
    """Return the high surface-flow divide used as the plotting origin."""
    crossings = periodic_zero_crossings(
        profile["phase x/L"], profile["surface u (m/yr)"]
    )
    return max(crossings)


def shift_profile_to_divide(profile, divide_phase):
    shifted = profile.copy()
    shifted["phase x/L"] = np.mod(
        shifted["phase x/L"].to_numpy(float) - divide_phase, 1.0
    )
    return shifted.sort_values("phase x/L").reset_index(drop=True)


def representative_negative_phase(table):
    row = table.loc[
        table["run kind"].eq("q12 source-density audit")
        & table["candidate source count"].eq(721)
    ]
    if row.empty:
        row = table.loc[
            table["run kind"].eq("negative audit full")
            & table["right / left length ratio"].eq(12.0)
            & table["wrapped-Gaussian sigma / short flank length"].eq(0.10)
        ]
    return float(row.iloc[0]["source phase at minimum J"])


def draw_velocity_arrow(axis, phase, elevation, u, w, geometry, span_y):
    """Draw one fixed-display-length arrow with the solved flow direction."""
    coordinate_velocity = np.array(
        [u / geometry.wavelength, w / geometry.mean_thickness], dtype=float
    )
    axes_velocity = np.array(
        [coordinate_velocity[0], coordinate_velocity[1] / span_y]
    )
    magnitude = np.linalg.norm(axes_velocity)
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        return
    axes_step = 0.053 * axes_velocity / magnitude
    data_step = np.array([axes_step[0], axes_step[1] * span_y])
    start = np.array([phase, elevation]) - 0.5 * data_step
    end = np.array([phase, elevation]) + 0.5 * data_step
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.25,
            "color": FLOW_COLOR,
            "mutation_scale": 9,
        },
        zorder=7,
    )


def draw_geometry(axis, flow, surface_profile, table):
    geometry = PeriodicConvergentGeometry(12.0, 2.0, 0.10)
    phase = np.linspace(0.0, 1.0, 2001)
    divide_phase = representative_divide_phase(surface_profile)
    original_phase = np.mod(phase + divide_phase, 1.0)
    x = original_phase * geometry.wavelength
    bed_unshifted = geometry.relief_height(x) / geometry.mean_thickness
    vertical_shift = -(float(np.max(bed_unshifted)) + 1.0)
    bed = bed_unshifted + vertical_shift
    surface = bed + 1.0
    trough_phase = float(phase[np.argmin(bed)])
    negative_phase = float(
        np.mod(representative_negative_phase(table) - divide_phase, 1.0)
    )

    axis.fill_between(phase, bed, surface, color=ICE_COLOR, zorder=0)
    axis.plot(phase, bed, color=BED_COLOR, linewidth=2.0, zorder=4)
    axis.plot(phase, surface, color=SURFACE_COLOR, linewidth=1.8, zorder=5)
    axis.axvline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    axis.axvline(1.0, color="0.45", linewidth=0.8, linestyle=":")

    arrow_phases = (0.018, 0.045, 0.075, 0.103, 0.19, 0.38, 0.62, 0.84, 0.96)
    mid = flow.loc[
        np.isclose(flow["vertical fraction above bed"], 0.50)
    ].copy()
    mid["display phase"] = np.mod(
        mid["phase x/L"].to_numpy(float) - divide_phase, 1.0
    )
    span_y = 3.55
    for requested in arrow_phases:
        row = mid.iloc[np.argmin(np.abs(mid["display phase"] - requested))]
        draw_velocity_arrow(
            axis,
            float(row["display phase"]),
            float(row["physical elevation/H"]) + vertical_shift,
            float(row["u (m/yr)"]),
            float(row["physical w (m/yr)"]),
            geometry,
            span_y,
        )

    axis.annotate(
        "",
        xy=(0.995, 0.015),
        xytext=(0.005, 0.015),
        arrowprops={"arrowstyle": "<->", "lw": 0.9, "color": "0.25"},
    )
    axis.text(
        0.50,
        0.035,
        "one period " + r"$L=30H=6$ km" + "; endpoints are the same flow divide",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    axis.annotate(
        "convergence + upwelling\n(required surface ablation)",
        xy=(trough_phase, np.interp(trough_phase, phase, surface)),
        xytext=(0.29, -2.72),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.3"},
        ha="center",
        va="top",
        fontsize=8.0,
    )
    source_y = float(np.interp(negative_phase, phase, surface))
    axis.scatter(
        [negative_phase],
        [source_y],
        s=42,
        facecolors="none",
        edgecolors=NEGATIVE_COLOR,
        linewidths=1.5,
        zorder=9,
    )
    axis.annotate(
        "fold-source band\n(~0.5% retained flux)",
        xy=(negative_phase, source_y),
        xytext=(0.105, -0.43),
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.9,
            "color": NEGATIVE_COLOR,
        },
        color=NEGATIVE_COLOR,
        fontsize=7.9,
        ha="left",
    )
    axis.annotate(
        "short, steep flank\nflow to the right",
        xy=(0.055, -1.30),
        xytext=(0.17, -1.62),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": FLOW_COLOR},
        ha="center",
        va="center",
        fontsize=8.0,
        color=FLOW_COLOR,
    )
    axis.text(
        0.58,
        -1.50,
        "long, gentle flank: flow to the left",
        ha="center",
        va="center",
        fontsize=8.0,
        color=FLOW_COLOR,
    )

    h_phase = 0.69
    local_bed = float(np.interp(h_phase, phase, bed))
    axis.annotate(
        "",
        xy=(h_phase, local_bed + 1.0),
        xytext=(h_phase, local_bed),
        arrowprops={"arrowstyle": "<->", "lw": 0.9, "color": "0.3"},
    )
    axis.text(h_phase + 0.015, local_bed + 0.5, r"$H$", va="center")
    axis.text(
        0.58,
        -0.33,
        r"$q=L_{\rm long}/L_{\rm short}=12$    "
        r"$\Delta b/H=2$" "\n"
        r"$\sigma/L_{\rm short}=0.10$    $\alpha=0$",
        ha="center",
        va="center",
        fontsize=8.0,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 2.0,
        },
    )
    axis.text(
        0.99,
        -3.08,
        "arrows: solved velocity\n(lengths normalized)",
        ha="right",
        va="bottom",
        fontsize=7.3,
        color="0.35",
    )

    axis.set_xlim(-0.008, 1.008)
    axis.set_ylim(-3.16, 0.27)
    axis.set_xlabel(r"Distance from the flow divide, $\xi/L$")
    axis.set_ylabel(r"Elevation / $H$ (vertically exaggerated)")
    panel_title(axis, "A", "Representative overturning geometry and solved flow")


def draw_surface_kinematics(specification, profile, table):
    subgrid = specification.subgridspec(
        2, 1, height_ratios=(1.0, 0.82), hspace=0.08
    )
    velocity_axis = plt.subplot(subgrid[0])
    balance_axis = plt.subplot(subgrid[1], sharex=velocity_axis)
    divide_phase = representative_divide_phase(profile)
    shifted = shift_profile_to_divide(profile, divide_phase)
    closed = close_periodic(shifted)
    phase = closed["phase x/L"].to_numpy(float)
    u = closed["surface u (m/yr)"].to_numpy(float)
    accumulation = closed["required accumulation a=-W (m/yr)"].to_numpy(float)
    trough_phase = float(
        shifted.iloc[np.argmin(shifted["bed elevation/H"])]["phase x/L"]
    )
    negative_phase = float(
        np.mod(representative_negative_phase(table) - divide_phase, 1.0)
    )

    velocity_axis.plot(phase, u, color=FLOW_COLOR, linewidth=1.7)
    velocity_axis.axhline(0.0, color="0.3", linewidth=0.8)
    velocity_axis.axvline(trough_phase, color="0.45", linestyle=":", linewidth=0.8)
    velocity_axis.axvline(
        negative_phase,
        color=NEGATIVE_COLOR,
        linestyle="--",
        linewidth=1.0,
    )
    velocity_axis.scatter(
        [negative_phase],
        [np.interp(negative_phase, phase, u)],
        s=32,
        facecolors="none",
        edgecolors=NEGATIVE_COLOR,
        linewidths=1.2,
        zorder=6,
    )
    velocity_axis.text(
        0.052,
        0.82,
        r"$u_s>0$",
        transform=velocity_axis.get_xaxis_transform(),
        ha="center",
        color=FLOW_COLOR,
    )
    velocity_axis.text(
        0.56,
        0.12,
        r"$u_s<0$",
        transform=velocity_axis.get_xaxis_transform(),
        ha="center",
        color=FLOW_COLOR,
    )
    velocity_axis.annotate(
        "convergence",
        xy=(trough_phase, 0.0),
        xytext=(0.20, 0.12),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.3"},
        fontsize=7.7,
    )
    velocity_axis.annotate(
        "fold source beside the flow divide",
        xy=(negative_phase, np.interp(negative_phase, phase, u)),
        xytext=(0.58, 0.70),
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.8,
            "color": NEGATIVE_COLOR,
        },
        color=NEGATIVE_COLOR,
        fontsize=7.7,
        ha="center",
    )
    velocity_axis.set_ylabel(r"$u_s$ (m yr$^{-1}$)")
    velocity_axis.tick_params(axis="x", labelbottom=False)
    velocity_axis.set_xlim(0.0, 1.0)
    panel_title(
        velocity_axis,
        "B",
        "Surface flow and required steady mass balance",
    )

    balance_axis.axhline(0.0, color="0.3", linewidth=0.8)
    balance_axis.fill_between(
        phase,
        0.0,
        accumulation,
        where=accumulation >= 0.0,
        color=POSITIVE_COLOR,
        alpha=0.38,
        interpolate=True,
        label="accumulation",
    )
    balance_axis.fill_between(
        phase,
        0.0,
        accumulation,
        where=accumulation < 0.0,
        color=ABLATION_COLOR,
        alpha=0.42,
        interpolate=True,
        label="ablation",
    )
    balance_axis.plot(phase, accumulation, color="0.25", linewidth=0.8)
    balance_axis.axvline(trough_phase, color="0.45", linestyle=":", linewidth=0.8)
    balance_axis.axvline(
        negative_phase,
        color=NEGATIVE_COLOR,
        linestyle="--",
        linewidth=1.0,
    )
    balance_axis.text(
        0.76,
        0.72,
        "accumulation",
        transform=balance_axis.transAxes,
        color=POSITIVE_COLOR,
        ha="center",
    )
    balance_axis.text(
        0.26,
        0.15,
        "ablation",
        transform=balance_axis.transAxes,
        color="#A34735",
        ha="center",
    )
    balance_axis.set_ylabel(r"surface balance, $a$ (m yr$^{-1}$)")
    balance_axis.set_xlabel(r"Distance from the flow divide, $\xi/L$")
    balance_axis.set_xlim(0.0, 1.0)
    return velocity_axis, balance_axis


def draw_smoothing_heatmap(axis, table):
    families = ((1.0, 4.0), (8.0, 2.75), (12.0, 2.0))
    smoothing = (0.05, 0.10, 0.20, 0.40)
    production = table.loc[
        table["run kind"].eq("production")
        & table["mean driving slope (rad)"].eq(0.0)
        & table["coordinate gauge phase / period"].eq(0.0)
    ]
    values = np.full((len(families), len(smoothing)), np.nan)
    for row_index, (q, relief) in enumerate(families):
        for column_index, width in enumerate(smoothing):
            selected = production.loc[
                production["right / left length ratio"].eq(q)
                & production["bed relief / ice thickness"].eq(relief)
                & production[
                    "wrapped-Gaussian sigma / short flank length"
                ].eq(width)
            ]
            values[row_index, column_index] = float(selected.iloc[0]["minimum J"])

    image = axis.imshow(
        values,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.0),
        aspect="auto",
        interpolation="nearest",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_color = "white" if abs(value) > 0.65 else "0.18"
            suffix = "*" if value < 0.0 else ""
            axis.text(
                column,
                row,
                f"{value:.2f}{suffix}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.6,
            )
    axis.set_xticks(np.arange(len(smoothing)))
    axis.set_xticklabels([f"{value:.2f}" for value in smoothing])
    axis.set_yticks(np.arange(len(families)))
    axis.set_yticklabels(
        [
            rf"$q={q:g},\ \Delta b/H={relief:g}$"
            for q, relief in families
        ]
    )
    axis.set_xlabel(r"Smoothing width, $\sigma/L_{\rm short}$")
    axis.set_ylabel("Geometry family")
    panel_title(axis, "C", "Zero-mean smoothing scan")
    colorbar = axis.figure.colorbar(image, ax=axis, pad=0.025, fraction=0.047)
    colorbar.set_label(r"isolated minimum steady $J$")
    axis.text(
        0.0,
        -0.26,
        "* sign persists under mesh refinement and a quarter-period shift; "
        "magnitude unresolved.  Source p01 ≥ 0.987 in every cell.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="0.30",
    )


def draw_mean_slope(specification, table):
    subgrid = specification.subgridspec(
        2, 1, height_ratios=(0.78, 1.15), hspace=0.08
    )
    velocity_axis = plt.subplot(subgrid[0])
    order_axis = plt.subplot(subgrid[1], sharex=velocity_axis)
    baseline = table.loc[
        table["run kind"].eq("production")
        & table["right / left length ratio"].eq(8.0)
        & table["wrapped-Gaussian sigma / short flank length"].eq(0.10)
        & table["mean driving slope (rad)"].eq(0.0)
    ]
    continuation = table.loc[table["run kind"].eq("mean-slope continuation")]
    curve = pd.concat((baseline, continuation), ignore_index=True).sort_values(
        "mean driving slope (rad)"
    )
    x = curve["mean driving slope (rad)"].to_numpy(float)
    minimum_u = curve["minimum sampled u (m/yr)"].to_numpy(float)
    crossing_index = np.flatnonzero(
        np.signbit(minimum_u[:-1]) != np.signbit(minimum_u[1:])
    )[-1]
    crossing = x[crossing_index] + (
        -minimum_u[crossing_index]
        * (x[crossing_index + 1] - x[crossing_index])
        / (minimum_u[crossing_index + 1] - minimum_u[crossing_index])
    )

    for axis in (velocity_axis, order_axis):
        axis.axvspan(0.0, crossing, color="#F7E4DF", alpha=0.65, zorder=0)
        axis.axvspan(crossing, 0.305, color="#DCEAF5", alpha=0.72, zorder=0)
        axis.axvline(crossing, color="0.35", linestyle="--", linewidth=0.9)

    velocity_axis.plot(x, minimum_u, color=FLOW_COLOR, marker="o", markersize=3.8)
    velocity_axis.axhline(0.0, color="0.25", linewidth=0.8)
    velocity_axis.text(
        0.053,
        0.14,
        "counterflow",
        transform=velocity_axis.get_xaxis_transform(),
        ha="center",
        color="#984D3C",
        fontsize=7.8,
    )
    velocity_axis.text(
        0.22,
        0.82,
        "one-directional throughflow",
        transform=velocity_axis.get_xaxis_transform(),
        ha="center",
        color="#3B6C8E",
        fontsize=7.8,
    )
    velocity_axis.annotate(
        rf"$u_{{\min}}=0$ at $\alpha\approx{crossing:.3f}$",
        xy=(crossing, 0.0),
        xytext=(0.48, 0.60),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.3"},
        fontsize=7.5,
    )
    velocity_axis.set_ylabel(r"min $u$ (m yr$^{-1}$)")
    velocity_axis.tick_params(axis="x", labelbottom=False)
    velocity_axis.set_xlim(-0.005, 0.305)
    panel_title(
        velocity_axis,
        "D",
        r"Mean-slope continuation ($q=8$, $\Delta b/H=2.75$, "
        r"$\sigma/L_{\rm short}=0.10$)",
    )

    minimum_j = curve["minimum J"].to_numpy(float)
    p01 = curve["first percentile of per-source minimum J"].to_numpy(float)
    order_axis.plot(
        x,
        p01,
        color=POSITIVE_COLOR,
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markersize=3.8,
        linewidth=1.1,
        label="source p01",
    )
    ordered = minimum_j >= 0.0
    order_axis.scatter(
        x[ordered],
        minimum_j[ordered],
        s=22,
        facecolors="white",
        edgecolors="0.25",
        linewidths=0.9,
        label="isolated minimum",
        zorder=4,
    )
    order_axis.scatter(
        x[~ordered],
        minimum_j[~ordered],
        s=28,
        color=NEGATIVE_COLOR,
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
    )
    audits = table.loc[
        table["run kind"].str.contains("mean-slope")
        & ~table["run kind"].eq("mean-slope continuation")
        & ~table["run kind"].eq("mean-slope fine")
    ]
    order_axis.scatter(
        audits["mean driving slope (rad)"],
        audits["minimum J"],
        marker="x",
        s=24,
        color="0.25",
        linewidths=0.8,
        label="mesh/source audits",
        zorder=6,
    )
    order_axis.axhline(0.0, color="0.25", linewidth=0.8)
    order_axis.set_yscale("symlog", linthresh=0.5, linscale=0.7)
    order_axis.set_yticks((-100.0, -10.0, -1.0, 0.0, 1.0))
    order_axis.set_yticklabels(
        (r"$-10^2$", r"$-10^1$", r"$-10^0$", "$0$", r"$10^0$")
    )
    order_axis.set_xlabel(r"Mean driving slope, $\alpha$ (rad)")
    order_axis.set_ylabel(r"minimum steady $J$ (symlog)")
    order_axis.set_xlim(-0.005, 0.305)
    order_axis.grid(axis="y", color="0.88", linewidth=0.5)
    order_axis.legend(loc="lower right", frameon=False, ncol=2)
    return velocity_axis, order_axis


def make_figure(table, surface_profile, flow):
    configure_plotting()
    figure = plt.figure(figsize=(13.1, 9.6), constrained_layout=True)
    outer = figure.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    geometry_axis = figure.add_subplot(outer[0, 0])
    draw_geometry(geometry_axis, flow, surface_profile, table)
    draw_surface_kinematics(outer[0, 1], surface_profile, table)
    smoothing_axis = figure.add_subplot(outer[1, 0])
    draw_smoothing_heatmap(smoothing_axis, table)
    draw_mean_slope(outer[1, 1], table)
    figure.suptitle(
        "Smooth periodic convergence: geometry, flow topology, and the narrow fold",
        fontsize=14,
    )
    return figure


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=repository / "data" / "steady_characteristic_periodic_convergent.csv",
    )
    parser.add_argument(
        "--surface",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_periodic_convergent_surface.csv",
    )
    parser.add_argument(
        "--flow",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_periodic_convergent_flow.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "figures"
        / "steady_characteristic_periodic_convergent",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    table = pd.read_csv(args.data)
    methods = set(table["diagnostic method version"].dropna().astype(str))
    if methods != {METHOD_VERSION}:
        raise ValueError(f"Expected only {METHOD_VERSION}; found {sorted(methods)}")
    surface_profile = pd.read_csv(args.surface)
    flow = pd.read_csv(args.flow)
    figure = make_figure(table, surface_profile, flow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    pdf = args.output.with_suffix(".pdf")
    figure.savefig(png, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
