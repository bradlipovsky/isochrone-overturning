"""Plot the smooth periodic convergent-basin diagnostic and audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steady_characteristic_periodic_convergent import (
    METHOD_VERSION,
    PeriodicConvergentGeometry,
)


COLORS = {
    1.0: "#4C78A8",
    8.0: "#E07B39",
    12.0: "#5B8C5A",
}


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


def panel_label(axis, label):
    axis.text(
        -0.12,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def draw_geometry(axis):
    x = np.linspace(0.0, 1.0, 1601)
    for q, relief, linestyle in ((8.0, 2.75, "-"), (12.0, 2.0, "--")):
        geometry = PeriodicConvergentGeometry(q, relief, 0.10)
        dimensional_x = x * geometry.wavelength
        bed = geometry.relief_height(dimensional_x) / geometry.mean_thickness
        surface = bed + 1.0
        color = COLORS[q]
        axis.plot(
            x,
            bed,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=rf"$q={q:g},\ \Delta b/H={relief:g}$",
        )
        axis.plot(
            x,
            surface,
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            alpha=0.85,
        )
    axis.axvline(0.0, color="0.35", linewidth=0.8)
    axis.axvline(1.0, color="0.35", linewidth=0.8)
    axis.text(
        0.5,
        0.98,
        "bed and surface are parallel (constant thickness)",
        transform=axis.transAxes,
        ha="center",
        va="top",
        color="0.25",
        fontsize=8.2,
    )
    axis.annotate(
        "periodic seam",
        xy=(1.0, 2.85),
        xytext=(0.80, 3.28),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.3"},
        color="0.25",
        fontsize=8.2,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.05, 3.85)
    axis.set_xlabel(r"Periodic phase, $x/L$")
    axis.set_ylabel(r"Elevation / $H$")
    axis.set_title("Geometry after wrapped-Gaussian smoothing")
    axis.legend(loc="lower right", frameon=False)
    panel_label(axis, "A")


def draw_smoothing(axis, table):
    production = table.loc[
        table["run kind"].eq("production")
        & table["mean driving slope (rad)"].eq(0.0)
        & table["coordinate gauge phase / period"].eq(0.0)
    ]
    for q in (1.0, 8.0, 12.0):
        subset = production.loc[
            production["right / left length ratio"].eq(q)
        ].sort_values("wrapped-Gaussian sigma / short flank length")
        x = subset["wrapped-Gaussian sigma / short flank length"]
        color = COLORS[q]
        axis.plot(
            x,
            subset["minimum J"],
            color=color,
            marker="o",
            linewidth=1.5,
            markersize=4.5,
            label=rf"$q={q:g}$: isolated minimum",
        )
        axis.plot(
            x,
            subset["first percentile of per-source minimum J"],
            color=color,
            marker="s",
            linestyle="--",
            linewidth=1.1,
            markersize=4.0,
            markerfacecolor="white",
            label=rf"$q={q:g}$: source p01",
        )
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks((0.05, 0.10, 0.20, 0.40))
    axis.set_xticklabels(("0.05", "0.10", "0.20", "0.40"))
    axis.set_ylim(-1.45, 1.12)
    axis.set_xlabel(r"Smoothing width, $\sigma/L_{\rm short}$")
    axis.set_ylabel(r"Minimum steady $J$")
    axis.set_title("Zero-mean forcing: negative values are isolated")
    axis.legend(ncol=2, loc="lower right", frameon=False, columnspacing=0.8)
    panel_label(axis, "B")


def draw_separatrix(axis, table):
    density = table.loc[
        table["run kind"].eq("q12 source-density audit")
    ].sort_values("candidate source count")
    x = density["candidate source count"].to_numpy(float)
    axis.plot(
        x,
        density["minimum J"],
        color=COLORS[12.0],
        marker="o",
        linewidth=1.5,
        label="isolated minimum",
    )
    axis.plot(
        x,
        density["first percentile of per-source minimum J"],
        color=COLORS[12.0],
        marker="s",
        markerfacecolor="white",
        linestyle="--",
        linewidth=1.1,
        label="source p01",
    )
    for _, row in density.iterrows():
        count = int(row.get("negative-J retained-source count", 0))
        axis.annotate(
            f"{count} source" + ("" if count == 1 else "s"),
            (
                row["candidate source count"],
                row["minimum J"],
            ),
            xytext=(3, -11 if row["minimum J"] < 0 else 5),
            textcoords="offset points",
            fontsize=7.4,
            color="0.25",
        )
    gauge = table.loc[
        table["run kind"].isin(
            ("q12 coordinate gauge audit", "q12 coordinate gauge full")
        )
    ]
    axis.scatter(
        gauge["candidate source count"] * 1.025,
        gauge["minimum J"],
        marker="D",
        s=32,
        facecolors="none",
        edgecolors="#8E5EA2",
        linewidths=1.2,
        label="same geometry, phase shifted 1/4 period",
        zorder=4,
    )
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.set_yscale("symlog", linthresh=0.5, linscale=0.7)
    axis.set_xlabel("Candidate surface-source count")
    axis.set_ylabel(r"Minimum steady $J$ (symlog)")
    axis.set_title(
        r"$q=12$, $\sigma/L_{\rm short}=0.10$: sub-percent overturning band"
    )
    axis.grid(axis="y", color="0.88", linewidth=0.5)

    flux_axis = axis.twinx()
    flux_axis.plot(
        x,
        100.0
        * density["negative-J retained accumulation-flux fraction"],
        color="#C44E52",
        marker="^",
        linestyle=":",
        linewidth=1.2,
        label="negative-J source flux",
    )
    flux_axis.set_ylabel("Negative-J retained flux (%)", color="#9C3D40")
    flux_axis.tick_params(axis="y", colors="#9C3D40")
    flux_axis.set_ylim(-0.03, 0.64)
    lines, labels = axis.get_legend_handles_labels()
    extra, extra_labels = flux_axis.get_legend_handles_labels()
    axis.legend(lines + extra, labels + extra_labels, loc="lower left", frameon=False)
    panel_label(axis, "C")


def draw_mean_slope(axis, table):
    baseline = table.loc[
        table["run kind"].eq("production")
        & table["right / left length ratio"].eq(8.0)
        & table["wrapped-Gaussian sigma / short flank length"].eq(0.10)
        & table["mean driving slope (rad)"].eq(0.0)
    ]
    continuation = table.loc[
        table["run kind"].eq("mean-slope continuation")
    ]
    curve = pd.concat((baseline, continuation), ignore_index=True).sort_values(
        "mean driving slope (rad)"
    )
    x = curve["mean driving slope (rad)"]
    axis.plot(
        x,
        curve["minimum J"],
        color=COLORS[8.0],
        marker="o",
        linewidth=1.4,
        markersize=4.3,
        label="isolated minimum",
    )
    axis.plot(
        x,
        curve["first percentile of per-source minimum J"],
        color=COLORS[8.0],
        marker="s",
        markerfacecolor="white",
        linestyle="--",
        linewidth=1.2,
        markersize=4.0,
        label="source p01",
    )
    audits = table.loc[
        table["run kind"].str.contains("mean-slope")
        & ~table["run kind"].eq("mean-slope continuation")
        & ~table["run kind"].eq("mean-slope fine")
    ]
    axis.scatter(
        audits["mean driving slope (rad)"],
        audits["minimum J"],
        marker="o",
        s=24,
        facecolors="none",
        edgecolors="0.25",
        linewidths=0.9,
        label="refinement minima",
        zorder=4,
    )
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.axvspan(0.14, 0.305, color="#DCEAF5", alpha=0.7, zorder=0)
    axis.text(
        0.205,
        0.96,
        "one-directional throughflow",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#3B6C8E",
        fontsize=8.0,
    )
    axis.set_yscale("symlog", linthresh=0.5, linscale=0.7)
    axis.set_xlim(-0.005, 0.305)
    axis.set_xlabel(r"Mean driving slope, $\alpha$ (rad)")
    axis.set_ylabel(r"Minimum steady $J$ (symlog)")
    axis.set_title("A narrow counterflow fold vanishes in throughflow")
    axis.grid(axis="y", color="0.88", linewidth=0.5)
    axis.legend(loc="lower right", frameon=False)
    panel_label(axis, "D")


def make_figure(table):
    configure_plotting()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 8.2),
        constrained_layout=True,
    )
    draw_geometry(axes[0, 0])
    draw_smoothing(axes[0, 1], table)
    draw_separatrix(axes[1, 0], table)
    draw_mean_slope(axes[1, 1], table)
    figure.suptitle(
        "Smooth periodic convergence: narrow separatrix-associated "
        "overturning; magnitude unresolved",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.012,
        "Direct steady characteristics; divergence-free streamfunction "
        "reconstruction enforces zero bed-normal flow. Squares show the "
        "per-source first percentile (1% source-quantile check).",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="0.25",
    )
    return figure


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_periodic_convergent.csv",
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
    figure = make_figure(table)
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
