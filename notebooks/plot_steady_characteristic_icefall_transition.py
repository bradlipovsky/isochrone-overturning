"""Plot the smooth icefall-to-gentle-tongue steady-characteristic sweep."""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


DEFAULT_MEAN_THICKNESS = 160.0


def cell_edges(values):
    values = np.asarray(values, dtype=float)
    middle = 0.5 * (values[1:] + values[:-1])
    return np.r_[values[0] - (middle[0] - values[0]), middle, values[-1] + (values[-1] - middle[-1])]


def panel_label(axis, letter, title):
    axis.set_title(
        f"{letter}  {title}",
        loc="left",
        pad=8,
        fontweight="semibold",
    )


def draw_geometry(axis):
    mean_thickness = DEFAULT_MEAN_THICKNESS
    wavelength = 4.0 * mean_thickness
    slope_ratio = 12.0
    thickness_ratio = 3.0
    sharpness = 4
    mean_slope = 0.05
    window_mean = comb(2 * sharpness, sharpness) / 4.0**sharpness
    coefficients = np.array(
        [
            2.0 ** (1 - 2 * sharpness)
            * comb(2 * sharpness, sharpness - harmonic)
            for harmonic in range(1, sharpness + 1)
        ]
    )
    harmonics = np.arange(1, sharpness + 1, dtype=float)
    gentle_slope = mean_slope / (
        1.0 + (slope_ratio - 1.0) * window_mean
    )
    slope_contrast = gentle_slope * (slope_ratio - 1.0)
    thickness_contrast = (thickness_ratio - 1.0) / (
        window_mean + thickness_ratio * (1.0 - window_mean)
    )
    x = np.linspace(-0.5 * wavelength, 0.5 * wavelength, 1201)
    phase = 2.0 * np.pi * np.mod(x, wavelength) / wavelength
    window = (0.5 * (1.0 + np.cos(phase))) ** sharpness
    surface = np.sum(
        -slope_contrast
        * coefficients[:, None]
        * wavelength
        / (2.0 * np.pi * harmonics[:, None])
        * np.sin(harmonics[:, None] * phase[None, :]),
        axis=0,
    )
    thickness = mean_thickness * (
        1.0 + thickness_contrast * (window_mean - window)
    )
    bed = surface - thickness
    xh = x / mean_thickness
    zh = surface / mean_thickness
    bh = bed / mean_thickness
    axis.fill_between(xh, bh, zh, color="#d8ecf4", zorder=0)
    axis.plot(xh, zh, color="#176d8c", lw=2.1)
    axis.plot(xh, bh, color="#5a4335", lw=2.1)
    axis.axvline(0.0, color="0.72", lw=0.8, ls=(0, (3, 3)))
    axis.annotate(
        "short, steep, thin\nicefall",
        xy=(0.0, 0.0),
        xytext=(0.72, -0.20),
        arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"),
        ha="center",
        va="top",
    )
    tongue_x = -0.37 * wavelength
    tongue_index = int(np.argmin(np.abs(x - tongue_x)))
    axis.annotate(
        "long, gentle, thick tongue",
        xy=(tongue_x / mean_thickness, surface[tongue_index] / mean_thickness),
        xytext=(-1.05, -0.67),
        arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"),
        ha="center",
        va="center",
    )
    axis.text(
        0.97,
        0.05,
        r"example: $L/H=4$, $r_s=12$, $r_H=3$",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.3",
    )
    axis.set_xlabel(r"$x/H$")
    axis.set_ylabel(r"elevation$/H$ (mean-slope axes)")
    axis.set_xlim(-2.05, 2.05)
    axis.set_aspect("equal", adjustable="box")
    panel_label(axis, "A", "Smooth periodic icefall-to-tongue geometry")


def draw_map(axis, table, wavelength_ratio, letter, norm, cmap):
    selected = table[
        np.isclose(table["wavelength / mean thickness"], wavelength_ratio)
    ].copy()
    valid = selected["status"].str.startswith(("success", "analytic"))
    selected["plotted minimum J"] = selected["minimum J"].where(valid)
    slope_ratios = np.sort(selected["steep / gentle surface-slope ratio"].unique())
    thickness_ratios = np.sort(selected["thick / thin thickness ratio"].unique())
    matrix = (
        selected.pivot(
            index="thick / thin thickness ratio",
            columns="steep / gentle surface-slope ratio",
            values="plotted minimum J",
        )
        .reindex(index=thickness_ratios, columns=slope_ratios)
        .to_numpy(dtype=float)
    )
    image = axis.pcolormesh(
        cell_edges(slope_ratios),
        cell_edges(thickness_ratios),
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    for iy, thickness_ratio in enumerate(thickness_ratios):
        for ix, slope_ratio in enumerate(slope_ratios):
            value = matrix[iy, ix]
            if np.isfinite(value):
                axis.text(
                    slope_ratio,
                    thickness_ratio,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if abs(value) > 0.72 else "0.12",
                )
            else:
                axis.text(slope_ratio, thickness_ratio, "x", ha="center", va="center", color="0.3")
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(mpl.ticker.FixedLocator(slope_ratios))
    axis.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axis.set_yticks(thickness_ratios)
    axis.set_xlim(cell_edges(slope_ratios)[[0, -1]])
    axis.set_ylim(cell_edges(thickness_ratios)[[0, -1]])
    axis.set_xlabel(r"steep/gentle slope ratio, $r_s$")
    axis.set_ylabel(r"thick/thin ratio, $r_H$")
    panel_label(axis, letter, rf"$L/H={wavelength_ratio:g}$")
    if np.isclose(wavelength_ratio, 8.0):
        axis.text(
            0.99,
            1.02,
            r"x: weak-signal failure; refined $J=0.995$",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="0.35",
        )
    return image


def draw_mechanism(axis, table):
    successful = table[
        table["status"].str.startswith(("success", "analytic"))
        & table["minimum J"].notna()
        & table["u / source u at minimum J"].notna()
        & table["mechanism C at minimum J"].notna()
    ].copy()
    wavelength_values = np.sort(successful["wavelength / mean thickness"].unique())
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 0.92, len(wavelength_values)))
    for color, wavelength_ratio in zip(colors, wavelength_values):
        selected = successful[np.isclose(successful["wavelength / mean thickness"], wavelength_ratio)]
        size = 18.0 + 12.0 * selected["thick / thin thickness ratio"].to_numpy(dtype=float)
        axis.scatter(
            selected["u / source u at minimum J"],
            selected["mechanism C at minimum J"],
            s=size,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
            label=rf"$L/H={wavelength_ratio:g}$",
        )
    axis.axhline(-1.0, color="#9b2d30", lw=1.1, ls=(0, (5, 3)))
    axis.text(
        0.02,
        -0.98,
        r"fold threshold $C=-1$",
        transform=axis.get_yaxis_transform(),
        color="#9b2d30",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    axis.axhline(0.0, color="0.55", lw=0.8)
    axis.axvline(1.0, color="0.75", lw=0.8)
    axis.set_xlabel(r"speed factor at minimum $J$, $u/u_0$")
    axis.set_ylabel(r"integrated adverse-shear factor, $C$")
    axis.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.10),
    )
    axis.text(
        0.97,
        0.96,
        r"$J=(u/u_0)(1+C)$",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )
    panel_label(axis, "F", "Why the tangent compresses (or folds)")


def make_figure(table, output):
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 150,
        }
    )
    figure = plt.figure(figsize=(11.2, 12.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(0.85, 1.0, 1.0))
    axes = [figure.add_subplot(grid[row, column]) for row in range(3) for column in range(2)]
    draw_geometry(axes[0])
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cmap = mpl.colormaps["RdBu"].copy()
    cmap.set_bad("0.86")
    image = None
    for axis, ratio, letter in zip(axes[1:5], (2.0, 4.0, 8.0, 12.0), "BCDE"):
        image = draw_map(axis, table, ratio, letter, norm, cmap)
    draw_mechanism(axes[5], table)
    colorbar = figure.colorbar(
        image,
        ax=axes[1:5],
        orientation="horizontal",
        shrink=0.82,
        aspect=35,
        pad=0.06,
        extend="both",
    )
    colorbar.set_label(r"minimum steady $J$ (white = fold threshold)")
    figure.suptitle(
        "A smooth icefall-to-tongue transition: direct steady overturn diagnostic",
        fontsize=14,
        y=1.015,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repository / "data" / "steady_characteristic_icefall_transition.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "figures" / "steady_characteristic_icefall_transition.png",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    table = pd.read_csv(args.input)
    make_figure(table, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
