"""Plot the smooth localized-riegel pilot as one PNG summary."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from steady_characteristic_localized_packet_parameter_sweep import (
    LocalizedPacketGeometry,
)


def zero_white_colormap():
    return LinearSegmentedColormap.from_list(
        "negative-red_zero-white_positive-blue",
        ("#8b0a2a", "#d96b55", "#ffffff", "#73add1", "#0d4f7c"),
    )


def annotate_heatmap(ax, values, invalid=None, special=None):
    invalid = np.zeros_like(values, dtype=bool) if invalid is None else invalid
    for iy in range(values.shape[0]):
        for ix in range(values.shape[1]):
            value = values[iy, ix]
            if not np.isfinite(value):
                continue
            label = f"{value:.2f}" if abs(value) < 10.0 else ("<-10" if value < 0 else ">10")
            if invalid[iy, ix]:
                label += "*"
            if special is not None and (iy, ix) in special:
                label += f"\n[{special[(iy, ix)]:.2f}]"
            color = "white" if abs(np.clip(value, -1, 1)) > 0.65 else "#222222"
            ax.text(ix, iy, label, ha="center", va="center", fontsize=7.4, color=color)


def geometry_panel(ax):
    geometry = LocalizedPacketGeometry(
        wavelength=4.0,
        amplitude=0.90,
        mean_thickness=1.0,
        feature_width=1.0 / 12.0,
        feature_phase=0.5 * np.pi,
        feature_polarity=1,
    )
    x = np.linspace(0.0, 4.0, 6000, endpoint=False)
    bed = geometry.bed_height(x)
    ax.fill_between(x, bed, 0.0, color="#dcecf4", zorder=0)
    ax.plot(x, bed, color="#222222", lw=1.5)
    ax.axhline(0.0, color="#307da5", lw=1.3)
    ax.axhline(-1.0, color="#777777", lw=0.8, ls="--")

    peak_index = int(np.argmax(bed))
    peak_x, peak_z = x[peak_index], bed[peak_index]
    ax.annotate(
        "", xy=(peak_x, 0.0), xytext=(peak_x, peak_z),
        arrowprops=dict(arrowstyle="<->", color="#8b0a2a", lw=1.1),
    )
    ax.text(peak_x + 0.08, 0.5 * peak_z, r"$H_{\min}=0.10H$", color="#8b0a2a", va="center")
    ax.annotate(
        "", xy=(3.55, 0.0), xytext=(3.55, -1.0),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0),
    )
    ax.text(3.62, -0.5, r"mean $H$", color="#555555", va="center")

    half = 0.5 * geometry.feature_width
    arrow_y = peak_z - 0.23
    ax.annotate(
        "", xy=(1.0 + half, arrow_y), xytext=(1.0 - half, arrow_y),
        arrowprops=dict(arrowstyle="|-|", color="#8b0a2a", lw=1.2),
    )
    ax.text(1.0, arrow_y - 0.09, r"kernel FWHM $\ell=H/12$", color="#8b0a2a", ha="center", va="top")
    ax.annotate(
        "", xy=(4.0, 0.12), xytext=(0.0, 0.12),
        arrowprops=dict(arrowstyle="<->", color="#222222", lw=1.0),
    )
    ax.text(2.0, 0.16, r"broad period $L=4H$", ha="center", va="bottom")
    ax.annotate(
        "flow", xy=(3.10, -0.18), xytext=(2.55, -0.18),
        arrowprops=dict(arrowstyle="->", color="#307da5", lw=1.3),
        color="#307da5", va="center",
    )
    ax.text(
        0.04, -2.05,
        r"analytic bed, but nearly pinched and near-vertical: $\max|b_x|=17.2$"
        "\nrefined uniform mesh: $J_{\min}=-0.271$ after horizontal counterflow"
        "\nuniform-path audit: 14/138 sources (0.924% of retained flux) overturn",
        ha="left", va="bottom", fontsize=8.5,
    )
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(-2.14, 0.28)
    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$z/H$")
    ax.set_title(r"A  Geometry: broad trough + upstream-flank riegel", loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def refined_minimum(table):
    refined = table[
        table["localized-packet resolution"].astype(str).str.startswith("refined")
    ]
    if len(refined) != 1:
        raise ValueError(
            "expected exactly one refined row, found "
            f"{len(refined)} in focused refinement table"
        )
    return float(refined.iloc[0]["minimum J"])


def selected_panel(ax, table, focused_uniform, norm, cmap):
    amplitudes = np.sort(table["amplitude / mean thickness"].unique())
    periods = np.sort(table["wavelength / mean thickness"].unique())
    values = np.full((len(amplitudes), len(periods)), np.nan)
    invalid = np.zeros_like(values, dtype=bool)
    reversals = np.zeros_like(values, dtype=bool)
    for _, row in table.iterrows():
        iy = int(np.where(np.isclose(amplitudes, row["amplitude / mean thickness"]))[0][0])
        ix = int(np.where(np.isclose(periods, row["wavelength / mean thickness"]))[0][0])
        values[iy, ix] = row["minimum J"]
        invalid[iy, ix] = row["maximum relative streamfunction identity residual"] > 0.10
        reversals[iy, ix] = row["maximum velocity reversal count"] > 0
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    special = {
        (
            int(np.where(np.isclose(amplitudes, 0.9))[0][0]),
            int(np.where(np.isclose(periods, 4.0))[0][0]),
        ): refined_minimum(focused_uniform)
    }
    annotate_heatmap(ax, values, invalid=invalid, special=special)
    for iy, ix in zip(*np.where(reversals)):
        ax.add_patch(Rectangle((ix - 0.43, iy - 0.43), 0.86, 0.86, fill=False, ec="#111111", lw=1.4))
    ax.set_xticks(np.arange(len(periods)), [f"{value:g}" for value in periods])
    ax.set_yticks(np.arange(len(amplitudes)), [f"{value:g}" for value in amplitudes])
    ax.set_xlabel(r"broad period, $L/H$")
    ax.set_ylabel(r"amplitude, $A/H=1-H_{\min}/H$")
    ax.set_title(r"B  Minimum $J$: selected $\ell/H=1/12$, $\phi/(2\pi)=1/4$, riegel", loc="left")
    ax.text(
        0.0, -0.25,
        "square: horizontal-velocity reversal; [ ]: refined uniform mesh\n"
        "*: path-identity residual >10%",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
    )
    return image


def shape_panel(ax, table, focused_uniform, focused_width, norm, cmap):
    widths = np.sort(table["localized feature FWHM / mean thickness"].unique())
    phases = np.sort(table["localized feature phase / (2 pi)"].unique())
    columns = [(polarity, phase) for polarity in (-1, 1) for phase in phases]
    values = np.full((len(widths), len(columns)), np.nan)
    reversals = np.zeros_like(values, dtype=bool)
    for _, row in table.iterrows():
        iy = int(np.where(np.isclose(widths, row["localized feature FWHM / mean thickness"]))[0][0])
        ix = columns.index((int(row["localized feature polarity"]), row["localized feature phase / (2 pi)"]))
        values[iy, ix] = row["minimum J"]
        reversals[iy, ix] = row["maximum velocity reversal count"] > 0
    ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    special = {
        (
            int(np.where(np.isclose(widths, 1.0 / 12.0))[0][0]),
            columns.index((1, 0.25)),
        ): refined_minimum(focused_uniform),
        (
            int(np.where(np.isclose(widths, 1.0 / 8.0))[0][0]),
            columns.index((1, 0.25)),
        ): refined_minimum(focused_width),
    }
    annotate_heatmap(ax, values, special=special)
    for iy, ix in zip(*np.where(reversals)):
        ax.add_patch(Rectangle((ix - 0.43, iy - 0.43), 0.86, 0.86, fill=False, ec="#111111", lw=1.4))
    labels = []
    for polarity, phase in columns:
        symbol = "notch" if polarity < 0 else "riegel"
        phase_text = {0.0: "0", 0.25: "¼", 0.5: "½", 0.75: "¾"}[round(float(phase), 2)]
        labels.append(f"{symbol}\n{phase_text}")
    width_labels = ["1/12" if np.isclose(v, 1/12) else "1/8" if np.isclose(v, 1/8) else f"{v:g}" for v in widths]
    ax.set_xticks(np.arange(len(columns)), labels, fontsize=7.5)
    ax.set_yticks(np.arange(len(widths)), width_labels)
    ax.set_xlabel(r"feature polarity and relative phase, $\phi/(2\pi)$")
    ax.set_ylabel(r"imposed-kernel FWHM, $\ell/H$")
    ax.set_title(r"C  Width, phase, and polarity at $A/H=0.90$, $L/H=4$", loc="left")
    ax.text(
        0.0, -0.25,
        "Only a narrow upstream-flank riegel reverses flow;\n"
        "bracketed cells are focused refinements.",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    repository = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--selected-input",
        type=Path,
        default=repository / "data" / "steady_characteristic_localized_packet_selected.csv",
    )
    parser.add_argument(
        "--shape-input",
        type=Path,
        default=repository / "data" / "steady_characteristic_localized_packet_shape_pilot.csv",
    )
    parser.add_argument(
        "--focused-uniform-input",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_localized_packet_focused_uniform.csv",
    )
    parser.add_argument(
        "--focused-width-input",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_localized_packet_focused_width_1_8.csv",
    )
    args = parser.parse_args()
    selected = pd.read_csv(args.selected_input)
    shape = pd.read_csv(args.shape_input)
    focused_uniform = pd.read_csv(args.focused_uniform_input)
    focused_width = pd.read_csv(args.focused_width_input)
    cmap = zero_white_colormap()
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5})
    figure = plt.figure(figsize=(15.5, 6.2), constrained_layout=False)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.055), hspace=0.90, wspace=0.34)
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    geometry_panel(axes[0])
    image = selected_panel(axes[1], selected, focused_uniform, norm, cmap)
    shape_panel(axes[2], shape, focused_uniform, focused_width, norm, cmap)
    color_axis = figure.add_subplot(grid[1, 1:])
    colorbar = figure.colorbar(image, cax=color_axis, orientation="horizontal", extend="both")
    colorbar.set_label(r"minimum steady $J$ (colors clipped outside $[-1,1]$)")
    figure.suptitle(
        "A smooth localized obstacle can fold isochrones only in an extreme counterflow regime",
        fontsize=14,
        y=0.995,
    )
    output = args.output or (
        repository / "figures" / "steady_characteristic_localized_packet_summary.png"
    )
    figure.savefig(output, dpi=240, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
