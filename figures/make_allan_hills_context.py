"""Plot the Allan Hills core locations and published age-depth reversals."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATA = ROOT / "data"

BASEMAP = FIGURES / "allan_hills_context_basemap.png"
LOCATOR = FIGURES / "antarctica_locator.png"
NATURE_DATA = DATA / "marks-peterson.s41586-025-10032-y.source-data-fig1.xlsx"
PNAS_DATA = DATA / "shackleton.pnas.2502681122.sd01.xlsx"

# UTM zone 57S bounds for the cached 10 m Sentinel-2 image, in metres.
MAP_BOUNDS = (507000.0, 1478000.0, 514000.0, 1485000.0)

# Published coordinates transformed from WGS84 to UTM zone 57S.
CORE_LOCATIONS = {
    "ALHIC1901": (509123.494, 1483063.682, "o"),
    "ALHIC1902": (509574.985, 1481612.303, "^"),
}

NORMAL_COLOR = "#2c7bb6"
REVERSAL_COLOR = "#d95f02"


def refresh_basemap() -> None:
    """Refresh the cached contrast-enhanced Sentinel-2 true-color image."""
    try:
        import rasterio
        from PIL import Image
        from rasterio.merge import merge
    except ImportError as exc:
        raise RuntimeError(
            "Refreshing the basemap requires rasterio and Pillow."
        ) from exc

    # Copernicus Sentinel-2 L2A, acquired 2025-12-30. The two adjacent
    # Element 84 Earth Search COGs are needed because the cores straddle an
    # MGRS tile boundary.
    scene = "S2B_57C{tile}_20251230_1_L2A"
    template = (
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
        "sentinel-s2-l2a-cogs/57/C/{tile}/2025/12/"
        + scene
        + "/{band}.tif"
    )

    channels = []
    for band in ("B04", "B03", "B02"):
        parts = []
        for tile in ("VQ", "WQ"):
            url = template.format(tile=tile, band=band)
            with rasterio.open(url) as source:
                part, _ = merge([source], bounds=MAP_BOUNDS, res=10)
            parts.append(part[0])

        channel = parts[0].copy()
        for part in parts[1:]:
            valid = part > 0
            channel[valid] = part[valid]
        channels.append(channel)

    rgb = np.dstack(channels).astype(float)
    rgb = np.clip(rgb / 12000.0, 0.0, 1.0) ** 0.8
    luminance = rgb.mean(axis=2, keepdims=True)
    rgb = np.clip(luminance + 1.15 * (rgb - luminance), 0.0, 1.0)
    Image.fromarray((255 * rgb).astype("uint8")).save(BASEMAP, optimize=True)


def read_nature_alhic1901() -> tuple[np.ndarray, np.ndarray]:
    """Read the ALHIC1901 points used in Nature Figure 1b."""
    workbook = openpyxl.load_workbook(NATURE_DATA, read_only=True, data_only=True)
    sheet = workbook["1b"]

    header_row = None
    for row in sheet.iter_rows():
        if row[0].value == "Depth":
            header_row = row[0].row
            break
    if header_row is None:
        raise ValueError("Could not find the age-depth header in Nature source data")

    values = []
    for row in sheet.iter_rows(min_row=header_row + 2, values_only=True):
        depth, age_ka = row[:2]
        if depth is None or age_ka is None:
            break
        values.append((float(depth), float(age_ka) / 1000.0))
    workbook.close()

    values.sort()
    depth, age = np.asarray(values).T
    return depth, age


def read_pnas_core(core_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read and average replicated depth-age observations for one core."""
    workbook = openpyxl.load_workbook(PNAS_DATA, read_only=True, data_only=True)
    sheet = workbook["Ar and Water isotopes"]

    by_depth: defaultdict[float, list[float]] = defaultdict(list)
    for row in sheet.iter_rows(min_row=3, values_only=True):
        core, depth, age = row[0], row[1], row[8]
        if core == core_name and isinstance(depth, (int, float)) and isinstance(age, (int, float)):
            by_depth[float(depth)].append(float(age))
    workbook.close()

    values = sorted((depth, float(np.mean(ages))) for depth, ages in by_depth.items())
    depth, age = np.asarray(values).T
    return depth, age


def plot_age_depth(ax: plt.Axes, depth: np.ndarray, age: np.ndarray, title: str) -> None:
    ax.plot(age, depth, color=NORMAL_COLOR, linewidth=1.05, zorder=1)
    for index in range(len(depth) - 1):
        if age[index + 1] < age[index]:
            ax.plot(
                age[index : index + 2],
                depth[index : index + 2],
                color=REVERSAL_COLOR,
                linewidth=1.8,
                solid_capstyle="round",
                zorder=2,
            )
    ax.scatter(age, depth, s=9, facecolor="white", edgecolor="black", linewidth=0.45, zorder=3)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("$^{40}$Ar$_{\mathrm{atm}}$ age (Ma)")
    ax.set_ylabel("Core depth (m)")
    ax.grid(color="0.88", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=2.5)


def make_figure() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "savefig.dpi": 300,
        }
    )

    depth_1901, age_1901 = read_nature_alhic1901()
    depth_1902, age_1902 = read_pnas_core("ALHIC1902")

    figure = plt.figure(figsize=(7.2, 4.35), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.28, 0.92),
        left=0.075,
        right=0.985,
        bottom=0.12,
        top=0.95,
        wspace=0.28,
        hspace=0.42,
    )

    map_ax = figure.add_subplot(grid[:, 0])
    west, south, east, north = (bound / 1000.0 for bound in MAP_BOUNDS)
    image_extent = (west, east, south, north)
    map_ax.imshow(mpimg.imread(BASEMAP), extent=image_extent, origin="upper")

    label_effect = [path_effects.withStroke(linewidth=2.2, foreground="white")]
    for name, (easting, northing, marker) in CORE_LOCATIONS.items():
        x, y = easting / 1000.0, northing / 1000.0
        map_ax.scatter(
            x,
            y,
            marker=marker,
            s=58,
            facecolor="#f4c542",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )
        offset = (0.12, 0.16) if name == "ALHIC1901" else (0.12, -0.22)
        label = map_ax.text(x + offset[0], y + offset[1], name, weight="bold", zorder=5)
        label.set_path_effects(label_effect)

    # A one-kilometre scale bar and north arrow.
    scale_y = 1478.32
    map_ax.plot([507.35, 508.35], [scale_y, scale_y], color="black", linewidth=2.2)
    map_ax.text(507.85, scale_y + 0.11, "1 km", ha="center", va="bottom", path_effects=label_effect)
    map_ax.annotate(
        "N",
        xy=(513.45, 1484.55),
        xytext=(513.45, 1483.88),
        ha="center",
        va="bottom",
        weight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 0.9},
    )

    locator_ax = inset_axes(map_ax, width="25%", height="25%", loc="upper left", borderpad=1.05)
    locator_ax.imshow(mpimg.imread(LOCATOR))
    locator_ax.set_xticks([])
    locator_ax.set_yticks([])
    locator_ax.set_facecolor((1, 1, 1, 0.82))
    for spine in locator_ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.6)
    locator_ax.text(0.03, 0.97, "Antarctica", transform=locator_ax.transAxes, va="top", fontsize=6.5)

    map_ax.set_xlim(west, east)
    map_ax.set_ylim(south, north)
    map_ax.set_aspect("equal")
    map_ax.set_title("(a) Allan Hills blue-ice area", loc="left", pad=4)
    map_ax.set_xlabel("UTM 57S easting (km)")
    map_ax.set_ylabel("UTM 57S northing (km)")
    map_ax.tick_params(direction="out", length=2.5)

    age_1901_ax = figure.add_subplot(grid[0, 1])
    plot_age_depth(age_1901_ax, depth_1901, age_1901, "(b) ALHIC1901")
    age_1901_ax.set_xlim(0.25, 4.15)

    age_1902_ax = figure.add_subplot(grid[1, 1])
    plot_age_depth(age_1902_ax, depth_1902, age_1902, "(c) ALHIC1902")
    age_1902_ax.set_xlim(0.25, 6.25)

    legend = [
        Line2D([0], [0], color=NORMAL_COLOR, linewidth=1.4, label="age increases with depth"),
        Line2D([0], [0], color=REVERSAL_COLOR, linewidth=1.8, label="age decreases with depth"),
    ]
    age_1902_ax.legend(handles=legend, loc="upper right", frameon=False, handlelength=2.5)

    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"allan_hills_context.{suffix}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-basemap",
        action="store_true",
        help="download the Sentinel-2 bands and refresh the cached local map",
    )
    args = parser.parse_args()
    if args.refresh_basemap:
        refresh_basemap()
    make_figure()


if __name__ == "__main__":
    main()
