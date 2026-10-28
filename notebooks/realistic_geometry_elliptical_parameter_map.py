"""Map overturning across a two-dimensional slice of elliptical troughs.

The full symmetric finite-reach ellipse has four geometric parameters after
translations: center and boundary thickness, half-domain length, and one
additional horizontal-curvature scale.  Here the last two are fixed at
L = 128.3 m and horizontal diameter D = 350 m.  The varied parameters are
therefore the ice thickness at the open boundaries and at the trough center.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

from realistic_geometry_elliptical_family import flatten_result, run_case


HORIZONTAL_RADIUS = 175.0  # m
REFERENCE_VERTICAL_RADIUS = 250.0  # m
REFERENCE_CENTER_HEIGHT = 100.0  # m
REFERENCE_CUT_FRACTION = 0.80
REFERENCE_HALF_DOMAIN = (
    REFERENCE_CUT_FRACTION
    * HORIZONTAL_RADIUS
    * np.sqrt(
        1.0
        - (REFERENCE_CENTER_HEIGHT / REFERENCE_VERTICAL_RADIUS) ** 2
    )
)
BOUNDARY_THICKNESSES = np.array(
    [20.0, 40.0, 60.0, 70.0, 80.0, 100.0, 120.0, 140.0]
)
CENTER_THICKNESSES = np.array(
    [80.0, 100.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0]
)


def geometry_from_thicknesses(boundary_thickness, center_thickness):
    """Construct the unique ellipse in the fixed-horizontal-scale slice."""
    if center_thickness <= boundary_thickness:
        raise ValueError("center thickness must exceed boundary thickness")
    shape_at_boundary = np.sqrt(
        1.0 - (REFERENCE_HALF_DOMAIN / HORIZONTAL_RADIUS) ** 2
    )
    vertical_radius = (center_thickness - boundary_thickness) / (
        1.0 - shape_at_boundary
    )
    center_height = vertical_radius - center_thickness
    return {
        "name": (
            f"symmetric ellipse, H_boundary={boundary_thickness:g} m, "
            f"H_center={center_thickness:g} m"
        ),
        "vertical radius": vertical_radius,
        "center height": center_height,
        "left radius": HORIZONTAL_RADIUS,
        "right radius": HORIZONTAL_RADIUS,
        "half domain": REFERENCE_HALF_DOMAIN,
    }


def run_grid(resolution, maximum_age, time_step, source_count):
    rows = []
    for center_thickness in CENTER_THICKNESSES:
        for boundary_thickness in BOUNDARY_THICKNESSES:
            inputs = {
                "boundary thickness input (m)": boundary_thickness,
                "center thickness input (m)": center_thickness,
                "fixed horizontal radius (m)": HORIZONTAL_RADIUS,
                "fixed half-domain length (m)": REFERENCE_HALF_DOMAIN,
                "input target spacing (m)": resolution,
                "input maximum age (yr)": maximum_age,
                "input time step (yr)": time_step,
                "input source count": source_count,
            }
            if center_thickness <= boundary_thickness:
                rows.append(
                    inputs
                    | {
                        "status": "invalid: center thickness does not exceed boundary thickness",
                        "minimum J": np.nan,
                    }
                )
                continue
            geometry = geometry_from_thicknesses(
                boundary_thickness, center_thickness
            )
            inputs |= {
                "vertical radius (m)": geometry["vertical radius"],
                "vertical center height (m)": geometry["center height"],
            }
            print(
                f"Solving H_boundary={boundary_thickness:g} m, "
                f"H_center={center_thickness:g} m",
                flush=True,
            )
            try:
                result = flatten_result(
                    run_case(
                        geometry,
                        resolution,
                        maximum_age=maximum_age,
                        time_step=time_step,
                        source_count=source_count,
                    )
                )
                row = inputs | {"status": "success"} | result
                print(
                    json.dumps(
                        {
                            "H_boundary": boundary_thickness,
                            "H_center": center_thickness,
                        "minimum J": row["minimum J"],
                        "minimum fixed-interior J": row[
                            "minimum fixed-interior J"
                        ],
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                row = inputs | {
                    "status": f"failed: {type(error).__name__}: {error}",
                    "minimum J": np.nan,
                }
                print(row["status"], flush=True)
            rows.append(row)
    return pd.DataFrame(rows)


def cell_edges(values):
    """Return pcolormesh edges for an increasing vector of cell centers."""
    values = np.asarray(values, dtype=float)
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.r_[
        values[0] - (midpoints[0] - values[0]),
        midpoints,
        values[-1] + (values[-1] - midpoints[-1]),
    ]


def draw_geometry_schematic(axis):
    """Define the four geometric quantities used by this parameter slice."""
    diameter = 2.0 * HORIZONTAL_RADIUS
    half_domain = REFERENCE_HALF_DOMAIN / diameter
    half_horizontal_diameter = 0.5

    # An illustrative member of the family.  The annotations, rather than
    # these particular thicknesses, define the parameterization.
    boundary_thickness = 0.16
    center_thickness = 0.43
    boundary_shape = np.sqrt(
        1.0 - (half_domain / half_horizontal_diameter) ** 2
    )
    vertical_radius = (
        center_thickness - boundary_thickness
    ) / (1.0 - boundary_shape)
    center_height = vertical_radius - center_thickness

    x_full = np.linspace(
        -half_horizontal_diameter, half_horizontal_diameter, 601
    )
    bed_full = center_height - vertical_radius * np.sqrt(
        np.maximum(
            0.0, 1.0 - (x_full / half_horizontal_diameter) ** 2
        )
    )
    x_domain = np.linspace(-half_domain, half_domain, 401)
    bed_domain = center_height - vertical_radius * np.sqrt(
        1.0 - (x_domain / half_horizontal_diameter) ** 2
    )

    # The pale continuation shows the underlying ellipse; only the dark
    # segment between +/-L is part of the modeled glacier.
    axis.plot(x_full, bed_full, color="0.78", linewidth=1.1, zorder=1)
    axis.fill_between(
        x_domain,
        bed_domain,
        0.0,
        color="#d9edf7",
        alpha=0.9,
        zorder=2,
    )
    axis.plot(x_domain, bed_domain, color="0.15", linewidth=1.8, zorder=3)
    axis.plot(
        [-half_domain, half_domain],
        [0.0, 0.0],
        color="#3b78a3",
        linewidth=1.8,
        zorder=4,
    )
    for side in (-half_domain, half_domain):
        bed_side = center_height - vertical_radius * np.sqrt(
            1.0 - (side / half_horizontal_diameter) ** 2
        )
        axis.plot(
            [side, side],
            [bed_side, 0.0],
            color="0.35",
            linewidth=1.0,
            zorder=4,
        )

    arrow = dict(arrowstyle="<->", color="0.15", shrinkA=0, shrinkB=0)
    axis.annotate(
        "",
        xy=(0.0, -center_thickness),
        xytext=(0.0, 0.0),
        arrowprops=arrow,
        zorder=5,
    )
    axis.text(
        0.018,
        -0.5 * center_thickness,
        r"$H_c$",
        ha="left",
        va="center",
    )
    axis.annotate(
        "",
        xy=(-half_domain, -boundary_thickness),
        xytext=(-half_domain, 0.0),
        arrowprops=arrow,
        zorder=5,
    )
    axis.text(
        -half_domain - 0.018,
        -0.5 * boundary_thickness,
        r"$H_b$",
        ha="right",
        va="center",
    )
    axis.annotate(
        "",
        xy=(half_domain, 0.075),
        xytext=(0.0, 0.075),
        arrowprops=arrow,
    )
    axis.text(
        0.5 * half_domain,
        0.092,
        rf"$L={REFERENCE_HALF_DOMAIN:.1f}\,$m (fixed)",
        ha="center",
        va="bottom",
    )
    diameter_arrow_height = -0.55
    axis.annotate(
        "",
        xy=(half_horizontal_diameter, diameter_arrow_height),
        xytext=(-half_horizontal_diameter, diameter_arrow_height),
        arrowprops=arrow,
    )
    axis.text(
        0.0,
        diameter_arrow_height - 0.025,
        rf"$D=2a={diameter:.0f}\,$m (fixed)",
        ha="center",
        va="top",
    )

    axis.text(
        -half_domain + 0.015,
        0.012,
        "surface",
        color="#285f85",
        ha="left",
        va="bottom",
    )
    axis.text(
        0.23,
        -0.36,
        "elliptical bed",
        color="0.25",
        rotation=34,
        ha="left",
        va="center",
    )
    axis.text(
        0.0,
        -0.665,
        r"$H_c=-z_b(0),\quad H_b=-z_b(\pm L),\quad$"
        r"$z_b=c-b\sqrt{1-(x/a)^2}$",
        ha="center",
        va="top",
        fontsize=9,
    )
    axis.set(
        xlim=(-0.58, 0.58),
        ylim=(-0.74, 0.16),
        title="A   Geometry and fixed horizontal scales",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def save_figure(table, path):
    """Plot minimum J with zero exactly at the white color-map midpoint."""
    status = table.pivot(
        index="center thickness input (m)",
        columns="boundary thickness input (m)",
        values="status",
    ).reindex(index=CENTER_THICKNESSES, columns=BOUNDARY_THICKNESSES)
    values = table.pivot(
        index="center thickness input (m)",
        columns="boundary thickness input (m)",
        values="minimum fixed-interior J",
    ).reindex(index=CENTER_THICKNESSES, columns=BOUNDARY_THICKNESSES)
    array = values.to_numpy(dtype=float)
    success = status.to_numpy() == "success"
    finite_success = success & np.isfinite(array)
    if not np.any(finite_success):
        raise RuntimeError("parameter map contains no successful finite values")

    diameter = 2.0 * HORIZONTAL_RADIUS
    normalized_boundary_thicknesses = BOUNDARY_THICKNESSES / diameter
    normalized_center_thicknesses = CENTER_THICKNESSES / diameter
    x_edges = cell_edges(normalized_boundary_thicknesses)
    y_edges = cell_edges(normalized_center_thicknesses)
    limit = max(0.25, float(np.nanmax(np.abs(array[finite_success]))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    diverging = LinearSegmentedColormap.from_list(
        "negative_red_zero_white_positive_blue",
        ["#67001f", "#d6604d", "#ffffff", "#4393c3", "#053061"],
        N=257,
    )
    figure, (schematic_axis, map_axis) = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
        constrained_layout=True,
    )
    draw_geometry_schematic(schematic_axis)
    map_axis.pcolormesh(
        x_edges,
        y_edges,
        np.ones_like(array),
        cmap=ListedColormap(["0.88"]),
        shading="flat",
        rasterized=True,
    )
    image = map_axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_where(~finite_success, array),
        cmap=diverging,
        norm=norm,
        shading="flat",
        rasterized=True,
    )

    map_axis.text(
        0.345,
        0.275,
        "geometrically\ninvalid",
        color="0.25",
        ha="center",
        va="center",
        rotation=37,
        fontsize=9,
    )
    colorbar = figure.colorbar(image, ax=map_axis)
    colorbar.set_label(
        r"Minimum fixed-interior $J=\partial x/\partial x_0$"
        "\n(to 4 kyr or first interior exit)"
    )
    map_axis.set(
        xlabel=r"Boundary (minimum) thickness, $H_b/D$",
        ylabel=r"Center (maximum) thickness, $H_c/D$",
        xlim=(x_edges[0], x_edges[-1]),
        ylim=(y_edges[0], y_edges[-1]),
        title=r"B   Minimum fixed-interior $J$",
    )
    figure.savefig(path, dpi=240)
    if str(path).lower().endswith(".png"):
        figure.savefig(str(path)[:-4] + ".pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--maximum-age", type=float, default=4_000.0)
    parser.add_argument("--time-step", type=float, default=1.0)
    parser.add_argument("--source-count", type=int, default=401)
    parser.add_argument("--csv")
    parser.add_argument("--figure")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read --csv and only regenerate the figure",
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv) if args.csv else (
        repository / "data" / "realistic_geometry_elliptical_parameter_map.csv"
    )
    figure_path = Path(args.figure) if args.figure else (
        repository / "figures" / "realistic_geometry_elliptical_parameter_map.png"
    )
    if args.plot_only:
        table = pd.read_csv(csv_path)
    else:
        table = run_grid(
            args.resolution,
            args.maximum_age,
            args.time_step,
            args.source_count,
        )
        table.to_csv(csv_path, index=False, na_rep="NA")
    save_figure(table, figure_path)


if __name__ == "__main__":
    main()
