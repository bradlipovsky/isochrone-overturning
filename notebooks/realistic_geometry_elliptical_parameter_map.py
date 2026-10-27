"""Map overturning across a two-dimensional slice of elliptical troughs.

The full symmetric finite-reach ellipse has four geometric parameters after
translations: center and boundary thickness, half-domain length, and one
additional horizontal-curvature scale.  Here the last two are fixed at the
values of the existing narrow symmetric trough.  The varied parameters are
therefore the ice thickness at the open boundaries and at the trough center.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
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
    raw_values = table.pivot(
        index="center thickness input (m)",
        columns="boundary thickness input (m)",
        values="minimum J",
    ).reindex(index=CENTER_THICKNESSES, columns=BOUNDARY_THICKNESSES)
    array = values.to_numpy(dtype=float)
    raw_array = raw_values.to_numpy(dtype=float)
    success = status.to_numpy() == "success"
    finite_success = success & np.isfinite(array)
    if not np.any(finite_success):
        raise RuntimeError("parameter map contains no successful finite values")

    x_edges = cell_edges(BOUNDARY_THICKNESSES)
    y_edges = cell_edges(CENTER_THICKNESSES)
    limit = max(0.25, float(np.nanmax(np.abs(array[finite_success]))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    diverging = LinearSegmentedColormap.from_list(
        "negative_red_zero_white_positive_blue",
        ["#67001f", "#d6604d", "#ffffff", "#4393c3", "#053061"],
        N=257,
    )
    figure, axis = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    axis.pcolormesh(
        x_edges,
        y_edges,
        np.ones_like(array),
        cmap=ListedColormap(["0.88"]),
        shading="flat",
    )
    image = axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_where(~finite_success, array),
        cmap=diverging,
        norm=norm,
        shading="flat",
    )

    valid_rows = table[table["status"].str.startswith("success")]
    axis.scatter(
        valid_rows["boundary thickness input (m)"],
        valid_rows["center thickness input (m)"],
        s=9,
        color="0.15",
        linewidths=0,
        label="successful sample",
    )
    failed_rows = table[table["status"].str.startswith("failed")]
    if len(failed_rows):
        axis.scatter(
            failed_rows["boundary thickness input (m)"],
            failed_rows["center thickness input (m)"],
            s=28,
            marker="x",
            color="0.15",
            linewidths=1.0,
            label="failed sample",
        )
    axis.scatter(
        [70.0],
        [150.0],
        marker="*",
        s=150,
        facecolor="#ffd92f",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="existing narrow case",
    )
    x_grid, y_grid = np.meshgrid(
        BOUNDARY_THICKNESSES, CENTER_THICKNESSES
    )
    if np.nanmin(array[finite_success]) < 0.0 < np.nanmax(array[finite_success]):
        interior_contour = axis.contour(
            x_grid,
            y_grid,
            np.ma.masked_where(~finite_success, array),
            levels=[0.0],
            colors="0.1",
            linewidths=1.2,
        )
    raw_finite = success & np.isfinite(raw_array)
    if (
        np.any(raw_finite)
        and np.nanmin(raw_array[raw_finite]) < 0.0
        < np.nanmax(raw_array[raw_finite])
    ):
        raw_contour = axis.contour(
            x_grid,
            y_grid,
            np.ma.masked_where(~raw_finite, raw_array),
            levels=[0.0],
            colors="0.35",
            linewidths=0.9,
            linestyles="--",
        )
    colorbar = figure.colorbar(image, ax=axis, extend="both")
    colorbar.set_label(
        r"Minimum fixed-interior $J=\partial x/\partial x_0$"
        " through 4 kyr or first interior exit"
    )
    axis.set(
        xlabel="Ice thickness at each open boundary (m)",
        ylabel="Maximum ice thickness at trough center (m)",
        xlim=(x_edges[0], x_edges[-1]),
        ylim=(y_edges[0], y_edges[-1]),
        title=(
            "Symmetric elliptical trough: fixed horizontal geometry\n"
            rf"horizontal radius = {HORIZONTAL_RADIUS:.0f} m, "
            rf"half-domain = {REFERENCE_HALF_DOMAIN:.1f} m"
        ),
    )
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="0.1", linewidth=1.2),
            Line2D(
                [0],
                [0],
                color="0.35",
                linewidth=0.9,
                linestyle="--",
            ),
        ]
    )
    labels.extend(
        [
            r"fixed-interior $J=0$",
            r"raw lifetime $J=0$",
        ]
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            markersize=8,
            linestyle="none",
            markerfacecolor="0.88",
            markeredgecolor="0.6",
        )
    )
    labels.append("geometrically invalid / not sampled")
    axis.legend(handles, labels, frameon=False, loc="upper left")
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
