"""Map overturning across a two-dimensional slice of elliptical troughs.

The full symmetric finite-reach ellipse has four geometric parameters after
translations: center and boundary thickness, half-domain length, and one
additional horizontal-curvature scale.  Here the last two are fixed at
L = 128.3 m and horizontal diameter D = 350 m.  The varied parameters are
therefore the ice thickness at the open boundaries and at the trough center.
The production sweep uses a regular 10 m grid, H_b = 10--260 m and
H_c = 60--320 m, so diagonal structure is resolved rather than inferred
from a few cells.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
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
BOUNDARY_THICKNESSES = np.arange(10.0, 261.0, 10.0)
CENTER_THICKNESSES = np.arange(60.0, 321.0, 10.0)


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


def case_inputs(
    boundary_thickness,
    center_thickness,
    resolution,
    maximum_age,
    time_step,
    source_count,
):
    """Return the inputs recorded for every valid or invalid grid cell."""
    return {
        "boundary thickness input (m)": boundary_thickness,
        "center thickness input (m)": center_thickness,
        "fixed horizontal radius (m)": HORIZONTAL_RADIUS,
        "fixed half-domain length (m)": REFERENCE_HALF_DOMAIN,
        "input target spacing (m)": resolution,
        "input maximum age (yr)": maximum_age,
        "input time step (yr)": time_step,
        "input source count": source_count,
    }


def solve_parameter_case(task):
    """Solve one cell; kept at module scope for spawn-based parallelism."""
    (
        boundary_thickness,
        center_thickness,
        resolution,
        maximum_age,
        time_step,
        source_count,
    ) = task
    inputs = case_inputs(*task)
    geometry = geometry_from_thicknesses(boundary_thickness, center_thickness)
    inputs |= {
        "vertical radius (m)": geometry["vertical radius"],
        "vertical center height (m)": geometry["center height"],
    }
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
        return inputs | {"status": "success"} | result
    except Exception as error:
        return inputs | {
            "status": f"failed: {type(error).__name__}: {error}",
            "minimum J": np.nan,
        }


def add_derived_diagnostics(table):
    """Add geometry and transit-scale diagnostics used to interpret the map."""
    table = table.copy()
    boundary = table["boundary thickness input (m)"].astype(float)
    center = table["center thickness input (m)"].astype(float)
    relief = center - boundary
    table["bed relief H_center - H_boundary (m)"] = relief
    table["center / boundary thickness"] = center / boundary
    shape_at_boundary = np.sqrt(
        1.0 - (REFERENCE_HALF_DOMAIN / HORIZONTAL_RADIUS) ** 2
    )
    vertical_radius = relief / (1.0 - shape_at_boundary)
    table["maximum absolute bed slope"] = (
        vertical_radius
        * REFERENCE_HALF_DOMAIN
        / HORIZONTAL_RADIUS**2
        / shape_at_boundary
    )
    table["center bed curvature (1/m)"] = (
        vertical_radius / HORIZONTAL_RADIUS**2
    )
    if {
        "maximum grid u (m/yr)",
        "fixed-interior evaluation end age (yr)",
    }.issubset(table.columns):
        table["kinematic exposure Umax*T/(2L)"] = (
            table["maximum grid u (m/yr)"]
            * table["fixed-interior evaluation end age (yr)"]
            / (2.0 * REFERENCE_HALF_DOMAIN)
        )
        table["empirical strain exposure Smax*Umax*T/(2L)"] = (
            table["maximum absolute bed slope"]
            * table["kinematic exposure Umax*T/(2L)"]
        )
    if "minimum grid u (m/yr)" in table:
        table["contains counterflow"] = table["minimum grid u (m/yr)"] < 0.0
    return table


def run_grid(
    resolution,
    maximum_age,
    time_step,
    source_count,
    jobs=1,
    existing=None,
    checkpoint_path=None,
    checkpoint_every=5,
):
    """Run the regular grid, optionally reusing exactly matching old cells."""
    rows = []
    reusable = {}
    if existing is not None and len(existing):
        for _, row in existing.iterrows():
            key = (
                float(row["boundary thickness input (m)"]),
                float(row["center thickness input (m)"]),
                float(row["input target spacing (m)"]),
                float(row["input maximum age (yr)"]),
                float(row["input time step (yr)"]),
                int(row["input source count"]),
            )
            reusable[key] = row.to_dict()

    pending = []
    for center_thickness in CENTER_THICKNESSES:
        for boundary_thickness in BOUNDARY_THICKNESSES:
            task = (
                float(boundary_thickness),
                float(center_thickness),
                float(resolution),
                float(maximum_age),
                float(time_step),
                int(source_count),
            )
            inputs = case_inputs(*task)
            if center_thickness <= boundary_thickness:
                rows.append(
                    inputs
                    | {
                        "status": "invalid: center thickness does not exceed boundary thickness",
                        "minimum J": np.nan,
                    }
                )
            elif task in reusable and reusable[task].get("status") == "success":
                rows.append(reusable[task])
            else:
                pending.append(task)

    print(
        f"Grid has {len(rows) + len(pending)} cells: "
        f"{len(pending)} solves pending, "
        f"{sum(row.get('status') == 'success' for row in rows)} reused.",
        flush=True,
    )
    executor = None
    if jobs == 1:
        solved = map(solve_parameter_case, pending)
    else:
        context = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=jobs, mp_context=context)
        solved = executor.map(solve_parameter_case, pending)
    try:
        for index, row in enumerate(solved, start=1):
            rows.append(row)
            print(
                json.dumps(
                    {
                        "completed": f"{index}/{len(pending)}",
                        "H_boundary": row["boundary thickness input (m)"],
                        "H_center": row["center thickness input (m)"],
                        "status": row["status"],
                        "minimum fixed-interior J": row.get(
                            "minimum fixed-interior J"
                        ),
                    }
                ),
                flush=True,
            )
            if (
                checkpoint_path is not None
                and index % checkpoint_every == 0
            ):
                checkpoint = add_derived_diagnostics(
                    pd.DataFrame(rows).sort_values(
                        [
                            "center thickness input (m)",
                            "boundary thickness input (m)",
                        ]
                    )
                )
                checkpoint.to_csv(
                    checkpoint_path, index=False, na_rep="NA"
                )
    finally:
        if executor is not None:
            executor.shutdown()
    return add_derived_diagnostics(
        pd.DataFrame(rows).sort_values(
            [
                "center thickness input (m)",
                "boundary thickness input (m)",
            ]
        )
    )


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
    """Plot the age-conditioned map and its strain-exposure collapse."""
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
    figure, (schematic_axis, map_axis, exposure_axis) = plt.subplots(
        1,
        3,
        figsize=(16.8, 5.7),
        gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]},
        constrained_layout=True,
    )
    draw_geometry_schematic(schematic_axis)
    invalid = ~success
    no_deposited_cohort = success & ~finite_success
    map_axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_where(~invalid, np.ones_like(array)),
        cmap=ListedColormap(["0.88"]),
        shading="flat",
        rasterized=True,
    )
    map_axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_where(
            ~no_deposited_cohort, np.ones_like(array)
        ),
        cmap=ListedColormap(["#f1e5b8"]),
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
        0.62,
        0.41,
        "geometrically\ninvalid",
        color="0.25",
        ha="center",
        va="center",
        rotation=43,
        fontsize=9,
    )
    map_axis.text(
        0.57,
        0.86,
        "no modeled surface\naccumulation\n(no deposited cohort)",
        color="0.28",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox={
            "facecolor": "#f1e5b8",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 1.5,
        },
    )
    colorbar = figure.colorbar(image, ax=map_axis, shrink=0.93)
    colorbar.set_label(
        r"Minimum fixed-interior $J=\partial x/\partial x_0$"
        "\n(to 4 kyr or first interior exit)"
    )
    map_axis.set(
        xlabel=r"Boundary (minimum) thickness, $H_b/D$",
        ylabel=r"Center (maximum) thickness, $H_c/D$",
        xlim=(x_edges[0], x_edges[-1]),
        ylim=(y_edges[0], y_edges[-1]),
        title=r"B   Age/exit-conditioned minimum $J$",
    )

    finite_table = table.loc[
        (table["status"] == "success")
        & np.isfinite(table["minimum fixed-interior J"])
    ].copy()
    exposure = finite_table[
        "empirical strain exposure Smax*Umax*T/(2L)"
    ].to_numpy(dtype=float)
    minimum_j = finite_table["minimum fixed-interior J"].to_numpy(
        dtype=float
    )
    groups = (
        (
            finite_table["fixed-interior censoring"] == "first interior exit",
            "#777777",
            "o",
            "first interior exit",
        ),
        (
            finite_table["fixed-interior censoring"] == "age ceiling",
            "#4393c3",
            "s",
            "4 kyr ceiling",
        ),
        (
            finite_table["fixed-interior censoring"] == "J<0 observed",
            "#b2182b",
            "o",
            r"$J<0$ observed",
        ),
    )
    for select, color, marker, label in groups:
        select = np.asarray(select)
        exposure_axis.scatter(
            exposure[select],
            minimum_j[select],
            s=17,
            marker=marker,
            facecolor=color,
            edgecolor="none",
            alpha=0.68,
            label=label,
            rasterized=True,
        )
    correlation = np.corrcoef(exposure, minimum_j)[0, 1]
    exposure_axis.axhline(0.0, color="0.45", linewidth=0.8, zorder=0)
    exposure_axis.text(
        0.96,
        0.95,
        rf"$r={correlation:.2f}$"
        "\n(descriptive; fixed geometry slice)",
        transform=exposure_axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    exposure_axis.set(
        xlabel=(
            "Empirical strain-exposure proxy,\n"
            r"$\Pi=s_{\max}U_{\max}T_{\rm eval}/(2L)$"
        ),
        ylabel=r"Minimum fixed-interior $J$",
        xlim=(0.0, max(3.5, 1.03 * np.nanmax(exposure))),
        ylim=(-1.08, 1.08),
        title="C   Accumulated strain exposure",
    )
    exposure_axis.legend(
        loc="lower left",
        frameon=False,
        fontsize=8.5,
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    figure.savefig(path, dpi=240)
    if str(path).lower().endswith(".png"):
        figure.savefig(str(path)[:-4] + ".pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--maximum-age", type=float, default=4_000.0)
    parser.add_argument("--time-step", type=float, default=1.0)
    parser.add_argument("--source-count", type=int, default=401)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Independent spawn-based worker processes (keep OMP_NUM_THREADS=1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful cells with exactly matching numerical inputs",
    )
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
        table = add_derived_diagnostics(pd.read_csv(csv_path))
        table.to_csv(csv_path, index=False, na_rep="NA")
    else:
        existing = (
            pd.read_csv(csv_path)
            if args.resume and csv_path.exists()
            else None
        )
        table = run_grid(
            args.resolution,
            args.maximum_age,
            args.time_step,
            args.source_count,
            jobs=args.jobs,
            existing=existing,
            checkpoint_path=csv_path,
        )
        table.to_csv(csv_path, index=False, na_rep="NA")
    save_figure(table, figure_path)


if __name__ == "__main__":
    main()
