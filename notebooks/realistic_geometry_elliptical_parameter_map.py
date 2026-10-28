"""Map age-dependent overturning across elliptical trough geometries.

The full symmetric finite-reach ellipse has four geometric parameters after
translations: center and boundary thickness, half-domain length, and one
additional horizontal-curvature scale.  Here the last two are fixed at
L = 128.3 m and horizontal diameter D = 350 m.  The varied parameters are
therefore the ice thickness at the open boundaries and at the trough center.
The production sweep uses a regular 10 m grid, H_b = 10--260 m and
H_c = 60--320 m.  Each steady velocity field is integrated once to 8 kyr;
the fixed-interior minimum J is retained at 2 and 8 kyr, or until the cohort
first exits, together with the first zero-crossing age.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from realistic_geometry_elliptical_family import (
    fixed_interior_censoring_column,
    fixed_interior_end_age_column,
    fixed_interior_minimum_column,
    flatten_result,
    run_case,
)


plt.rcParams.update(
    {
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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
REPORT_AGES = (2_000.0, 8_000.0)


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
                fixed_interior_report_ages=tuple(
                    age for age in REPORT_AGES if age <= maximum_age
                ),
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
        report_ages = tuple(age for age in REPORT_AGES if age <= maximum_age)
        required_columns = {
            "first fixed-interior zero-crossing age (yr)",
            *(
                fixed_interior_minimum_column(age)
                for age in report_ages
            ),
            *(
                fixed_interior_end_age_column(age)
                for age in report_ages
            ),
            *(
                fixed_interior_censoring_column(age)
                for age in report_ages
            ),
        }
        if required_columns.issubset(existing.columns):
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
        else:
            print(
                "Existing table lacks the requested time-checkpoint columns; "
                "no rows will be reused.",
                flush=True,
            )

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
    axis.set(
        xlim=(-0.58, 0.58),
        ylim=(-0.64, 0.16),
    )
    axis.set_title("A   Geometry", loc="left")
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def save_figure(table, path):
    """Plot geometry, two time horizons, and first-overturn age."""
    minimum_columns = {
        age: fixed_interior_minimum_column(age) for age in REPORT_AGES
    }
    required_columns = {
        "status",
        "first fixed-interior zero-crossing age (yr)",
        fixed_interior_censoring_column(REPORT_AGES[-1]),
        *minimum_columns.values(),
    }
    missing = required_columns - set(table.columns)
    if missing:
        raise RuntimeError(
            "parameter map lacks required 2/8 kyr diagnostics: "
            + ", ".join(sorted(missing))
        )

    def pivot(column):
        return table.pivot(
            index="center thickness input (m)",
            columns="boundary thickness input (m)",
            values=column,
        ).reindex(index=CENTER_THICKNESSES, columns=BOUNDARY_THICKNESSES)

    status = table.pivot(
        index="center thickness input (m)",
        columns="boundary thickness input (m)",
        values="status",
    ).reindex(index=CENTER_THICKNESSES, columns=BOUNDARY_THICKNESSES)
    j_by_age = {
        age: pivot(column).to_numpy(dtype=float)
        for age, column in minimum_columns.items()
    }
    crossing_age = pivot(
        "first fixed-interior zero-crossing age (yr)"
    ).to_numpy(dtype=float)
    censoring = pivot(
        fixed_interior_censoring_column(REPORT_AGES[-1])
    ).to_numpy(dtype=str)
    status_array = status.fillna("").to_numpy(dtype=str)
    success = status_array == "success"
    geometric_invalid = np.char.startswith(status_array, "invalid:")
    solver_failure = np.char.startswith(status_array, "failed:")
    finite_cohort = success & np.isfinite(j_by_age[REPORT_AGES[-1]])
    no_deposited_cohort = success & ~finite_cohort
    crossed = (
        finite_cohort
        & np.isfinite(crossing_age)
        & (crossing_age <= REPORT_AGES[-1])
    )
    exited_without_crossing = (
        finite_cohort
        & ~crossed
        & (censoring == "first interior exit")
    )
    age_censored = (
        finite_cohort & ~crossed & (censoring == "age ceiling")
    )
    classified = (
        geometric_invalid
        | solver_failure
        | no_deposited_cohort
        | crossed
        | exited_without_crossing
        | age_censored
    )
    if not np.all(classified):
        unclassified = np.unique(status_array[~classified])
        raise RuntimeError(
            "unclassified parameter-map cells: " + ", ".join(unclassified)
        )
    if not np.any(finite_cohort):
        raise RuntimeError("parameter map contains no successful finite values")

    diameter = 2.0 * HORIZONTAL_RADIUS
    normalized_boundary_thicknesses = BOUNDARY_THICKNESSES / diameter
    normalized_center_thicknesses = CENTER_THICKNESSES / diameter
    x_edges = cell_edges(normalized_boundary_thicknesses)
    y_edges = cell_edges(normalized_center_thicknesses)
    norm = TwoSlopeNorm(vmin=-1.01, vcenter=0.0, vmax=1.01)
    diverging = LinearSegmentedColormap.from_list(
        "negative_red_zero_white_positive_blue",
        ["#67001f", "#d6604d", "#ffffff", "#4393c3", "#053061"],
        N=257,
    )
    category_colors = {
        "invalid": "#dedede",
        "no cohort": "#e7dcec",
        "exit": "#c9d5dc",
        "age": "#f7f7f7",
        "failure": "#efd0d0",
    }

    figure = plt.figure(figsize=(8.0, 6.6), constrained_layout=True)
    grid = figure.add_gridspec(
        4,
        3,
        width_ratios=(1.0, 1.0, 0.055),
        height_ratios=(1.0, 0.12, 1.0, 0.075),
    )
    schematic_axis = figure.add_subplot(grid[0, 0])
    crossing_axis = figure.add_subplot(grid[0, 1])
    time_colorbar_axis = figure.add_subplot(grid[0, 2])
    category_axis = figure.add_subplot(grid[1, :2])
    early_axis = figure.add_subplot(grid[2, 0])
    late_axis = figure.add_subplot(grid[2, 1])
    j_colorbar_axis = figure.add_subplot(grid[3, :2])
    draw_geometry_schematic(schematic_axis)

    def paint_mask(axis, mask, color):
        axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_where(~mask, np.ones(mask.shape)),
            cmap=ListedColormap([color]),
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            rasterized=True,
        )

    def paint_common_categories(axis):
        paint_mask(axis, geometric_invalid, category_colors["invalid"])
        paint_mask(axis, no_deposited_cohort, category_colors["no cohort"])
        if np.any(solver_failure):
            paint_mask(axis, solver_failure, category_colors["failure"])

    def format_map_axis(axis, title, show_ylabel=True):
        axis.set(
            xlabel=r"Boundary thickness, $H_b/D$",
            ylabel=r"Center thickness, $H_c/D$" if show_ylabel else None,
            xlim=(x_edges[0], x_edges[-1]),
            ylim=(y_edges[0], y_edges[-1]),
        )
        axis.set_title(title, loc="left")
        axis.set_aspect("equal", adjustable="box")
        if not show_ylabel:
            axis.tick_params(axis="y", labelleft=False)

    def label_common_regions(axis, label_no_cohort=False):
        axis.text(
            0.62,
            0.41,
            "geometrically\ninvalid",
            color="0.25",
            ha="center",
            va="center",
            rotation=43,
            fontsize=7.5,
        )
        if label_no_cohort:
            axis.text(
                0.52,
                0.87,
                "no deposited\ncohort",
                color="0.28",
                ha="center",
                va="center",
                fontsize=7.5,
            )

    paint_common_categories(crossing_axis)
    paint_mask(
        crossing_axis, exited_without_crossing, category_colors["exit"]
    )
    paint_mask(crossing_axis, age_censored, category_colors["age"])
    time_image = crossing_axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_where(~crossed, crossing_age / 1_000.0),
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=REPORT_AGES[-1] / 1_000.0),
        shading="flat",
        rasterized=True,
    )
    format_map_axis(
        crossing_axis, "B   First fixed-interior overturn age"
    )
    label_common_regions(crossing_axis, label_no_cohort=True)
    time_colorbar = figure.colorbar(
        time_image, cax=time_colorbar_axis
    )
    time_colorbar.set_ticks(np.arange(0.0, 8.1, 2.0))
    time_colorbar.set_label(r"First $J=0$ age, $t_{J=0}$ (kyr)")
    outcome_handles = [
        Patch(
            facecolor=category_colors["invalid"],
            edgecolor="0.65",
            label="geometrically invalid",
        ),
        Patch(
            facecolor=category_colors["no cohort"],
            edgecolor="0.65",
            label="no deposited cohort",
        ),
        Patch(
            facecolor=category_colors["exit"],
            edgecolor="0.65",
            label="exited before overturn",
        ),
        Patch(
            facecolor=category_colors["age"],
            edgecolor="0.65",
            label="still in domain; no overturn by 8 kyr (right-censored)",
        ),
    ]
    if np.any(solver_failure):
        outcome_handles.append(
            Patch(
                facecolor=category_colors["failure"],
                edgecolor="0.65",
                label="solver failure",
            )
        )
    category_axis.axis("off")
    category_axis.legend(
        handles=outcome_handles,
        loc="center",
        ncol=2,
        frameon=False,
        handlelength=1.3,
        columnspacing=2.0,
    )

    j_image = None
    for axis, age, panel in (
        (early_axis, REPORT_AGES[0], "C"),
        (late_axis, REPORT_AGES[1], "D"),
    ):
        paint_common_categories(axis)
        values = j_by_age[age]
        j_image = axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_where(~finite_cohort, values),
            cmap=diverging,
            norm=norm,
            shading="flat",
            rasterized=True,
        )
        format_map_axis(
            axis,
            rf"{panel}   Minimum $J$ through {age / 1_000:g} kyr or exit",
            show_ylabel=axis is early_axis,
        )
        label_common_regions(axis)
    j_colorbar = figure.colorbar(
        j_image,
        cax=j_colorbar_axis,
        orientation="horizontal",
    )
    j_colorbar.set_ticks(np.arange(-1.0, 1.01, 0.5))
    j_colorbar.set_label(
        r"Minimum fixed-interior $J=\partial x/\partial x_0$"
        " (negative indicates overturning)"
    )
    figure.savefig(path, dpi=300)
    if str(path).lower().endswith(".png"):
        figure.savefig(str(path)[:-4] + ".pdf", dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--maximum-age", type=float, default=8_000.0)
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
    if not args.plot_only:
        if args.maximum_age < max(REPORT_AGES):
            parser.error(
                f"--maximum-age must be at least {max(REPORT_AGES):g} yr"
            )
        if args.time_step <= 0.0:
            parser.error("--time-step must be positive")
        if any(
            not np.isclose(
                age / args.time_step,
                round(age / args.time_step),
                rtol=0.0,
                atol=1.0e-10,
            )
            for age in REPORT_AGES
        ):
            parser.error(
                "--time-step must divide the 2 kyr and 8 kyr reporting "
                "horizons exactly"
            )
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
