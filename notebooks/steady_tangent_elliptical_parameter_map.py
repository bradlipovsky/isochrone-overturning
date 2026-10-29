"""Audit buffered Eulerian tangent estimates in symmetric elliptical troughs.

This calculation replaces finite-duration particle cohorts with the stationary
equal-age tangent equation

    v . grad(T) = grad(v) T,       T_surface = (1, 0),       J = T_x.

The minimum is taken over the central 80% of surface deposition sources.  A
small outflow buffer suppresses the weak-DG
boundary layer.  The production value uses surface provenance >= 0.95 and a
two-cell outflow buffer; all nine combinations of provenance >= 0.90, 0.95,
or 0.99 and a one-, two-, or three-cell buffer are retained to expose mask
sensitivity.  No age horizon or time-dependent particle calculation is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import firedrake as fd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

from realistic_geometry_elliptical_family import (
    make_sampler,
    side_locations,
    solve_stokes,
)
from realistic_geometry_elliptical_parameter_map import (
    BOUNDARY_THICKNESSES,
    CENTER_THICKNESSES,
    HORIZONTAL_RADIUS,
    cell_edges,
    draw_geometry_schematic,
    geometry_from_thicknesses,
)
from steady_age_elliptical_diagnostic import (
    solve_stationary_tangent,
    solve_upwind_transport,
)


PROVENANCE_THRESHOLDS = (0.90, 0.95, 0.99)
OUTFLOW_BUFFER_CELLS = (1.0, 2.0, 3.0)
MAIN_PROVENANCE_THRESHOLD = 0.95
MAIN_OUTFLOW_BUFFER_CELLS = 2.0
SOURCE_COUNT = 401
PROVENANCE_DEGREE = 1

MAIN_MINIMUM_COLUMN = (
    "minimum stationary tangent J "
    "(provenance>=0.95, outflow buffer=2h)"
)
MAIN_FIRST_PERCENTILE_COLUMN = (
    "first percentile stationary tangent J "
    "(provenance>=0.95, outflow buffer=2h)"
)
RAW_MINIMUM_COLUMN = (
    "raw minimum stationary tangent J "
    "(provenance>=0.95, no outflow buffer)"
)
RAW_FIRST_PERCENTILE_COLUMN = (
    "raw first percentile stationary tangent J "
    "(provenance>=0.95, no outflow buffer)"
)

plt.rcParams.update(
    {
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sensitivity_column(provenance_threshold, buffer_cells):
    """Return the explicit CSV label for one stationary mask."""
    return (
        "minimum stationary tangent J "
        f"(provenance>={provenance_threshold:.2f}, "
        f"outflow buffer={buffer_cells:g}h)"
    )


def solve_stationary_parameter_fields(velocity):
    """Solve only the stationary fields required for a minimum-J map."""
    mesh = velocity.function_space().mesh()
    x, _ = fd.SpatialCoordinate(mesh)
    provenance = solve_upwind_transport(
        velocity,
        degree=PROVENANCE_DEGREE,
        volume_source=fd.Constant(0.0),
        surface_inflow_value=fd.Constant(1.0),
    )
    source_x = solve_upwind_transport(
        velocity,
        degree=1,
        volume_source=fd.Constant(0.0),
        surface_inflow_value=x,
    )
    tangent = solve_stationary_tangent(velocity)

    cell_space = fd.FunctionSpace(mesh, "DG", 0)
    vector_cell_space = fd.VectorFunctionSpace(mesh, "DG", 0)
    return {
        "coordinates": fd.Function(vector_cell_space).interpolate(
            fd.SpatialCoordinate(mesh)
        ),
        "velocity": fd.project(velocity, vector_cell_space),
        "provenance": fd.project(provenance, cell_space),
        "source x": fd.project(source_x, cell_space),
        "tangent J": fd.project(tangent[0], cell_space),
    }


def minimum_tangent_j(
    fields,
    source_interval,
    geometry,
    resolution,
    provenance_threshold,
    buffer_cells,
):
    """Return minimum stationary J after a stated provenance/outflow mask."""
    coordinates = np.asarray(fields["coordinates"].dat.data_ro)
    velocity = np.asarray(fields["velocity"].dat.data_ro)
    provenance = np.asarray(fields["provenance"].dat.data_ro).reshape(-1)
    source_x = np.asarray(fields["source x"].dat.data_ro).reshape(-1)
    tangent_j = np.asarray(fields["tangent J"].dat.data_ro).reshape(-1)
    source_min, source_max = source_interval
    left_side, right_side = side_locations(geometry)
    distance = buffer_cells * resolution

    # Only mask a boundary where the velocity actually leaves the domain.
    # Inflow cells are already separated by the stationary provenance field.
    outflow_buffer = (
        (
            (coordinates[:, 0] < left_side + distance)
            & (velocity[:, 0] < 0.0)
        )
        | (
            (coordinates[:, 0] > right_side - distance)
            & (velocity[:, 0] > 0.0)
        )
        | (
            (coordinates[:, 1] > -distance)
            & (velocity[:, 1] > 0.0)
        )
    )
    valid = (
        np.isfinite(provenance)
        & np.isfinite(source_x)
        & np.isfinite(tangent_j)
        & (provenance >= provenance_threshold)
        & (source_x >= source_min)
        & (source_x <= source_max)
        & (coordinates[:, 1] < -0.25)
        & ~outflow_buffer
    )
    if not np.any(valid):
        raise RuntimeError("stationary mask contains no surface-fed cells")
    values = tangent_j[valid]
    return {
        "minimum": float(np.min(values)),
        "first percentile": float(np.quantile(values, 0.01)),
        "cell count": int(np.count_nonzero(valid)),
    }


def case_inputs(boundary_thickness, center_thickness, resolution):
    """Return inputs recorded for every valid or invalid parameter cell."""
    return {
        "boundary thickness input (m)": float(boundary_thickness),
        "center thickness input (m)": float(center_thickness),
        "fixed horizontal radius (m)": HORIZONTAL_RADIUS,
        "input target spacing (m)": float(resolution),
        "stationary source count": SOURCE_COUNT,
        "stationary provenance DG degree": PROVENANCE_DEGREE,
        "main provenance threshold": MAIN_PROVENANCE_THRESHOLD,
        "main outflow buffer (cells)": MAIN_OUTFLOW_BUFFER_CELLS,
    }


def central_surface_source_interval(geometry, sample, source_count=SOURCE_COUNT):
    """Return the central 80% of depositing surface sources, or ``None``."""
    left_side, _ = side_locations(geometry)
    sources = np.linspace(0.96 * left_side, 0.04 * left_side, source_count)
    accumulation = -sample("w", sources, np.full_like(sources, -0.05))
    depositing_sources = sources[accumulation > 1.0e-6]
    margin = int(np.ceil(0.10 * len(depositing_sources)))
    interior = depositing_sources[
        margin : len(depositing_sources) - margin
    ]
    if len(interior) == 0:
        return None, len(depositing_sources)
    return (
        (float(np.min(interior)), float(np.max(interior))),
        len(depositing_sources),
    )


def solve_parameter_case(task):
    """Solve one geometry; module scope permits spawn-based parallelism."""
    boundary_thickness, center_thickness, resolution = task
    inputs = case_inputs(*task)
    geometry = geometry_from_thicknesses(boundary_thickness, center_thickness)
    inputs |= {
        "vertical radius (m)": geometry["vertical radius"],
        "vertical center height (m)": geometry["center height"],
    }
    try:
        start = perf_counter()
        velocity, stokes = solve_stokes(geometry, resolution)
        stokes_seconds = perf_counter() - start
        sample, speed_range = make_sampler(geometry, velocity)
        source_interval, depositing_source_count = central_surface_source_interval(
            geometry, sample, source_count=SOURCE_COUNT
        )
        if source_interval is None:
            return inputs | {
                "status": "no surface-deposition source",
                "mesh cells": stokes["cells"],
                "Stokes solve wall time (s)": stokes_seconds,
                "minimum grid u (m/yr)": speed_range[0],
                "maximum grid u (m/yr)": speed_range[1],
                "depositing surface source count": depositing_source_count,
                MAIN_MINIMUM_COLUMN: np.nan,
            }

        start = perf_counter()
        fields = solve_stationary_parameter_fields(velocity)
        transport_seconds = perf_counter() - start
        try:
            raw = minimum_tangent_j(
                fields,
                source_interval,
                geometry,
                resolution,
                MAIN_PROVENANCE_THRESHOLD,
                buffer_cells=0.0,
            )
        except RuntimeError:
            raw = {
                "minimum": np.nan,
                "first percentile": np.nan,
                "cell count": 0,
            }
        sensitivity = {}
        minima = []
        sensitivity_mask_count = 0
        for provenance_threshold in PROVENANCE_THRESHOLDS:
            for buffer_cells in OUTFLOW_BUFFER_CELLS:
                column = sensitivity_column(
                    provenance_threshold, buffer_cells
                )
                try:
                    summary = minimum_tangent_j(
                        fields,
                        source_interval,
                        geometry,
                        resolution,
                        provenance_threshold,
                        buffer_cells,
                    )
                except RuntimeError:
                    sensitivity[column] = np.nan
                    sensitivity[
                        column.replace("minimum", "first percentile", 1)
                    ] = np.nan
                    sensitivity[
                        column.replace(
                            "minimum", "retained cell count", 1
                        )
                    ] = 0
                    continue
                sensitivity[column] = summary["minimum"]
                sensitivity[
                    column.replace("minimum", "first percentile", 1)
                ] = summary["first percentile"]
                sensitivity[
                    column.replace("minimum", "retained cell count", 1)
                ] = summary["cell count"]
                minima.append(summary["minimum"])
                sensitivity_mask_count += 1

        minima = np.asarray(minima)
        main_minimum = sensitivity[MAIN_MINIMUM_COLUMN]
        sign_changes = bool(
            len(minima)
            and np.any(minima < 0.0)
            and np.any(minima >= 0.0)
        )
        if not np.isfinite(main_minimum):
            sign_classification = "unresolved: no stationary streamtube"
        elif sign_changes:
            sign_classification = "unresolved: mask-sensitive sign"
        elif main_minimum < -0.05:
            sign_classification = "resolved overturn"
        elif main_minimum > 0.05:
            sign_classification = "resolved ordered"
        else:
            sign_classification = "unresolved: |minimum J|<=0.05"
        if len(minima) and np.all(minima < 0.0):
            sign_robustness = "negative for all available masks"
        elif len(minima) and np.all(minima >= 0.0):
            sign_robustness = "nonnegative for all available masks"
        elif len(minima):
            sign_robustness = "mask-sensitive sign"
        else:
            sign_robustness = "no sensitivity mask retained cells"

        provenance_values = np.asarray(
            fields["provenance"].dat.data_ro
        ).reshape(-1)

        return inputs | {
            "status": (
                "success"
                if np.isfinite(main_minimum)
                else "no resolved stationary streamtube"
            ),
            "mesh cells": stokes["cells"],
            "Stokes solve wall time (s)": stokes_seconds,
            "stationary transport wall time (s)": transport_seconds,
            "minimum grid u (m/yr)": speed_range[0],
            "maximum grid u (m/yr)": speed_range[1],
            "depositing surface source count": depositing_source_count,
            "projected provenance minimum": float(
                np.min(provenance_values)
            ),
            "projected provenance maximum": float(
                np.max(provenance_values)
            ),
            "fixed source minimum x0 (m)": source_interval[0],
            "fixed source maximum x0 (m)": source_interval[1],
            RAW_MINIMUM_COLUMN: raw["minimum"],
            RAW_FIRST_PERCENTILE_COLUMN: raw["first percentile"],
            MAIN_FIRST_PERCENTILE_COLUMN: sensitivity[
                MAIN_FIRST_PERCENTILE_COLUMN
            ],
            "stationary J classification": sign_classification,
            "stationary sign robustness": sign_robustness,
            "sensitivity masks with retained cells": sensitivity_mask_count,
            "sensitivity mask set complete": sensitivity_mask_count == 9,
            "sign changes across available masks": sign_changes,
            "raw/buffered sign disagreement": (
                bool(
                    np.isfinite(raw["minimum"])
                    and np.isfinite(main_minimum)
                    and ((raw["minimum"] < 0.0) != (main_minimum < 0.0))
                )
            ),
            "minimum J across available masks": (
                float(np.min(minima)) if len(minima) else np.nan
            ),
            "maximum of minimum J across available masks": (
                float(np.max(minima)) if len(minima) else np.nan
            ),
        } | sensitivity
    except Exception as error:
        return inputs | {
            "status": f"failed: {type(error).__name__}: {error}",
            MAIN_MINIMUM_COLUMN: np.nan,
        }


def run_grid(
    boundary_thicknesses,
    center_thicknesses,
    resolution,
    jobs=1,
    existing=None,
    checkpoint_path=None,
    checkpoint_every=10,
):
    """Run a regular grid, optionally reusing matching successful rows."""
    reusable = {}
    if existing is not None and len(existing):
        required = {
            MAIN_MINIMUM_COLUMN,
            "stationary sign robustness",
            "stationary provenance DG degree",
        }
        if required.issubset(existing.columns):
            for _, row in existing.iterrows():
                if int(row["stationary provenance DG degree"]) != PROVENANCE_DEGREE:
                    continue
                key = (
                    float(row["boundary thickness input (m)"]),
                    float(row["center thickness input (m)"]),
                    float(row["input target spacing (m)"]),
                )
                reusable[key] = row.to_dict()

    rows = []
    pending = []
    for center_thickness in center_thicknesses:
        for boundary_thickness in boundary_thicknesses:
            task = (
                float(boundary_thickness),
                float(center_thickness),
                float(resolution),
            )
            if center_thickness <= boundary_thickness:
                rows.append(
                    case_inputs(*task)
                    | {
                        "status": (
                            "invalid: center thickness does not exceed "
                            "boundary thickness"
                        ),
                        MAIN_MINIMUM_COLUMN: np.nan,
                    }
                )
            elif task in reusable and reusable[task].get("status") in (
                "success",
                "no surface-deposition source",
                "no resolved stationary streamtube",
            ):
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
                        "minimum stationary J": row.get(
                            MAIN_MINIMUM_COLUMN
                        ),
                        "sign robustness": row.get(
                            "stationary sign robustness"
                        ),
                    }
                ),
                flush=True,
            )
            if checkpoint_path is not None and index % checkpoint_every == 0:
                pd.DataFrame(rows).sort_values(
                    [
                        "center thickness input (m)",
                        "boundary thickness input (m)",
                    ]
                ).to_csv(checkpoint_path, index=False, na_rep="NA")
    finally:
        if executor is not None:
            executor.shutdown()
    return add_robustness_classifications(
        pd.DataFrame(rows).sort_values(
            [
                "center thickness input (m)",
                "boundary thickness input (m)",
            ]
        )
    )


def add_robustness_classifications(table):
    """Backfill mask robustness deterministically for reused/checkpoint rows."""
    table = table.copy()
    sensitivity_columns = [
        sensitivity_column(threshold, buffer_cells)
        for threshold in PROVENANCE_THRESHOLDS
        for buffer_cells in OUTFLOW_BUFFER_CELLS
    ]
    for column in sensitivity_columns:
        if column not in table:
            table[column] = np.nan

    for index, row in table.iterrows():
        values = pd.to_numeric(
            row[sensitivity_columns], errors="coerce"
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        main = pd.to_numeric(
            pd.Series([row.get(MAIN_MINIMUM_COLUMN)]), errors="coerce"
        ).iloc[0]
        raw = pd.to_numeric(
            pd.Series([row.get(RAW_MINIMUM_COLUMN)]), errors="coerce"
        ).iloc[0]
        sign_changes = bool(
            len(values)
            and np.any(values < 0.0)
            and np.any(values >= 0.0)
        )
        table.at[index, "sensitivity masks with retained cells"] = len(values)
        table.at[index, "sensitivity mask set complete"] = len(values) == 9
        table.at[index, "sign changes across available masks"] = sign_changes
        table.at[index, "raw/buffered sign disagreement"] = bool(
            np.isfinite(raw)
            and np.isfinite(main)
            and ((raw < 0.0) != (main < 0.0))
        )
        if len(values):
            table.at[index, "minimum J across available masks"] = np.min(values)
            table.at[index, "maximum of minimum J across available masks"] = np.max(values)
            if np.all(values < 0.0):
                robustness = "negative for all available masks"
            elif np.all(values >= 0.0):
                robustness = "nonnegative for all available masks"
            else:
                robustness = "mask-sensitive sign"
            table.at[index, "stationary sign robustness"] = robustness
        if row.get("status") != "success" or not np.isfinite(main):
            continue
        if sign_changes:
            classification = "unresolved: mask-sensitive sign"
        elif main < -0.05:
            classification = "resolved overturn"
        elif main > 0.05:
            classification = "resolved ordered"
        else:
            classification = "unresolved: |minimum J|<=0.05"
        table.at[index, "stationary J classification"] = classification
    return table


def save_figure(table, path):
    """Plot geometry and the conventional buffered stationary estimate."""
    table = add_robustness_classifications(table)
    boundary_thicknesses = np.sort(
        table["boundary thickness input (m)"].unique().astype(float)
    )
    center_thicknesses = np.sort(
        table["center thickness input (m)"].unique().astype(float)
    )
    status = table.pivot(
        index="center thickness input (m)",
        columns="boundary thickness input (m)",
        values="status",
    ).reindex(index=center_thicknesses, columns=boundary_thicknesses)
    minimum_j = table.pivot(
        index="center thickness input (m)",
        columns="boundary thickness input (m)",
        values=MAIN_MINIMUM_COLUMN,
    ).reindex(index=center_thicknesses, columns=boundary_thicknesses)
    status_values = status.fillna("").to_numpy(dtype=str)
    success = status_values == "success"
    invalid = np.char.startswith(status_values, "invalid:")
    no_source = status_values == "no surface-deposition source"
    no_streamtube = status_values == "no resolved stationary streamtube"
    failures = np.char.startswith(status_values, "failed:")
    values = minimum_j.to_numpy(dtype=float)
    if not np.any(success & np.isfinite(values)):
        raise RuntimeError("stationary parameter map has no finite values")

    diameter = 2.0 * HORIZONTAL_RADIUS
    x_centers = boundary_thicknesses / diameter
    y_centers = center_thicknesses / diameter
    x_edges = cell_edges(x_centers)
    y_edges = cell_edges(y_centers)
    diverging = LinearSegmentedColormap.from_list(
        "negative_red_zero_white_positive_blue",
        ["#67001f", "#d6604d", "#ffffff", "#4393c3", "#053061"],
        N=257,
    )
    normalization = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    figure, (schematic_axis, map_axis) = plt.subplots(
        1,
        2,
        figsize=(8.1, 3.65),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.02, 1.0)},
    )
    draw_geometry_schematic(schematic_axis)
    image = map_axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_where(~success, values),
        cmap=diverging,
        norm=normalization,
        shading="flat",
        rasterized=True,
    )
    if np.any(invalid):
        map_axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_where(~invalid, np.ones(invalid.shape)),
            cmap=ListedColormap(["#dedede"]),
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            rasterized=True,
        )
        map_axis.text(
            0.64,
            0.42,
            "geometrically\ninvalid",
            color="0.25",
            ha="center",
            va="center",
            rotation=43,
            fontsize=8.0,
        )
    if np.any(no_source):
        map_axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_where(~no_source, np.ones(no_source.shape)),
            cmap=ListedColormap(["#e7dcec"]),
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            rasterized=True,
        )
        map_axis.text(
            0.57,
            0.895,
            "no surface-\ndeposition source",
            color="0.28",
            ha="center",
            va="center",
            fontsize=7.5,
        )
    if np.any(no_streamtube):
        map_axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_where(~no_streamtube, np.ones(no_streamtube.shape)),
            cmap=ListedColormap(["#f1e6c8"]),
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            rasterized=True,
        )
        map_axis.text(
            0.39,
            0.755,
            "provenance\nmask empty",
            color="0.28",
            ha="center",
            va="center",
            fontsize=7.0,
        )
    if np.any(failures):
        map_axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_where(~failures, np.ones(failures.shape)),
            cmap=ListedColormap(["#efd0d0"]),
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            rasterized=True,
        )

    map_axis.set(
        xlabel=r"Boundary thickness, $H_b/D$",
        ylabel=r"Center thickness, $H_c/D$",
        xlim=(x_edges[0], x_edges[-1]),
        ylim=(y_edges[0], y_edges[-1]),
    )
    map_axis.set_title(
        "B   Buffered stationary estimate", loc="left"
    )
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.tick_params(direction="in")
    colorbar = figure.colorbar(
        image,
        ax=map_axis,
        orientation="horizontal",
        pad=0.18,
        fraction=0.08,
        aspect=30,
        extend="both",
    )
    colorbar.set_ticks(np.arange(-1.0, 1.01, 0.5))
    colorbar.set_label(
        r"Buffered stationary minimum $J=T_x$ "
        r"(20 m buffer; provenance $\geq 0.95$)"
    )
    mask_sensitive_count = int(
        np.count_nonzero(
            table["stationary J classification"]
            == "unresolved: mask-sensitive sign"
        )
    )
    map_axis.text(
        0.5,
        -0.31,
        f"Mask-sensitive sign: {mask_sensitive_count} cells; "
        "continuous conventional estimates are shown.",
        transform=map_axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.25",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument(
        "--boundary-thickness",
        type=float,
        action="append",
        help="Hb in m; repeat for a custom smoke-test grid",
    )
    parser.add_argument(
        "--center-thickness",
        type=float,
        action="append",
        help="Hc in m; repeat for a custom smoke-test grid",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "data" / "steady_tangent_elliptical_parameter_map.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=root / "figures" / "steady_tangent_elliptical_parameter_map.png",
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        table = pd.read_csv(args.csv)
        save_figure(table, args.figure)
        print(f"Wrote {args.figure}", flush=True)
        return

    boundary_thicknesses = np.asarray(
        args.boundary_thickness or BOUNDARY_THICKNESSES, dtype=float
    )
    center_thicknesses = np.asarray(
        args.center_thickness or CENTER_THICKNESSES, dtype=float
    )
    existing = None
    if args.resume and args.csv.exists():
        existing = pd.read_csv(args.csv)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    table = run_grid(
        np.sort(np.unique(boundary_thicknesses)),
        np.sort(np.unique(center_thicknesses)),
        args.resolution,
        jobs=args.jobs,
        existing=existing,
        checkpoint_path=args.csv,
    )
    table.to_csv(args.csv, index=False, na_rep="NA")
    print(f"Wrote {args.csv}", flush=True)
    save_figure(table, args.figure)
    print(f"Wrote {args.figure}", flush=True)


if __name__ == "__main__":
    main()
