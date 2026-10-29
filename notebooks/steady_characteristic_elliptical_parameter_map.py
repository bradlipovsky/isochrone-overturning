"""Map symmetric-trough overturning with direct steady characteristics.

For each fixed full-Stokes velocity, surface-origin characteristics carry the
equal-age material tangent

    dX/dt = v(X),       dT/dt = grad(v)(X) T,       T(0) = (1, 0).

Every retained source is advanced independently to its first sampled-domain
boundary exit.  The mapped diagnostic is the minimum horizontal tangent
``J = T_x`` reached before that individual exit.  Particle age parameterizes
the steady characteristic; neither the glacier nor a deposited cohort evolves
in time.  This removes both a reporting-age choice and the earlier convention
of stopping all sources when the first source exits.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

from realistic_geometry_elliptical_family import (
    bed_height,
    side_locations,
    solve_stokes,
)
from realistic_geometry_elliptical_parameter_map import (
    BOUNDARY_THICKNESSES,
    CENTER_THICKNESSES,
    HORIZONTAL_RADIUS,
    REFERENCE_HALF_DOMAIN,
    cell_edges,
    draw_geometry_schematic,
    geometry_from_thicknesses,
)
from steady_tangent_elliptical_validation import make_variational_sampler


DIRECT_MINIMUM_COLUMN = (
    "minimum direct steady-characteristic J to individual exit (central 80%)"
)
DIRECT_PERCENTILE_COLUMN = (
    "first percentile of per-source minimum J (central 80%)"
)
CLASSIFICATION_COLUMN = "direct steady-characteristic classification"
TRIM_FRACTIONS = (0.70, 0.80, 0.90)
PRODUCTION_TRIM_FRACTION = 0.80
TRACED_TRIM_FRACTION = 0.90

DEFAULT_RESOLUTION = 10.0
DEFAULT_TIME_STEP = 10.0
DEFAULT_SOURCE_COUNT = 161
DEFAULT_SAMPLER_NX = 601
DEFAULT_SAMPLER_NZ = 181
DEFAULT_MAXIMUM_AGE = 120_000.0

SIDE_FRACTION = 0.994
BED_SIGMA = 0.006
SURFACE_SIGMA = 0.99994

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


def classify(minimum_j):
    """Apply a conservative numerical resolution band around J=0."""
    if not np.isfinite(minimum_j):
        return "not evaluated"
    if minimum_j < -0.05:
        return "resolved overturn"
    if minimum_j > 0.05:
        return "resolved ordered"
    return "unresolved: |minimum J|<=0.05"


def retained_surface_sources(
    geometry,
    sample,
    source_count,
    retained_fraction=TRACED_TRIM_FRACTION,
):
    """Return a centered fraction of depositing left-surface sources."""
    left_side, _ = side_locations(geometry)
    candidates = np.linspace(
        0.96 * left_side, 0.04 * left_side, source_count
    )
    accumulation = -sample(
        "w", candidates, np.full_like(candidates, -0.05)
    )
    depositing = candidates[accumulation > 1.0e-6]
    margin_fraction = 0.5 * (1.0 - retained_fraction)
    margin = int(np.ceil(margin_fraction * len(depositing)))
    retained = depositing[margin : len(depositing) - margin]
    return retained, depositing


def central_source_subset(trajectories, depositing_sources, fraction):
    """Select one centered nested source subset from traced trajectories."""
    margin = int(np.ceil(0.5 * (1.0 - fraction) * len(depositing_sources)))
    sources = depositing_sources[margin : len(depositing_sources) - margin]
    if len(sources) == 0:
        return trajectories.iloc[0:0]
    source_values = trajectories["source x0 (m)"].to_numpy()
    mask = (source_values >= np.min(sources) - 1.0e-10) & (
        source_values <= np.max(sources) + 1.0e-10
    )
    return trajectories.loc[mask]


def trim_minimum_column(fraction):
    return (
        "minimum direct steady-characteristic J to individual exit "
        f"(central {100 * fraction:.0f}%)"
    )


def trim_percentile_column(fraction):
    return (
        "first percentile of per-source minimum J "
        f"(central {100 * fraction:.0f}%)"
    )


def local_sigma(points, geometry):
    """Bed-following height of each point."""
    bed = bed_height(points[:, 0], geometry)
    return (points[:, 1] - bed) / (-bed)


def rk4_step(sample_vector, sample_gradient, points, tangents, step):
    """Advance positions and tangent sensitivities by one vectorized step."""
    k1 = sample_vector(points)
    f1 = np.einsum("nij,nj->ni", sample_gradient(points), tangents)
    p2 = points + 0.5 * step * k1
    t2 = tangents + 0.5 * step * f1
    k2 = sample_vector(p2)
    f2 = np.einsum("nij,nj->ni", sample_gradient(p2), t2)
    p3 = points + 0.5 * step * k2
    t3 = tangents + 0.5 * step * f2
    k3 = sample_vector(p3)
    f3 = np.einsum("nij,nj->ni", sample_gradient(p3), t3)
    p4 = points + step * k3
    t4 = tangents + step * f3
    k4 = sample_vector(p4)
    f4 = np.einsum("nij,nj->ni", sample_gradient(p4), t4)
    return (
        points + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
        tangents + step * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0,
    )


def first_exit_fraction(
    old_points,
    new_points,
    old_sigma,
    new_sigma,
    left_boundary,
    right_boundary,
):
    """Linearly locate the first conservative domain boundary crossed."""
    count = len(old_points)
    fraction = np.ones(count)
    kind = np.full(count, "numerical failure", dtype=object)
    candidates = (
        (
            "left side",
            old_points[:, 0],
            new_points[:, 0],
            left_boundary,
            new_points[:, 0] <= left_boundary,
        ),
        (
            "right side",
            old_points[:, 0],
            new_points[:, 0],
            right_boundary,
            new_points[:, 0] >= right_boundary,
        ),
        ("bed", old_sigma, new_sigma, BED_SIGMA, new_sigma <= BED_SIGMA),
        (
            "surface",
            old_sigma,
            new_sigma,
            SURFACE_SIGMA,
            new_sigma >= SURFACE_SIGMA,
        ),
    )
    for name, old_value, new_value, boundary, crossed in candidates:
        denominator = new_value - old_value
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate_fraction = (boundary - old_value) / denominator
        eligible = (
            crossed
            & np.isfinite(candidate_fraction)
            & (candidate_fraction >= 0.0)
            & (candidate_fraction <= fraction)
        )
        fraction[eligible] = candidate_fraction[eligible]
        kind[eligible] = name
    return fraction, kind


def trace_sources_to_exit(
    geometry,
    sample_vector,
    sample_gradient,
    sources,
    time_step=DEFAULT_TIME_STEP,
    maximum_age=DEFAULT_MAXIMUM_AGE,
):
    """Carry all source tangents independently to first boundary exit."""
    left_side, right_side = side_locations(geometry)
    left_boundary = SIDE_FRACTION * left_side
    right_boundary = SIDE_FRACTION * right_side
    count = len(sources)
    points = np.column_stack((sources, np.full(count, -0.05)))
    tangents = np.column_stack((np.ones(count), np.zeros(count)))
    active = np.ones(count, dtype=bool)
    age = np.zeros(count)
    minimum_j = np.ones(count)
    minimum_age = np.zeros(count)
    first_zero_age = np.full(count, np.nan)
    exit_age = np.full(count, np.nan)
    exit_kind = np.full(count, "active", dtype=object)
    initial_velocity = sample_vector(points)
    previous_u = initial_velocity[:, 0]
    minimum_u = previous_u.copy()
    reversal_count = np.zeros(count, dtype=int)

    maximum_steps = int(np.ceil(maximum_age / time_step))
    for _ in range(maximum_steps):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        old_points = points[indices].copy()
        old_tangents = tangents[indices].copy()
        old_age = age[indices].copy()
        old_sigma = local_sigma(old_points, geometry)
        candidate_points, candidate_tangents = rk4_step(
            sample_vector,
            sample_gradient,
            old_points,
            old_tangents,
            time_step,
        )
        candidate_sigma = local_sigma(candidate_points, geometry)
        finite = np.all(np.isfinite(candidate_points), axis=1) & np.all(
            np.isfinite(candidate_tangents), axis=1
        )
        remains = (
            finite
            & (candidate_sigma > BED_SIGMA)
            & (candidate_sigma < SURFACE_SIGMA)
            & (candidate_points[:, 0] > left_boundary)
            & (candidate_points[:, 0] < right_boundary)
        )

        accepted_points = candidate_points.copy()
        accepted_tangents = candidate_tangents.copy()
        accepted_age = old_age + time_step
        exiting = ~remains
        step_exit_kind = np.full(len(indices), "", dtype=object)
        if np.any(exiting):
            fraction, local_kind = first_exit_fraction(
                old_points[exiting],
                candidate_points[exiting],
                old_sigma[exiting],
                candidate_sigma[exiting],
                left_boundary,
                right_boundary,
            )
            accepted_points[exiting] = old_points[exiting] + fraction[:, None] * (
                candidate_points[exiting] - old_points[exiting]
            )
            accepted_tangents[exiting] = old_tangents[exiting] + fraction[:, None] * (
                candidate_tangents[exiting] - old_tangents[exiting]
            )
            accepted_age[exiting] = old_age[exiting] + fraction * time_step
            step_exit_kind[exiting] = local_kind

        old_j = old_tangents[:, 0]
        accepted_j = accepted_tangents[:, 0]
        crossed = (
            np.isnan(first_zero_age[indices])
            & (old_j > 0.0)
            & (accepted_j <= 0.0)
        )
        if np.any(crossed):
            fraction = old_j[crossed] / (
                old_j[crossed] - accepted_j[crossed]
            )
            first_zero_age[indices[crossed]] = old_age[crossed] + fraction * (
                accepted_age[crossed] - old_age[crossed]
            )
        improves = accepted_j < minimum_j[indices]
        if np.any(improves):
            selected = indices[improves]
            minimum_j[selected] = accepted_j[improves]
            minimum_age[selected] = accepted_age[improves]

        accepted_velocity = sample_vector(accepted_points)
        accepted_u = accepted_velocity[:, 0]
        minimum_u[indices] = np.minimum(minimum_u[indices], accepted_u)
        scale = np.maximum(
            np.maximum(np.abs(previous_u[indices]), np.abs(accepted_u)),
            1.0e-12,
        )
        reversal = (
            np.signbit(previous_u[indices]) != np.signbit(accepted_u)
        ) & (
            np.minimum(np.abs(previous_u[indices]), np.abs(accepted_u))
            > 1.0e-6 * scale
        )
        reversal_count[indices] += reversal.astype(int)
        previous_u[indices] = accepted_u
        points[indices] = accepted_points
        tangents[indices] = accepted_tangents
        age[indices] = accepted_age
        if np.any(exiting):
            exited_indices = indices[exiting]
            active[exited_indices] = False
            exit_age[exited_indices] = accepted_age[exiting]
            exit_kind[exited_indices] = step_exit_kind[exiting]

    if np.any(active):
        exit_age[active] = age[active]
        exit_kind[active] = "safety cap"
    return pd.DataFrame(
        {
            "source x0 (m)": sources,
            "minimum J": minimum_j,
            "age of minimum J (yr)": minimum_age,
            "first J=0 age (yr)": first_zero_age,
            "minimum path u (m/yr)": minimum_u,
            "horizontal velocity reversal count": reversal_count,
            "exit age (yr)": exit_age,
            "exit kind": exit_kind,
        }
    )


def case_inputs(
    boundary,
    center,
    resolution,
    time_step,
    source_count,
    traced_fraction,
    sampler_nx,
    sampler_nz,
    maximum_age,
):
    """Record every numerical and geometric production input."""
    return {
        "boundary thickness input (m)": boundary,
        "center thickness input (m)": center,
        "fixed horizontal radius (m)": HORIZONTAL_RADIUS,
        "input target spacing (m)": resolution,
        "characteristic time step (yr)": time_step,
        "candidate source count": source_count,
        "traced depositing-source fraction": traced_fraction,
        "velocity sampler nx": sampler_nx,
        "velocity sampler nz": sampler_nz,
        "safety maximum age (yr)": maximum_age,
        "side exit fraction": SIDE_FRACTION,
        "bed exit sigma": BED_SIGMA,
        "surface exit sigma": SURFACE_SIGMA,
    }


def solve_parameter_case(task):
    """Solve one geometry and trace all retained steady characteristics."""
    (
        boundary,
        center,
        resolution,
        time_step,
        source_count,
        traced_fraction,
        sampler_nx,
        sampler_nz,
        maximum_age,
    ) = task
    inputs = case_inputs(*task)
    geometry = geometry_from_thicknesses(boundary, center)
    inputs |= {
        "vertical radius (m)": geometry["vertical radius"],
        "vertical center height (m)": geometry["center height"],
    }
    try:
        started = perf_counter()
        velocity, stokes = solve_stokes(geometry, resolution)
        stokes_seconds = perf_counter() - started
        started = perf_counter()
        sample, sample_vector, sample_gradient = make_variational_sampler(
            geometry, velocity, nx=sampler_nx, nz=sampler_nz
        )
        sampler_seconds = perf_counter() - started
        sources, depositing = retained_surface_sources(
            geometry,
            sample,
            source_count,
            retained_fraction=traced_fraction,
        )
        if len(sources) == 0:
            return inputs | {
                "status": "no surface-deposition source",
                "mesh cells": stokes["cells"],
                "Stokes solve wall time (s)": stokes_seconds,
                "sampler construction wall time (s)": sampler_seconds,
                "depositing source count": len(depositing),
                "retained source count": 0,
                DIRECT_MINIMUM_COLUMN: np.nan,
                CLASSIFICATION_COLUMN: "not evaluated",
            }

        started = perf_counter()
        trajectories = trace_sources_to_exit(
            geometry,
            sample_vector,
            sample_gradient,
            sources,
            time_step=time_step,
            maximum_age=maximum_age,
        )
        trace_seconds = perf_counter() - started
        failures = trajectories["exit kind"] == "numerical failure"
        safety_caps = trajectories["exit kind"] == "safety cap"
        nested = {
            fraction: central_source_subset(
                trajectories, depositing, fraction
            )
            for fraction in TRIM_FRACTIONS
        }
        if any(len(values) == 0 for values in nested.values()):
            raise RuntimeError("a nested central source subset is empty")
        production = nested[PRODUCTION_TRIM_FRACTION]
        minimum_index = production["minimum J"].idxmin()
        minimum_row = production.loc[minimum_index]
        minimum_j = float(minimum_row["minimum J"])
        minimum_source = float(minimum_row["source x0 (m)"])
        production_sources = production["source x0 (m)"].to_numpy()
        source_spacing = (
            float(np.min(np.diff(production_sources)))
            if len(production_sources) > 1
            else np.nan
        )
        at_left_edge = bool(
            np.isclose(
                minimum_source,
                np.min(production_sources),
                rtol=0.0,
                atol=1.0e-10,
            )
        )
        at_right_edge = bool(
            np.isclose(
                minimum_source,
                np.max(production_sources),
                rtol=0.0,
                atol=1.0e-10,
            )
        )
        trim_minima = np.array(
            [nested[fraction]["minimum J"].min() for fraction in TRIM_FRACTIONS]
        )
        trim_sign_changes = bool(
            np.any(trim_minima < 0.0) and np.any(trim_minima >= 0.0)
        )
        if np.any(failures):
            status = "numerical characteristic failure"
        elif np.any(safety_caps):
            status = "characteristic safety-cap censoring"
        else:
            status = "success"
        exit_counts = trajectories["exit kind"].value_counts()
        return inputs | {
            "status": status,
            "mesh cells": stokes["cells"],
            "Stokes solve wall time (s)": stokes_seconds,
            "sampler construction wall time (s)": sampler_seconds,
            "characteristic wall time (s)": trace_seconds,
            "depositing source count": len(depositing),
            "traced source count": len(sources),
            "retained source count": len(production_sources),
            "fixed source minimum x0 (m)": float(
                np.min(production_sources)
            ),
            "fixed source maximum x0 (m)": float(
                np.max(production_sources)
            ),
            DIRECT_MINIMUM_COLUMN: minimum_j,
            DIRECT_PERCENTILE_COLUMN: float(
                np.quantile(production["minimum J"], 0.01)
            ),
            CLASSIFICATION_COLUMN: (
                "unresolved: trim-sensitive sign"
                if trim_sign_changes
                else classify(minimum_j)
            ),
            "sign changes across 70/80/90% trims": trim_sign_changes,
            "source x0 at minimum J (m)": minimum_source,
            "retained source spacing (m)": source_spacing,
            "minimum occurs at retained-source edge": (
                at_left_edge or at_right_edge
            ),
            "minimum retained-source edge": (
                "left" if at_left_edge else "right" if at_right_edge else "interior"
            ),
            "age at minimum J (yr)": float(
                minimum_row["age of minimum J (yr)"]
            ),
            "first J=0 age among retained sources (yr)": float(
                production["first J=0 age (yr)"].min(skipna=True)
            ),
            "minimum path u across retained sources (m/yr)": float(
                production["minimum path u (m/yr)"].min()
            ),
            "maximum retained-source horizontal velocity reversal count": int(
                production["horizontal velocity reversal count"].max()
            ),
            "minimum retained-source exit age (yr)": float(
                production["exit age (yr)"].min()
            ),
            "maximum retained-source exit age (yr)": float(
                production["exit age (yr)"].max()
            ),
            "left-side exit count": int(exit_counts.get("left side", 0)),
            "right-side exit count": int(exit_counts.get("right side", 0)),
            "bed exit count": int(exit_counts.get("bed", 0)),
            "surface exit count": int(exit_counts.get("surface", 0)),
            "safety-cap count": int(np.count_nonzero(safety_caps)),
            "numerical-failure count": int(np.count_nonzero(failures)),
        } | {
            trim_minimum_column(fraction): float(
                nested[fraction]["minimum J"].min()
            )
            for fraction in TRIM_FRACTIONS
        } | {
            trim_percentile_column(fraction): float(
                np.quantile(nested[fraction]["minimum J"], 0.01)
            )
            for fraction in TRIM_FRACTIONS
        }
    except Exception as error:
        return inputs | {
            "status": f"failed: {type(error).__name__}: {error}",
            DIRECT_MINIMUM_COLUMN: np.nan,
            CLASSIFICATION_COLUMN: "not evaluated",
        }


def run_grid(
    boundary_thicknesses,
    center_thicknesses,
    resolution,
    time_step,
    source_count,
    traced_fraction,
    sampler_nx,
    sampler_nz,
    maximum_age,
    jobs=1,
    existing=None,
    checkpoint_path=None,
    checkpoint_every=5,
):
    """Run the grid with spawn workers periodically recycled for Gmsh."""
    reusable = {}
    if existing is not None and len(existing):
        required = {
            DIRECT_MINIMUM_COLUMN,
            CLASSIFICATION_COLUMN,
            "velocity sampler nx",
            "velocity sampler nz",
            "traced depositing-source fraction",
        }
        if required.issubset(existing.columns):
            for _, row in existing.iterrows():
                key = (
                    float(row["boundary thickness input (m)"]),
                    float(row["center thickness input (m)"]),
                    float(row["input target spacing (m)"]),
                    float(row["characteristic time step (yr)"]),
                    int(row["candidate source count"]),
                    float(row["traced depositing-source fraction"]),
                    int(row["velocity sampler nx"]),
                    int(row["velocity sampler nz"]),
                    float(row["safety maximum age (yr)"]),
                )
                reusable[key] = row.to_dict()

    rows = []
    pending = []
    for center in center_thicknesses:
        for boundary in boundary_thicknesses:
            task = (
                float(boundary),
                float(center),
                float(resolution),
                float(time_step),
                int(source_count),
                float(traced_fraction),
                int(sampler_nx),
                int(sampler_nz),
                float(maximum_age),
            )
            if center <= boundary:
                rows.append(
                    case_inputs(*task)
                    | {
                        "status": (
                            "invalid: center thickness does not exceed "
                            "boundary thickness"
                        ),
                        DIRECT_MINIMUM_COLUMN: np.nan,
                        CLASSIFICATION_COLUMN: "not evaluated",
                    }
                )
            elif task in reusable and reusable[task].get("status") in (
                "success",
                "no surface-deposition source",
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
    if jobs == 1:
        solved = map(solve_parameter_case, pending)
        pool = None
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=jobs, maxtasksperchild=40)
        solved = pool.imap(solve_parameter_case, pending, chunksize=1)
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
                        "minimum direct J": row.get(DIRECT_MINIMUM_COLUMN),
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
        if pool is not None:
            pool.close()
            pool.join()
    return add_geometry_diagnostics(
        pd.DataFrame(rows).sort_values(
            ["center thickness input (m)", "boundary thickness input (m)"]
        )
    )


def extend_censored_cases(table, maximum_age, jobs=1):
    """Rerun only safety-capped cells with a larger adaptive ceiling."""
    table = table.copy()
    indices = table.index[
        table["status"] == "characteristic safety-cap censoring"
    ].tolist()
    tasks = []
    for index in indices:
        row = table.loc[index]
        tasks.append(
            (
                float(row["boundary thickness input (m)"]),
                float(row["center thickness input (m)"]),
                float(row["input target spacing (m)"]),
                float(row["characteristic time step (yr)"]),
                int(row["candidate source count"]),
                float(row["traced depositing-source fraction"]),
                int(row["velocity sampler nx"]),
                int(row["velocity sampler nz"]),
                float(maximum_age),
            )
        )
    print(
        f"Extending {len(tasks)} safety-capped cells to {maximum_age:g} yr.",
        flush=True,
    )
    if jobs == 1:
        solved = map(solve_parameter_case, tasks)
        pool = None
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=jobs, maxtasksperchild=20)
        solved = pool.imap(solve_parameter_case, tasks, chunksize=1)
    try:
        for number, (index, row) in enumerate(zip(indices, solved), start=1):
            for column in row:
                if column not in table:
                    table[column] = np.nan
            table.loc[index, :] = np.nan
            for column, value in row.items():
                table.at[index, column] = value
            print(
                f"Extended {number}/{len(tasks)}: "
                f"Hb={row['boundary thickness input (m)']:g}, "
                f"Hc={row['center thickness input (m)']:g}, "
                f"{row['status']}",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return add_geometry_diagnostics(table)


def add_geometry_diagnostics(table):
    """Add relief, ellipse radius, and maximum modeled bed slope."""
    table = table.copy()
    boundary = table["boundary thickness input (m)"].astype(float)
    center = table["center thickness input (m)"].astype(float)
    relief = center - boundary
    shape_at_boundary = np.sqrt(
        1.0 - (REFERENCE_HALF_DOMAIN / HORIZONTAL_RADIUS) ** 2
    )
    vertical_radius = relief / (1.0 - shape_at_boundary)
    table["bed relief H_center - H_boundary (m)"] = relief
    table["vertical radius from thicknesses (m)"] = vertical_radius
    table["maximum absolute modeled bed slope"] = (
        vertical_radius
        * REFERENCE_HALF_DOMAIN
        / HORIZONTAL_RADIUS**2
        / shape_at_boundary
    )
    return table


def save_figure(table, path):
    """Plot geometry and the direct minimum-to-individual-exit map."""
    table = add_geometry_diagnostics(table)
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
    minimum = table.pivot(
        index="center thickness input (m)",
        columns="boundary thickness input (m)",
        values=DIRECT_MINIMUM_COLUMN,
    ).reindex(index=center_thicknesses, columns=boundary_thicknesses)
    status_values = status.fillna("").to_numpy(dtype=str)
    success = status_values == "success"
    invalid = np.char.startswith(status_values, "invalid:")
    no_source = status_values == "no surface-deposition source"
    censored = np.char.find(status_values, "censor") >= 0
    failures = np.char.startswith(status_values, "failed:") | (
        status_values == "numerical characteristic failure"
    )
    values = minimum.to_numpy(dtype=float)
    if not np.any(success & np.isfinite(values)):
        raise RuntimeError("direct characteristic map has no finite values")

    diameter = 2.0 * HORIZONTAL_RADIUS
    x_edges = cell_edges(boundary_thicknesses / diameter)
    y_edges = cell_edges(center_thicknesses / diameter)
    diverging = LinearSegmentedColormap.from_list(
        "negative_red_zero_white_positive_blue",
        ["#67001f", "#d6604d", "#ffffff", "#4393c3", "#053061"],
        N=257,
    )
    normalization = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    figure, (schematic_axis, map_axis) = plt.subplots(
        1,
        2,
        figsize=(8.1, 3.75),
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

    def paint(mask, color):
        if np.any(mask):
            map_axis.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_where(~mask, np.ones(mask.shape)),
                cmap=ListedColormap([color]),
                vmin=0.0,
                vmax=1.0,
                shading="flat",
                rasterized=True,
            )

    paint(invalid, "#dedede")
    paint(no_source, "#e7dcec")
    paint(censored, "#f1e6c8")
    paint(failures, "#efd0d0")
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
        map_axis.text(
            0.50,
            0.875,
            "no surface-\ndeposition source",
            color="0.28",
            ha="center",
            va="center",
            fontsize=7.5,
        )
    map_axis.set(
        xlabel=r"Boundary thickness, $H_b/D$",
        ylabel=r"Center thickness, $H_c/D$",
        xlim=(x_edges[0], x_edges[-1]),
        ylim=(y_edges[0], y_edges[-1]),
    )
    map_axis.set_title(
        "B   Minimum along steady paths to individual exit", loc="left"
    )
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.tick_params(direction="in")
    colorbar = figure.colorbar(
        image,
        ax=map_axis,
        orientation="horizontal",
        pad=0.13,
        fraction=0.08,
        aspect=30,
        extend="both",
    )
    colorbar.set_ticks(np.arange(-1.0, 1.01, 0.5))
    colorbar.set_label(
        r"Minimum direct $J=T_x$ (central 80% of depositing sources)"
    )

    classifications = table.loc[table["status"] == "success", CLASSIFICATION_COLUMN]
    count = classifications.value_counts()
    note = (
        f"Resolved overturn: {count.get('resolved overturn', 0)}; "
        f"resolved ordered: {count.get('resolved ordered', 0)}; "
        f"trim-sensitive: {count.get('unresolved: trim-sensitive sign', 0)}; "
        f"|J|≤0.05: {count.get('unresolved: |minimum J|<=0.05', 0)}."
    )
    map_axis.text(
        0.5,
        -0.24,
        note,
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
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--time-step", type=float, default=DEFAULT_TIME_STEP)
    parser.add_argument("--source-count", type=int, default=DEFAULT_SOURCE_COUNT)
    parser.add_argument("--sampler-nx", type=int, default=DEFAULT_SAMPLER_NX)
    parser.add_argument("--sampler-nz", type=int, default=DEFAULT_SAMPLER_NZ)
    parser.add_argument(
        "--maximum-age", type=float, default=DEFAULT_MAXIMUM_AGE
    )
    parser.add_argument("--boundary-thickness", type=float, action="append")
    parser.add_argument("--center-thickness", type=float, action="append")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extend-censored-to",
        type=float,
        help="Rerun only safety-capped CSV rows with this larger age ceiling",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=(
            root / "data" / "steady_characteristic_elliptical_parameter_map.csv"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=(
            root
            / "figures"
            / "steady_characteristic_elliptical_parameter_map.png"
        ),
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.extend_censored_to is not None:
        if not args.csv.exists():
            parser.error("--extend-censored-to requires an existing --csv")
        table = extend_censored_cases(
            pd.read_csv(args.csv),
            args.extend_censored_to,
            jobs=args.jobs,
        )
        table.to_csv(args.csv, index=False, na_rep="NA")
        save_figure(table, args.figure)
        print(f"Wrote {args.csv}", flush=True)
        print(f"Wrote {args.figure}", flush=True)
        return

    if args.plot_only:
        table = pd.read_csv(args.csv)
        save_figure(table, args.figure)
        print(f"Wrote {args.figure}", flush=True)
        return

    boundary_thicknesses = np.sort(
        np.unique(args.boundary_thickness or BOUNDARY_THICKNESSES)
    )
    center_thicknesses = np.sort(
        np.unique(args.center_thickness or CENTER_THICKNESSES)
    )
    existing = pd.read_csv(args.csv) if args.resume and args.csv.exists() else None
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    table = run_grid(
        boundary_thicknesses,
        center_thicknesses,
        args.resolution,
        args.time_step,
        args.source_count,
        TRACED_TRIM_FRACTION,
        args.sampler_nx,
        args.sampler_nz,
        args.maximum_age,
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
