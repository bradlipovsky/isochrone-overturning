"""Steady-tangent phase map for the finite convergent-flow geometry.

For each fixed geometry, the full-Stokes velocity is solved once.  Surface-
origin characteristics then carry the equal-age material tangent

    dX/da = v(X),               dT/da = grad(v)(X) T,

from deposition to their first physical exit.  ``J = T_x`` is therefore the
exact variational horizontal-order diagnostic for the steady velocity field;
no deposited cohort or arbitrary reporting age is used.

The primary map reports the minimum resolved J after an a-priori 5--95%
surface-flux source trim and outside a two-vertical-cell surface-outflow
buffer.  The unbuffered full-residence minimum, its clearance, every exit
status, and safety-cap censoring are retained in the machine-readable table.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import traceback

import firedrake as fd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_DIRECTORY = NOTEBOOK_DIRECTORY.parent
if str(NOTEBOOK_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIRECTORY))

import stagnation_flow_linear_sliding as convergent  # noqa: E402


DATA_PATH = REPOSITORY_DIRECTORY / "data/steady_tangent_convergent_parameter_map.csv"
VALIDATION_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_parameter_map_validation.csv"
)
EULERIAN_AUDIT_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_eulerian_audit.csv"
)
METADATA_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_parameter_map_metadata.json"
)
FIGURE_PATH = (
    REPOSITORY_DIRECTORY
    / "figures/steady_tangent_convergent_single_thickness_diagnostics.png"
)
FIGURE_PDF_PATH = (
    REPOSITORY_DIRECTORY
    / "figures/steady_tangent_convergent_single_thickness_diagnostics.pdf"
)
REPORT_PATH = NOTEBOOK_DIRECTORY / "steady_tangent_convergent_parameter_map.md"

TOTAL_LENGTH = 6_000.0
THICKNESS = 200.0
DIRECT_LENGTH_RATIOS = np.r_[np.arange(1.0, 11.0), 12.0]
LENGTH_RATIOS = np.sort(
    np.r_[1.0 / DIRECT_LENGTH_RATIOS[1:], DIRECT_LENGTH_RATIOS]
)
RELIEF_RATIOS = np.arange(0.25, 4.01, 0.25)
SCREEN_MESH = (180, 24)
SOURCE_COUNT = 81
TIME_STEP_YR = 25.0
MAXIMUM_AGE_YR = 120_000.0
SOURCE_FLUX_QUANTILES = (0.05, 0.95)
PROVENANCE_THRESHOLD = 0.95
DIRECT_SOLVER = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

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


def set_geometry(length_ratio: float, relief_ratio: float) -> None:
    """Set one member of the fixed-span, fixed-thickness geometry family."""
    convergent.LEFT_LENGTH = TOTAL_LENGTH / (1.0 + length_ratio)
    convergent.RIGHT_LENGTH = TOTAL_LENGTH - convergent.LEFT_LENGTH
    convergent.RELIEF = relief_ratio * THICKNESS
    convergent.THICKNESS = THICKNESS


def source_families(
    result: dict,
    source_count: int,
    flux_quantiles: tuple[float, float] = SOURCE_FLUX_QUANTILES,
) -> list[dict]:
    """Return the central 90% surface-flux interval of every source basin."""
    x = result["x"]
    accumulation = result["surface mass balance"]
    intervals = convergent.connected_intervals(accumulation > 1.0e-5)
    families = []
    for family_id, (start, stop) in enumerate(intervals):
        if stop - start < 10:
            continue
        interval = np.arange(start, stop + 1)
        flux = np.maximum(accumulation[interval], 0.0)
        cumulative = np.cumsum(flux)
        if cumulative[-1] <= 0.0:
            continue
        cumulative /= cumulative[-1]
        lower = interval[np.searchsorted(cumulative, flux_quantiles[0])]
        upper = interval[np.searchsorted(cumulative, flux_quantiles[1])]
        sources = np.linspace(x[lower], x[upper], source_count)
        families.append(
            {
                "family id": family_id,
                "source side": "left" if np.mean(sources) < 0.0 else "right",
                "sources": sources,
            }
        )
    return families


def solve_velocity(
    length_ratio: float,
    relief_ratio: float,
    nx: int,
    nz: int,
    sampler_nx: int = 1201,
    sampler_nz: int = 101,
):
    """Solve Stokes flow and prepare the regular variational sampler."""
    set_geometry(length_ratio, relief_ratio)
    mesh, velocity, diagnostics = convergent.solve_stokes(nx, nz)
    sample, geometry, x, sigma, X, Z, values = convergent.make_sampler(
        velocity,
        nx_sample=sampler_nx,
        nz_sample=sampler_nz,
    )
    surface_velocity = geometry["surface_velocity"](x)
    accumulation = (
        surface_velocity[:, 0] * geometry["surface_slope"](x)
        - surface_velocity[:, 1]
    )
    result = {
        "mesh": mesh,
        "velocity": velocity,
        "diagnostics": diagnostics,
        "sample": sample,
        "geometry": geometry,
        "x": x,
        "sigma": sigma,
        "X": X,
        "Z": Z,
        "values": values,
        "surface mass balance": accumulation,
        "sampler nx": sampler_nx,
        "sampler nz": sampler_nz,
    }
    return result


def surface_slope_expression(mesh):
    """Exact UFL derivative of the piecewise-sine surface geometry."""
    x, _ = fd.SpatialCoordinate(mesh)
    left = -(
        0.5
        * np.pi
        * convergent.RELIEF
        / convergent.LEFT_LENGTH
        * fd.sin(np.pi * (-x) / convergent.LEFT_LENGTH)
    )
    right = (
        0.5
        * np.pi
        * convergent.RELIEF
        / convergent.RIGHT_LENGTH
        * fd.sin(np.pi * x / convergent.RIGHT_LENGTH)
    )
    return fd.conditional(x <= 0.0, left, right)


def solve_upwind_scalar(
    velocity,
    *,
    degree: int,
    volume_source,
    surface_inflow_value,
):
    """Solve v.grad(q)=source with data on accumulating surface marker 4."""
    mesh = velocity.function_space().mesh()
    space = fd.FunctionSpace(mesh, "DG", degree)
    trial = fd.TrialFunction(space)
    test = fd.TestFunction(space)
    normal = fd.FacetNormal(mesh)
    normal_velocity = fd.dot(velocity, normal)
    outgoing = 0.5 * (normal_velocity + abs(normal_velocity))
    incoming = 0.5 * (normal_velocity - abs(normal_velocity))
    bilinear = (
        -trial * fd.div(test * velocity) * fd.dx
        + fd.inner(
            outgoing("+") * trial("+") - outgoing("-") * trial("-"),
            fd.jump(test),
        )
        * fd.dS
        + outgoing * trial * test * fd.ds
    )
    linear = (
        volume_source * test * fd.dx
        - surface_inflow_value * incoming * test * fd.ds(4)
    )
    solution = fd.Function(space)
    fd.solve(bilinear == linear, solution, solver_parameters=DIRECT_SOLVER)
    return solution


def solve_stationary_tangent(velocity):
    """Solve v.grad(T)=grad(v)T with T=(1,s_x) at deposition."""
    mesh = velocity.function_space().mesh()
    space = fd.VectorFunctionSpace(mesh, "DG", 1)
    trial = fd.TrialFunction(space)
    test = fd.TestFunction(space)
    normal = fd.FacetNormal(mesh)
    normal_velocity = fd.dot(velocity, normal)
    outgoing = 0.5 * (normal_velocity + abs(normal_velocity))
    incoming = 0.5 * (normal_velocity - abs(normal_velocity))
    bilinear = (
        -fd.inner(trial, fd.div(fd.outer(test, velocity))) * fd.dx
        + fd.inner(
            outgoing("+") * trial("+") - outgoing("-") * trial("-"),
            fd.jump(test),
        )
        * fd.dS
        + fd.inner(outgoing * trial, test) * fd.ds
        - fd.inner(fd.dot(fd.grad(velocity), trial), test) * fd.dx
    )
    surface_tangent = fd.as_vector((1.0, surface_slope_expression(mesh)))
    linear = -incoming * fd.inner(surface_tangent, test) * fd.ds(4)
    tangent = fd.Function(space)
    fd.solve(bilinear == linear, tangent, solver_parameters=DIRECT_SOLVER)
    return tangent


def solve_stationary_fields(velocity):
    """Solve age, provenance, source label, accumulation, and tangent fields."""
    mesh = velocity.function_space().mesh()
    x, _ = fd.SpatialCoordinate(mesh)
    slope = surface_slope_expression(mesh)
    deposition_accumulation_expression = velocity[0] * slope - velocity[1]
    age = solve_upwind_scalar(
        velocity,
        degree=1,
        volume_source=fd.Constant(1.0),
        surface_inflow_value=fd.Constant(0.0),
    )
    provenance = solve_upwind_scalar(
        velocity,
        degree=0,
        volume_source=fd.Constant(0.0),
        surface_inflow_value=fd.Constant(1.0),
    )
    source_x = solve_upwind_scalar(
        velocity,
        degree=1,
        volume_source=fd.Constant(0.0),
        surface_inflow_value=x,
    )
    deposition_accumulation = solve_upwind_scalar(
        velocity,
        degree=1,
        volume_source=fd.Constant(0.0),
        surface_inflow_value=deposition_accumulation_expression,
    )
    tangent = solve_stationary_tangent(velocity)

    cell_space = fd.FunctionSpace(mesh, "DG", 0)
    vector_cell_space = fd.VectorFunctionSpace(mesh, "DG", 0)
    return {
        "cell coordinates": fd.Function(vector_cell_space).interpolate(
            fd.SpatialCoordinate(mesh)
        ),
        "cell velocity": fd.project(velocity, vector_cell_space),
        "cell age": fd.project(age, cell_space),
        "cell provenance": fd.project(provenance, cell_space),
        "cell source x": fd.project(source_x, cell_space),
        "cell accumulation": fd.project(deposition_accumulation, cell_space),
        "cell age z derivative": fd.project(age.dx(1), cell_space),
        "cell age-gradient J": fd.project(
            -deposition_accumulation * age.dx(1), cell_space
        ),
        "cell tangent J": fd.project(tangent[0], cell_space),
    }


def summarize_stationary_fields(
    fields: dict,
    result: dict,
    families: list[dict],
    *,
    outflow_buffer_m: float,
) -> dict:
    """Summarize two source-basin streamtubes without mixing their labels."""
    coordinates = np.asarray(fields["cell coordinates"].dat.data_ro)
    values = {
        name: np.asarray(function.dat.data_ro).reshape(-1)
        for name, function in fields.items()
        if name not in ("cell coordinates", "cell velocity")
    }
    source_ranges = [
        (float(np.min(family["sources"])), float(np.max(family["sources"])))
        for family in families
    ]
    source_label = values["cell source x"]
    basin_masks = [
        (source_label >= lower) & (source_label <= upper)
        for lower, upper in source_ranges
    ]
    source_mask = np.logical_or.reduce(basin_masks)
    surface = convergent.flank_shape(coordinates[:, 0]) + THICKNESS
    clearance = surface - coordinates[:, 1]
    local_mass_balance = np.interp(
        coordinates[:, 0],
        result["x"],
        result["surface mass balance"],
    )
    finite = np.ones(len(coordinates), dtype=bool)
    for value in values.values():
        finite &= np.isfinite(value)
    raw = (
        finite
        & source_mask
        & (values["cell provenance"] >= PROVENANCE_THRESHOLD)
        & (values["cell age"] >= 0.0)
        & (clearance > 0.25)
    )
    # The numerical buffer is needed only below an ablating (outflow) part of
    # the surface.  Accumulating surface cells carry prescribed inflow data.
    resolved = raw & (
        (local_mass_balance >= 0.0) | (clearance >= outflow_buffer_m)
    )
    if not np.any(resolved):
        raise RuntimeError("stationary streamtube mask contains no resolved cells")

    tangent_j = values["cell tangent J"]
    age_j = values["cell age-gradient J"]
    raw_indices = np.flatnonzero(raw)
    resolved_indices = np.flatnonzero(resolved)
    raw_index = int(raw_indices[np.argmin(tangent_j[raw])])
    resolved_index = int(resolved_indices[np.argmin(tangent_j[resolved])])
    identity_error = tangent_j[resolved] - age_j[resolved]
    sign_test = (np.abs(tangent_j[resolved]) > 0.1) & (
        np.abs(age_j[resolved]) > 0.1
    )
    negative = resolved & (tangent_j < 0.0)

    def basin_for_index(index: int) -> int:
        matches = [number for number, mask in enumerate(basin_masks) if mask[index]]
        return matches[0] if matches else -1

    summary = {
        "minimum_resolved_J": float(tangent_j[resolved_index]),
        "resolved_minimum_age_yr": float(values["cell age"][resolved_index]),
        "resolved_minimum_x_m": float(coordinates[resolved_index, 0]),
        "resolved_minimum_z_m": float(coordinates[resolved_index, 1]),
        "resolved_minimum_source_x_m": float(source_label[resolved_index]),
        "resolved_minimum_source_basin": basin_for_index(resolved_index),
        "resolved_minimum_surface_clearance_m": float(clearance[resolved_index]),
        "minimum_raw_J": float(tangent_j[raw_index]),
        "raw_minimum_age_yr": float(values["cell age"][raw_index]),
        "raw_minimum_x_m": float(coordinates[raw_index, 0]),
        "raw_minimum_z_m": float(coordinates[raw_index, 1]),
        "raw_minimum_source_x_m": float(source_label[raw_index]),
        "raw_minimum_source_basin": basin_for_index(raw_index),
        "raw_minimum_surface_clearance_m": float(clearance[raw_index]),
        "first_negative_cell_age_yr": (
            float(np.min(values["cell age"][negative]))
            if np.any(negative)
            else np.nan
        ),
        "stationary_raw_cell_count": int(np.count_nonzero(raw)),
        "stationary_resolved_cell_count": int(np.count_nonzero(resolved)),
        "stationary_negative_J_cell_count": int(np.count_nonzero(negative)),
        "stationary_resolved_J_1st_percentile": float(
            np.quantile(tangent_j[resolved], 0.01)
        ),
        "stationary_oldest_resolved_age_yr": float(
            np.max(values["cell age"][resolved])
        ),
        "stationary_tangent_age_gradient_RMSE": float(
            np.sqrt(np.mean(identity_error**2))
        ),
        "stationary_tangent_age_gradient_correlation": float(
            np.corrcoef(tangent_j[resolved], age_j[resolved])[0, 1]
        ),
        "stationary_tangent_age_gradient_sign_disagreements": int(
            np.count_nonzero(
                sign_test
                & (
                    np.sign(tangent_j[resolved])
                    != np.sign(age_j[resolved])
                )
            )
        ),
        "source_basin_count": len(source_ranges),
        "source_basin_ranges_m": "; ".join(
            f"{lower:.6g}:{upper:.6g}" for lower, upper in source_ranges
        ),
    }
    for basin_id, mask in enumerate(basin_masks):
        basin_resolved = resolved & mask
        summary[f"source_basin_{basin_id}_resolved_cell_count"] = int(
            np.count_nonzero(basin_resolved)
        )
        summary[f"source_basin_{basin_id}_minimum_resolved_J"] = (
            float(np.min(tangent_j[basin_resolved]))
            if np.any(basin_resolved)
            else np.nan
        )
    return summary


def rk4_step(sample, gradient, points, tangents, step):
    """Advance positions and their tangent sensitivities by one RK4 step."""
    k1 = sample(points)
    f1 = np.einsum("nij,nj->ni", gradient(points), tangents)
    p2 = points + 0.5 * step * k1
    t2 = tangents + 0.5 * step * f1
    k2 = sample(p2)
    f2 = np.einsum("nij,nj->ni", gradient(p2), t2)
    p3 = points + 0.5 * step * k2
    t3 = tangents + 0.5 * step * f2
    k3 = sample(p3)
    f3 = np.einsum("nij,nj->ni", gradient(p3), t3)
    p4 = points + step * k3
    t4 = tangents + step * f3
    k4 = sample(p4)
    f4 = np.einsum("nij,nj->ni", gradient(p4), t4)
    return (
        points + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
        tangents + step * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0,
    )


def local_sigma(points, geometry):
    """Bed-following vertical coordinate for one point array."""
    bed = geometry["bed"](points[:, 0])
    surface = geometry["surface"](points[:, 0])
    return (points[:, 1] - bed) / (surface - bed)


def first_exit_fraction(old_points, new_points, old_sigma, new_sigma):
    """Linearly locate the first sampled-domain boundary crossed in a step."""
    count = len(old_points)
    fraction = np.ones(count)
    kind = np.full(count, "numerical failure", dtype=object)
    candidates = (
        (
            "left divide",
            old_points[:, 0],
            new_points[:, 0],
            -convergent.LEFT_LENGTH + 0.5,
            new_points[:, 0] <= -convergent.LEFT_LENGTH + 0.5,
        ),
        (
            "right divide",
            old_points[:, 0],
            new_points[:, 0],
            convergent.RIGHT_LENGTH - 0.5,
            new_points[:, 0] >= convergent.RIGHT_LENGTH - 0.5,
        ),
        ("bed", old_sigma, new_sigma, 0.0022, new_sigma <= 0.0022),
        ("surface", old_sigma, new_sigma, 0.998, new_sigma >= 0.998),
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


def trace_steady_family(
    result: dict,
    family: dict,
    *,
    time_step: float,
    maximum_age: float,
    outflow_buffer_m: float,
) -> pd.DataFrame:
    """Carry T along a steady source family and summarize each physical path."""
    sources = family["sources"]
    geometry = result["geometry"]
    sample = result["sample"]
    gradient = geometry["velocity_gradient"]
    bed = geometry["bed"](sources)
    surface = geometry["surface"](sources)
    points = np.column_stack((sources, bed + 0.995 * (surface - bed)))
    source_slope = (
        geometry["bed_slope"](sources)
        + 0.995
        * (geometry["surface_slope"](sources) - geometry["bed_slope"](sources))
    )
    tangents = np.column_stack((np.ones(len(sources)), source_slope))
    source_velocity = geometry["surface_velocity"](sources)
    source_accumulation = (
        source_velocity[:, 0] * geometry["surface_slope"](sources)
        - source_velocity[:, 1]
    )

    count = len(sources)
    active = np.ones(count, dtype=bool)
    age = np.zeros(count)
    raw_minimum = tangents[:, 0].copy()
    raw_minimum_age = np.zeros(count)
    raw_minimum_clearance = surface - points[:, 1]
    resolved_minimum = np.full(count, np.inf)
    resolved_minimum_age = np.full(count, np.nan)
    resolved_minimum_clearance = np.full(count, np.nan)
    first_zero_age = np.full(count, np.nan)
    exit_age = np.full(count, np.nan)
    exit_kind = np.full(count, "active", dtype=object)
    velocity = sample(points)
    previous_u = velocity[:, 0]
    minimum_u = previous_u.copy()
    maximum_u = previous_u.copy()
    reversal_count = np.zeros(count, dtype=int)

    for _ in range(int(np.ceil(maximum_age / time_step))):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        old_points = points[indices].copy()
        old_tangents = tangents[indices].copy()
        old_age = age[indices].copy()
        old_sigma = local_sigma(old_points, geometry)
        candidate_points, candidate_tangents = rk4_step(
            sample,
            gradient,
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
            & (candidate_sigma > 0.0022)
            & (candidate_sigma < 0.998)
            & (candidate_points[:, 0] > -convergent.LEFT_LENGTH + 0.5)
            & (candidate_points[:, 0] < convergent.RIGHT_LENGTH - 0.5)
        )

        accepted_points = candidate_points.copy()
        accepted_tangents = candidate_tangents.copy()
        accepted_age = old_age + time_step
        step_exit_kind = np.full(len(indices), "", dtype=object)
        exiting = ~remains
        if np.any(exiting):
            exit_fraction, local_kind = first_exit_fraction(
                old_points[exiting],
                candidate_points[exiting],
                old_sigma[exiting],
                candidate_sigma[exiting],
            )
            accepted_points[exiting] = old_points[exiting] + exit_fraction[:, None] * (
                candidate_points[exiting] - old_points[exiting]
            )
            accepted_tangents[exiting] = old_tangents[exiting] + exit_fraction[:, None] * (
                candidate_tangents[exiting] - old_tangents[exiting]
            )
            accepted_age[exiting] = old_age[exiting] + exit_fraction * time_step
            step_exit_kind[exiting] = local_kind

        accepted_sigma = local_sigma(accepted_points, geometry)
        accepted_surface = geometry["surface"](accepted_points[:, 0])
        clearance = accepted_surface - accepted_points[:, 1]
        accepted_j = accepted_tangents[:, 0]
        old_j = old_tangents[:, 0]

        crossed_zero = (
            np.isnan(first_zero_age[indices])
            & (old_j > 0.0)
            & (accepted_j <= 0.0)
        )
        if np.any(crossed_zero):
            fraction = old_j[crossed_zero] / (
                old_j[crossed_zero] - accepted_j[crossed_zero]
            )
            first_zero_age[indices[crossed_zero]] = old_age[crossed_zero] + fraction * (
                accepted_age[crossed_zero] - old_age[crossed_zero]
            )

        improves_raw = accepted_j < raw_minimum[indices]
        if np.any(improves_raw):
            selected = indices[improves_raw]
            raw_minimum[selected] = accepted_j[improves_raw]
            raw_minimum_age[selected] = accepted_age[improves_raw]
            raw_minimum_clearance[selected] = clearance[improves_raw]

        local_mass_balance = np.interp(
            accepted_points[:, 0],
            result["x"],
            result["surface mass balance"],
        )
        resolved = (local_mass_balance >= 0.0) | (
            clearance >= outflow_buffer_m
        )
        improves_resolved = resolved & (accepted_j < resolved_minimum[indices])
        if np.any(improves_resolved):
            selected = indices[improves_resolved]
            resolved_minimum[selected] = accepted_j[improves_resolved]
            resolved_minimum_age[selected] = accepted_age[improves_resolved]
            resolved_minimum_clearance[selected] = clearance[improves_resolved]

        accepted_velocity = sample(accepted_points)
        accepted_u = accepted_velocity[:, 0]
        minimum_u[indices] = np.minimum(minimum_u[indices], accepted_u)
        maximum_u[indices] = np.maximum(maximum_u[indices], accepted_u)
        u_scale = np.maximum(
            np.maximum(np.abs(previous_u[indices]), np.abs(accepted_u)),
            1.0e-12,
        )
        reversal = (
            np.signbit(previous_u[indices]) != np.signbit(accepted_u)
        ) & (np.minimum(np.abs(previous_u[indices]), np.abs(accepted_u)) > 1.0e-6 * u_scale)
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
    resolved_minimum[~np.isfinite(resolved_minimum)] = np.nan
    rows = pd.DataFrame(
        {
            "family_id": family["family id"],
            "source_side": family["source side"],
            "source_x_m": sources,
            "source_accumulation_m_per_yr": source_accumulation,
            "raw_minimum_J": raw_minimum,
            "raw_minimum_age_yr": raw_minimum_age,
            "raw_minimum_surface_clearance_m": raw_minimum_clearance,
            "resolved_minimum_J": resolved_minimum,
            "resolved_minimum_age_yr": resolved_minimum_age,
            "resolved_minimum_surface_clearance_m": resolved_minimum_clearance,
            "first_J_zero_age_yr": first_zero_age,
            "minimum_path_u_m_per_yr": minimum_u,
            "maximum_path_u_m_per_yr": maximum_u,
            "horizontal_velocity_reversal_count": reversal_count,
            "exit_age_yr": exit_age,
            "exit_kind": exit_kind,
        }
    )
    return rows


def eulerian_flow_metrics(result: dict) -> dict:
    """Measure counter-slope pockets in the solved velocity field itself."""
    X = result["X"]
    u = result["values"][..., 0]
    away_from_join = np.abs(X) >= 0.005 * TOTAL_LENGTH
    counter_slope = away_from_join & (
        ((X < 0.0) & (u < 0.0)) | ((X > 0.0) & (u > 0.0))
    )
    maximum_abs_u = float(np.nanmax(np.abs(u)))
    significant = counter_slope & (
        np.abs(u) >= 0.01 * max(maximum_abs_u, 1.0e-12)
    )
    return {
        "eulerian_maximum_counterflow_speed_m_per_yr": (
            float(np.nanmax(np.abs(u[counter_slope])))
            if np.any(counter_slope)
            else 0.0
        ),
        "eulerian_significant_counterflow_area_fraction": float(
            np.count_nonzero(significant) / np.count_nonzero(away_from_join)
        ),
        "eulerian_counterflow_speed_threshold_m_per_yr": (
            0.01 * maximum_abs_u
        ),
    }


def add_derived_geometry_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add exact maximum short-flank slope and its corresponding angle."""
    table = table.copy()
    short_length = np.minimum(table["left_length_m"], table["right_length_m"])
    slope = 0.5 * np.pi * table["relief_m"] / short_length
    table["maximum_short_flank_slope"] = slope
    table["maximum_short_flank_angle_degrees"] = np.degrees(np.arctan(slope))
    return table


def summarize_characteristic_case(
    result: dict,
    trajectories: pd.DataFrame,
    *,
    length_ratio: float,
    relief_ratio: float,
    nx: int,
    nz: int,
    source_count: int,
    time_step: float,
    source_flux_quantiles: tuple[float, float],
    outflow_buffer_m: float,
    sampler_nx: int,
    sampler_nz: int,
    runtime: float,
) -> dict:
    """Reduce steady path results to an independent validation row."""
    diagnostics = result["diagnostics"]
    outflow_buffer = outflow_buffer_m
    finite_resolved = trajectories["resolved_minimum_J"].replace(
        [np.inf, -np.inf], np.nan
    )
    if finite_resolved.notna().any():
        resolved_index = finite_resolved.idxmin()
        resolved_row = trajectories.loc[resolved_index]
    else:
        resolved_row = None
    raw_index = trajectories["raw_minimum_J"].idxmin()
    raw_row = trajectories.loc[raw_index]
    safety_count = int(np.count_nonzero(trajectories["exit_kind"] == "safety cap"))
    failure_count = int(
        np.count_nonzero(trajectories["exit_kind"] == "numerical failure")
    )
    energy_ok = abs(diagnostics["relative_energy_residual"]) < 1.0e-3
    basal_ok = diagnostics["rms_basal_normal_speed"] < 1.0e-3
    valid = (
        resolved_row is not None
        and safety_count == 0
        and failure_count == 0
        and energy_ok
        and basal_ok
    )
    status = "valid"
    if safety_count:
        status = "censored_safety_cap"
    elif failure_count:
        status = "numerical_failure"
    elif not energy_ok or not basal_ok:
        status = "failed_stokes_diagnostic"
    elif resolved_row is None:
        status = "no_resolved_samples"
    resolved_minimum_value = (
        float(resolved_row["resolved_minimum_J"])
        if resolved_row is not None
        else np.nan
    )
    if resolved_minimum_value < -0.05:
        resolved_classification = "overturned"
    elif resolved_minimum_value > 0.05:
        resolved_classification = "ordered"
    else:
        resolved_classification = "unresolved"

    accumulation = result["surface mass balance"]
    x = result["x"]
    intervals = convergent.connected_intervals(accumulation > 1.0e-5)
    first_zero = trajectories["first_J_zero_age_yr"].min(skipna=True)
    active_at_raw_min = int(
        np.count_nonzero(trajectories["exit_age_yr"] >= raw_row["raw_minimum_age_yr"])
    )
    active_at_resolved_min = (
        int(
            np.count_nonzero(
                trajectories["exit_age_yr"]
                >= resolved_row["resolved_minimum_age_yr"]
            )
        )
        if resolved_row is not None
        else 0
    )
    summary = {
        "right_to_left_length_ratio": length_ratio,
        "relief_to_thickness_ratio": relief_ratio,
        "left_length_m": convergent.LEFT_LENGTH,
        "right_length_m": convergent.RIGHT_LENGTH,
        "relief_m": convergent.RELIEF,
        "thickness_m": THICKNESS,
        "total_length_m": TOTAL_LENGTH,
        "nx": nx,
        "nz": nz,
        "source_count_per_family": source_count,
        "source_flux_lower_quantile": source_flux_quantiles[0],
        "source_flux_upper_quantile": source_flux_quantiles[1],
        "time_step_yr": time_step,
        "sampler_nx": sampler_nx,
        "sampler_nz": sampler_nz,
        "maximum_age_safety_cap_yr": MAXIMUM_AGE_YR,
        "surface_outflow_buffer_m": outflow_buffer,
        "solution_provenance": "direct_solve",
        "valid": bool(valid),
        "status": status,
        "minimum_resolved_J": resolved_minimum_value,
        "resolved_per_source_minimum_J_1st_percentile": float(
            finite_resolved.dropna().quantile(0.01)
        ),
        "resolved_classification": resolved_classification,
        "resolved_minimum_age_yr": (
            float(resolved_row["resolved_minimum_age_yr"])
            if resolved_row is not None
            else np.nan
        ),
        "resolved_minimum_source_x_m": (
            float(resolved_row["source_x_m"])
            if resolved_row is not None
            else np.nan
        ),
        "resolved_minimum_source_side": (
            resolved_row["source_side"] if resolved_row is not None else ""
        ),
        "resolved_minimum_surface_clearance_m": (
            float(resolved_row["resolved_minimum_surface_clearance_m"])
            if resolved_row is not None
            else np.nan
        ),
        "active_sources_at_resolved_minimum": active_at_resolved_min,
        "minimum_raw_J": float(raw_row["raw_minimum_J"]),
        "raw_per_source_minimum_J_1st_percentile": float(
            trajectories["raw_minimum_J"].quantile(0.01)
        ),
        "raw_minimum_age_yr": float(raw_row["raw_minimum_age_yr"]),
        "raw_minimum_source_x_m": float(raw_row["source_x_m"]),
        "raw_minimum_source_side": raw_row["source_side"],
        "raw_minimum_surface_clearance_m": float(
            raw_row["raw_minimum_surface_clearance_m"]
        ),
        "active_sources_at_raw_minimum": active_at_raw_min,
        "first_J_zero_age_yr": float(first_zero) if np.isfinite(first_zero) else np.nan,
        "trajectory_count": int(len(trajectories)),
        "safety_cap_trajectory_count": safety_count,
        "numerical_failure_trajectory_count": failure_count,
        "surface_exit_count": int(np.count_nonzero(trajectories["exit_kind"] == "surface")),
        "bed_exit_count": int(np.count_nonzero(trajectories["exit_kind"] == "bed")),
        "left_divide_exit_count": int(
            np.count_nonzero(trajectories["exit_kind"] == "left divide")
        ),
        "right_divide_exit_count": int(
            np.count_nonzero(trajectories["exit_kind"] == "right divide")
        ),
        "maximum_exit_age_yr": float(trajectories["exit_age_yr"].max()),
        "minimum_path_u_m_per_yr": float(
            trajectories["minimum_path_u_m_per_yr"].min()
        ),
        "maximum_path_u_m_per_yr": float(
            trajectories["maximum_path_u_m_per_yr"].max()
        ),
        "counterflow_trajectory_count": int(
            np.count_nonzero(
                (
                    (trajectories["source_side"] == "left")
                    & (trajectories["minimum_path_u_m_per_yr"] < -1.0e-5)
                )
                | (
                    (trajectories["source_side"] == "right")
                    & (trajectories["maximum_path_u_m_per_yr"] > 1.0e-5)
                )
            )
        ),
        "maximum_counterflow_speed_m_per_yr": float(
            max(
                (
                    -trajectories.loc[
                        trajectories["source_side"] == "left",
                        "minimum_path_u_m_per_yr",
                    ].min()
                    if np.any(trajectories["source_side"] == "left")
                    else 0.0
                ),
                (
                    trajectories.loc[
                        trajectories["source_side"] == "right",
                        "maximum_path_u_m_per_yr",
                    ].max()
                    if np.any(trajectories["source_side"] == "right")
                    else 0.0
                ),
                0.0,
            )
        ),
        "maximum_horizontal_velocity_reversal_count": int(
            trajectories["horizontal_velocity_reversal_count"].max()
        ),
        "accumulation_interval_count": len(intervals),
        "peak_required_ablation_m_per_yr": float(max(-np.min(accumulation), 0.0)),
        "peak_accumulation_m_per_yr": float(max(np.max(accumulation), 0.0)),
        "integrated_positive_mass_balance_m2_per_yr": float(
            np.trapz(np.maximum(accumulation, 0.0), x)
        ),
        "integrated_negative_mass_balance_m2_per_yr": float(
            np.trapz(np.minimum(accumulation, 0.0), x)
        ),
        "relative_energy_residual": float(diagnostics["relative_energy_residual"]),
        "rms_basal_normal_speed_m_per_yr": float(
            diagnostics["rms_basal_normal_speed"]
        ),
        "runtime_s": runtime,
        "error": "",
    }
    summary.update(eulerian_flow_metrics(result))
    short_length = min(convergent.LEFT_LENGTH, convergent.RIGHT_LENGTH)
    short_slope = 0.5 * np.pi * convergent.RELIEF / short_length
    summary["maximum_short_flank_slope"] = short_slope
    summary["maximum_short_flank_angle_degrees"] = float(
        np.degrees(np.arctan(short_slope))
    )
    return summary


def run_case(
    length_ratio: float,
    relief_ratio: float,
    nx: int,
    nz: int,
    source_count: int = SOURCE_COUNT,
    time_step: float = TIME_STEP_YR,
    outflow_buffer_m: float | None = None,
    source_flux_quantiles: tuple[float, float] = SOURCE_FLUX_QUANTILES,
    sampler_nx: int = 1201,
    sampler_nz: int = 101,
) -> tuple[dict, pd.DataFrame | None]:
    """Solve one steady geometry and retain failures as explicit rows."""
    started = time.perf_counter()
    set_geometry(length_ratio, relief_ratio)
    if outflow_buffer_m is None:
        outflow_buffer_m = 2.0 * THICKNESS / nz
    base = {
        "right_to_left_length_ratio": length_ratio,
        "relief_to_thickness_ratio": relief_ratio,
        "left_length_m": convergent.LEFT_LENGTH,
        "right_length_m": convergent.RIGHT_LENGTH,
        "relief_m": convergent.RELIEF,
        "thickness_m": THICKNESS,
        "total_length_m": TOTAL_LENGTH,
        "nx": nx,
        "nz": nz,
        "source_count_per_family": source_count,
        "time_step_yr": time_step,
        "surface_outflow_buffer_m": outflow_buffer_m,
        "source_flux_lower_quantile": source_flux_quantiles[0],
        "source_flux_upper_quantile": source_flux_quantiles[1],
        "sampler_nx": sampler_nx,
        "sampler_nz": sampler_nz,
        "solution_provenance": "direct_solve",
        "valid": False,
        "status": "not_run",
        "error": "",
    }
    try:
        result = solve_velocity(
            length_ratio,
            relief_ratio,
            nx,
            nz,
            sampler_nx=sampler_nx,
            sampler_nz=sampler_nz,
        )
        families = source_families(
            result,
            source_count,
            flux_quantiles=source_flux_quantiles,
        )
        if not families:
            raise RuntimeError("no resolved accumulating surface source interval")
        paths = []
        for family in families:
            paths.append(
                trace_steady_family(
                    result,
                    family,
                    time_step=time_step,
                    maximum_age=MAXIMUM_AGE_YR,
                    outflow_buffer_m=outflow_buffer_m,
                )
            )
        trajectories = pd.concat(paths, ignore_index=True)
        row = summarize_characteristic_case(
            result,
            trajectories,
            length_ratio=length_ratio,
            relief_ratio=relief_ratio,
            nx=nx,
            nz=nz,
            source_count=source_count,
            time_step=time_step,
            source_flux_quantiles=source_flux_quantiles,
            outflow_buffer_m=outflow_buffer_m,
            sampler_nx=sampler_nx,
            sampler_nz=sampler_nz,
            runtime=time.perf_counter() - started,
        )
        return row, trajectories
    except Exception as error:
        base.update(
            {
                "status": type(error).__name__,
                "runtime_s": time.perf_counter() - started,
                "error": str(error),
                "minimum_resolved_J": np.nan,
                "minimum_raw_J": np.nan,
            }
        )
        traceback.print_exc()
        return base, None


def screen_worker(arguments: tuple[float, float, int, int]) -> dict:
    """Process-safe screen wrapper with one point-location retry."""
    q, relief, nx, nz = arguments
    row, _ = run_case(q, relief, nx, nz)
    if row["status"] == "PointNotInDomainError":
        row, _ = run_case(q, relief, nx + 1, nz)
        if row.get("valid", False):
            row["solution_provenance"] = "direct_solve_mesh_edge_retry"
    return row


def reflected_row(row: dict) -> dict:
    """Reflect a q>1 result to the exactly equivalent reciprocal geometry."""
    reflected = dict(row)
    q = float(row["right_to_left_length_ratio"])
    reflected["right_to_left_length_ratio"] = 1.0 / q
    reflected["left_length_m"] = row["right_length_m"]
    reflected["right_length_m"] = row["left_length_m"]
    reflected["solution_provenance"] = "exact_horizontal_reflection"
    for column in (
        "resolved_minimum_source_x_m",
        "raw_minimum_source_x_m",
        "resolved_minimum_x_m",
        "raw_minimum_x_m",
    ):
        reflected[column] = -row.get(column, np.nan)
    side_swap = {"left": "right", "right": "left"}
    for column in ("resolved_minimum_source_side", "raw_minimum_source_side"):
        reflected[column] = side_swap.get(row.get(column), row.get(column))
    reflected["left_divide_exit_count"] = row.get("right_divide_exit_count", 0)
    reflected["right_divide_exit_count"] = row.get("left_divide_exit_count", 0)
    minimum_u = row.get("minimum_path_u_m_per_yr", np.nan)
    maximum_u = row.get("maximum_path_u_m_per_yr", np.nan)
    reflected["minimum_path_u_m_per_yr"] = -maximum_u
    reflected["maximum_path_u_m_per_yr"] = -minimum_u
    if row.get("source_basin_count") == 2:
        for column in (
            "resolved_minimum_source_basin",
            "raw_minimum_source_basin",
        ):
            value = row.get(column, np.nan)
            reflected[column] = 1 - int(value) if np.isfinite(value) else value
        for suffix in ("resolved_cell_count", "minimum_resolved_J"):
            left = row.get(f"source_basin_0_{suffix}", np.nan)
            right = row.get(f"source_basin_1_{suffix}", np.nan)
            reflected[f"source_basin_0_{suffix}"] = right
            reflected[f"source_basin_1_{suffix}"] = left
        ranges = []
        for item in str(row.get("source_basin_ranges_m", "")).split("; "):
            if ":" not in item:
                continue
            lower, upper = (float(value) for value in item.split(":"))
            ranges.append((-upper, -lower))
        reflected["source_basin_ranges_m"] = "; ".join(
            f"{lower:.6g}:{upper:.6g}" for lower, upper in sorted(ranges)
        )
    return reflected


def run_screen(workers: int) -> pd.DataFrame:
    """Solve q>=1 and fill q<1 by exact horizontal reflection."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = [
        (q, relief, SCREEN_MESH[0], SCREEN_MESH[1])
        for relief in RELIEF_RATIOS
        for q in DIRECT_LENGTH_RATIOS
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(screen_worker, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows.append(row)
            if case[0] > 1.0:
                rows.append(reflected_row(row))
            table = pd.DataFrame(rows).sort_values(
                ["relief_to_thickness_ratio", "right_to_left_length_ratio"]
            )
            table.to_csv(DATA_PATH, index=False)
            print(
                "STEADY MAP COMPLETE "
                f"q={case[0]:g}, relief/H={case[1]:g}: {row['status']}, "
                f"resolved J={row.get('minimum_resolved_J', np.nan):.5g}, "
                f"raw J={row.get('minimum_raw_J', np.nan):.5g}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(
        ["relief_to_thickness_ratio", "right_to_left_length_ratio"]
    )


def run_validation() -> pd.DataFrame:
    """Test mesh, buffer, source trim, sampler, source count, and step size."""
    rows = []
    fixed_clearance = 2.0 * THICKNESS / SCREEN_MESH[1]
    cases = (
        ("transition ordered", 7.0, 2.25),
        ("transition overturned", 7.0, 2.50),
        ("counterflow reciprocal left", 0.125, 3.50),
        ("counterflow reciprocal right", 8.0, 3.50),
        ("extended extreme", 12.0, 4.00),
    )
    for name, q, relief in cases:
        variants = []
        meshes = (
            ((180, 24), (240, 32), (300, 40))
            if name == "extended extreme"
            else ((120, 16), (180, 24), (240, 32))
        )
        for nx, nz in meshes:
            variants.append(
                {
                    "check": "mesh at fixed clearance",
                    "nx": nx,
                    "nz": nz,
                    "source_count": 81,
                    "time_step": 25.0,
                    "outflow_buffer_m": fixed_clearance,
                    "source_flux_quantiles": (0.05, 0.95),
                    "sampler_nx": 1201,
                    "sampler_nz": 101,
                }
            )
        variants.extend(
            (
                {
                    "check": "source count",
                    "source_count": 41,
                },
                {
                    "check": "source count",
                    "source_count": 81,
                },
                {
                    "check": "source count",
                    "source_count": 161,
                },
                {
                    "check": "time step",
                    "source_count": 161,
                    "time_step": 50.0,
                },
                {
                    "check": "time step",
                    "source_count": 161,
                    "time_step": 25.0,
                },
                {
                    "check": "time step",
                    "source_count": 161,
                    "time_step": 12.5,
                },
                {
                    "check": "outflow clearance",
                    "source_count": 161,
                    "outflow_buffer_m": 10.0,
                },
                {
                    "check": "outflow clearance",
                    "source_count": 161,
                    "outflow_buffer_m": fixed_clearance,
                },
                {
                    "check": "outflow clearance",
                    "source_count": 161,
                    "outflow_buffer_m": 25.0,
                },
                {
                    "check": "source flux trim",
                    "source_count": 161,
                    "source_flux_quantiles": (0.01, 0.99),
                },
                {
                    "check": "source flux trim",
                    "source_count": 161,
                    "source_flux_quantiles": (0.05, 0.95),
                },
                {
                    "check": "source flux trim",
                    "source_count": 161,
                    "source_flux_quantiles": (0.10, 0.90),
                },
            )
        )
        if name in ("transition overturned", "extended extreme"):
            variants.extend(
                (
                    {
                        "check": "velocity sampler",
                        "source_count": 161,
                        "sampler_nx": 1201,
                        "sampler_nz": 101,
                    },
                    {
                        "check": "velocity sampler",
                        "source_count": 161,
                        "sampler_nx": 1801,
                        "sampler_nz": 151,
                    },
                )
            )

        defaults = {
            "nx": 180,
            "nz": 24,
            "source_count": 81,
            "time_step": 25.0,
            "outflow_buffer_m": fixed_clearance,
            "source_flux_quantiles": (0.05, 0.95),
            "sampler_nx": 1201,
            "sampler_nz": 101,
        }
        for variant in variants:
            parameters = defaults | variant
            check = parameters.pop("check")
            row, _ = run_case(q, relief, **parameters)
            if row.get("status") == "PointNotInDomainError":
                retry_parameters = dict(parameters)
                retry_parameters["nx"] += 1
                row, _ = run_case(q, relief, **retry_parameters)
                if row.get("valid", False):
                    row["solution_provenance"] = (
                        "direct_solve_validation_mesh_edge_retry"
                    )
            if row.get("status") == "PointNotInDomainError":
                retry_parameters = dict(parameters)
                retry_parameters["nx"] = max(2, retry_parameters["nx"] - 1)
                row, _ = run_case(q, relief, **retry_parameters)
                if row.get("valid", False):
                    row["solution_provenance"] = (
                        "direct_solve_validation_mesh_edge_retry"
                    )
            if (
                row.get("status") == "PointNotInDomainError"
                and q < 1.0
            ):
                reciprocal_row, _ = run_case(
                    1.0 / q,
                    relief,
                    **parameters,
                )
                if reciprocal_row.get("valid", False):
                    row = reflected_row(reciprocal_row)
                    row["solution_provenance"] = (
                        "exact_horizontal_reflection_validation_retry"
                    )
            if (
                row.get("status") == "PointNotInDomainError"
                and name == "extended extreme"
            ):
                retry_parameters = dict(parameters)
                retry_parameters.update({"nx": 300, "nz": 40})
                row, _ = run_case(q, relief, **retry_parameters)
                if row.get("valid", False):
                    row["solution_provenance"] = (
                        "direct_solve_alternate_fine_mesh_after_sampling_edge_failure"
                    )
            row.update({"case": name, "check": check})
            rows.append(row)
            pd.DataFrame(rows).to_csv(VALIDATION_PATH, index=False)
            print(
                f"VALIDATION {name}, {check}: q={q:g}, relief/H={relief:g}, "
                f"mesh={row['nx']}x{row['nz']}, "
                f"sources={row['source_count_per_family']}, "
                f"dt={row['time_step_yr']:g}, "
                f"clearance={row['surface_outflow_buffer_m']:g}, "
                f"J={row.get('minimum_resolved_J', np.nan):.5g}, "
                f"p01={row.get('resolved_per_source_minimum_J_1st_percentile', np.nan):.5g}",
                flush=True,
            )
    table = pd.DataFrame(rows).sort_values(
        [
            "case",
            "check",
            "nx",
            "nz",
            "source_count_per_family",
            "time_step_yr",
            "surface_outflow_buffer_m",
            "source_flux_lower_quantile",
            "sampler_nx",
            "sampler_nz",
        ]
    )
    table.to_csv(VALIDATION_PATH, index=False)
    return table


def run_eulerian_audit() -> pd.DataFrame:
    """Show why a global stationary DG map is rejected near the separatrix.

    Unlike the single-basin elliptical geometry, this convergent domain has an
    internal source-basin separatrix.  Upwind DG mixes the two transported
    tangent branches in its compressed neighborhood.  These spot checks are
    retained as a falsification audit, not used in the production heatmap.
    """
    rows = []
    cases = (
        ("characteristic ordered", 7.0, 2.25),
        ("transition", 7.0, 2.50),
        ("strong counterflow", 8.0, 3.50),
    )
    for name, q, relief in cases:
        started = time.perf_counter()
        set_geometry(q, relief)
        result = solve_velocity(q, relief, SCREEN_MESH[0], SCREEN_MESH[1])
        families = source_families(result, SOURCE_COUNT)
        fields = solve_stationary_fields(result["velocity"])
        row = summarize_stationary_fields(
            fields,
            result,
            families,
            outflow_buffer_m=2.0 * THICKNESS / SCREEN_MESH[1],
        )
        row.update(
            {
                "case": name,
                "right_to_left_length_ratio": q,
                "relief_to_thickness_ratio": relief,
                "nx": SCREEN_MESH[0],
                "nz": SCREEN_MESH[1],
                "runtime_s": time.perf_counter() - started,
                "accepted_for_phase_map": False,
                "rejection_reason": (
                    "source-basin separatrix compression contaminates "
                    "stationary DG tangent extrema"
                ),
            }
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(EULERIAN_AUDIT_PATH, index=False)
        print(
            f"EULERIAN AUDIT {name}: min={row['minimum_resolved_J']:.5g}, "
            f"p01={row['stationary_resolved_J_1st_percentile']:.5g}",
            flush=True,
        )
    return pd.DataFrame(rows)


def coordinate_edges(values: np.ndarray, logarithmic: bool = False) -> np.ndarray:
    """Cell edges for linearly or logarithmically displayed centers."""
    if logarithmic:
        middle = np.sqrt(values[:-1] * values[1:])
        return np.r_[values[0] ** 2 / middle[0], middle, values[-1] ** 2 / middle[-1]]
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (middle[0] - values[0]), middle, values[-1] + (values[-1] - middle[-1])]


def draw_geometry_schematic(axis: plt.Axes) -> None:
    """Draw the exact piecewise-sine geometry and define all four parameters."""
    left_length = 1.0
    right_length = 2.5
    relief = 0.60
    thickness = 0.72
    x = np.linspace(-left_length, right_length, 801)
    bed = np.where(
        x <= 0.0,
        relief * np.sin(0.5 * np.pi * (-x) / left_length) ** 2,
        relief * np.sin(0.5 * np.pi * x / right_length) ** 2,
    )
    surface = bed + thickness
    axis.fill_between(x, -0.20, bed, color="0.72", zorder=1)
    axis.fill_between(x, bed, surface, color="#D9EAF4", zorder=2)
    axis.plot(x, bed, color="0.12", linewidth=1.4, zorder=3)
    axis.plot(x, surface, color="#2C6E9B", linewidth=1.4, zorder=3)
    arrow = {"arrowstyle": "<->", "color": "0.12", "linewidth": 1.0, "shrinkA": 0, "shrinkB": 0}
    guide = {"color": "0.45", "linewidth": 0.75, "linestyle": ":"}

    dimension_y = -0.26
    for location, top in ((-left_length, relief), (0.0, 0.0), (right_length, relief)):
        axis.plot((location, location), (dimension_y - 0.02, top), **guide)
    axis.annotate("", xy=(0, dimension_y), xytext=(-left_length, dimension_y), arrowprops=arrow)
    axis.annotate("", xy=(right_length, dimension_y), xytext=(0, dimension_y), arrowprops=arrow)
    axis.text(-0.5 * left_length, dimension_y - 0.07, r"$L_{\rm left}$", ha="center", va="top")
    axis.text(0.5 * right_length, dimension_y - 0.07, r"$L_{\rm right}$", ha="center", va="top")

    relief_x = -left_length - 0.20
    axis.plot((relief_x, -left_length), (0, 0), **guide)
    axis.plot((relief_x, -left_length), (relief, relief), **guide)
    axis.annotate("", xy=(relief_x, relief), xytext=(relief_x, 0), arrowprops=arrow)
    axis.text(relief_x - 0.06, 0.5 * relief, r"$\Delta b$", ha="right", va="center")

    thickness_x = 0.32
    local_bed = float(np.interp(thickness_x, x, bed))
    axis.annotate(
        "",
        xy=(thickness_x, local_bed + thickness),
        xytext=(thickness_x, local_bed),
        arrowprops=arrow,
    )
    axis.text(thickness_x + 0.09, local_bed + 0.5 * thickness, r"$H$", ha="left", va="center")
    axis.text(
        0.5,
        0.025,
        r"$q=L_{\rm right}/L_{\rm left}$; vertical scale exaggerated",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        color="0.35",
        style="italic",
    )
    axis.set(xlim=(-1.43, 2.68), ylim=(-0.72, 1.55), title="(a) Geometry and parameters")
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def make_figure(table: pd.DataFrame) -> None:
    """Plot geometry, overturn, ablation, and Eulerian counterflow."""
    ratios = np.sort(table["right_to_left_length_ratio"].unique())
    reliefs = np.sort(table["relief_to_thickness_ratio"].unique())
    x_edges = coordinate_edges(ratios, logarithmic=True)
    y_edges = coordinate_edges(reliefs)

    def values_for(column: str) -> np.ndarray:
        values = np.full((len(reliefs), len(ratios)), np.nan)
        for _, row in table.iterrows():
            iy = int(
                np.flatnonzero(
                    np.isclose(reliefs, row["relief_to_thickness_ratio"])
                )[0]
            )
            ix = int(
                np.flatnonzero(
                    np.isclose(ratios, row["right_to_left_length_ratio"])
                )[0]
            )
            if bool(row.get("valid", False)) and np.isfinite(
                row.get(column, np.nan)
            ):
                values[iy, ix] = row[column]
        return values

    minimum_j = values_for("minimum_resolved_J")
    ablation = values_for("peak_required_ablation_m_per_yr")
    counterflow = values_for(
        "eulerian_significant_counterflow_area_fraction"
    )
    finite = minimum_j[np.isfinite(minimum_j)]
    if not len(finite):
        raise RuntimeError("no valid steady phase-map cells")
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("0.75")
    # A fixed display range keeps the transition legible; unsaturated extrema
    # remain in the CSV and colorbar extensions flag values beyond the scale.
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    fig = plt.figure(figsize=(12.4, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(1.0, 1.18),
        height_ratios=(1.0, 1.0, 0.15),
    )
    geometry_axis = fig.add_subplot(grid[0, 0])
    j_axis = fig.add_subplot(grid[0, 1])
    ablation_axis = fig.add_subplot(grid[1, 0])
    counterflow_axis = fig.add_subplot(grid[1, 1])
    caption_axis = fig.add_subplot(grid[2, :])
    caption_axis.axis("off")
    draw_geometry_schematic(geometry_axis)
    image = j_axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(minimum_j),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="0.88",
        linewidth=0.25,
    )
    j_axis.set(
        xlabel=r"flank-length ratio $q=L_{\rm right}/L_{\rm left}$",
        ylabel=r"bed-relief ratio $\Delta b/H$",
        title=r"(b) Minimum steady $J=T_x$ outside outflow buffer",
    )
    colorbar = fig.colorbar(image, ax=j_axis, pad=0.02, extend="both")
    colorbar.set_label(
        r"minimum $J=\partial x/\partial x_0$ (zero is white)"
    )
    j_axis.text(
        0.5,
        -0.19,
        r"Empirical onset tracks $S_{\max}\approx0.9$--$1.0$ "
        r"($S_{\max}=\frac{\pi}{2}\,\Delta b/L_{\rm short}$)",
        transform=j_axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.7,
        color="0.25",
    )

    diagnostic_specs = (
        (
            ablation_axis,
            ablation,
            "magma",
            "(c) Peak ablation required for steady geometry",
            r"$\max(-\dot a,0)$ (m yr$^{-1}$)",
            0.0,
            None,
        ),
        (
            counterflow_axis,
            counterflow,
            "viridis",
            "(d) Significant counter-slope flow area",
            "area fraction with counter-slope flow",
            0.0,
            float(np.nanmax(counterflow)),
        ),
    )
    for axis, values, color_map, title, label, minimum, maximum in diagnostic_specs:
        diagnostic_image = axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(values),
            cmap=color_map,
            vmin=minimum,
            vmax=maximum,
            shading="flat",
            edgecolors="0.88",
            linewidth=0.25,
        )
        axis.set(
            xlabel=r"flank-length ratio $q=L_{\rm right}/L_{\rm left}$",
            ylabel=r"bed-relief ratio $\Delta b/H$",
            title=title,
        )
        diagnostic_colorbar = fig.colorbar(
            diagnostic_image, ax=axis, pad=0.02
        )
        diagnostic_colorbar.set_label(label)

    ticks = np.array((1 / 12, 1 / 4, 1 / 2, 1, 2, 4, 8, 12))
    labels = (r"$1/12$", r"$1/4$", r"$1/2$", r"$1$", r"$2$", r"$4$", r"$8$", r"$12$")
    for axis in (j_axis, ablation_axis, counterflow_axis):
        axis.set_xscale("log")
        axis.set_xticks(ticks, labels)
        axis.set_xlim(x_edges[[0, -1]])
        axis.set_ylim(y_edges[[0, -1]])
        axis.set_yticks(np.arange(0.5, 4.01, 0.5))
        axis.minorticks_off()

    overturned_count = int(np.count_nonzero(finite < -0.05))
    unresolved_count = int(np.count_nonzero(np.abs(finite) <= 0.05))
    caption_axis.text(
        0.5,
        0.5,
        (
            rf"Fixed velocity; source-trimmed steady characteristics run to first exit. "
            rf"{overturned_count}/{len(finite)} cells have "
            rf"$J_{{\min}}<-0.05$ ({unresolved_count} unresolved within $|J_{{\min}}|\leq0.05$).  "
            rf"Sources: central 5--95% of accumulation flux; beneath ablating surface, "
            rf"$d_{{\rm surface}}\geq {2 * THICKNESS / SCREEN_MESH[1]:.1f}$ m."
            "\n"
            r"Transition is trim-sensitive ($q=7$, $\Delta b/H=2.50$: "
            r"$-12.68/-0.150/+0.147$ for 1--99/5--95/10--90%); "
            r"strong $q=1/8$, $\Delta b/H=3.5$ remains negative "
            r"($-44.3/-3.70/-1.30$). Trim dominates onset uncertainty."
        ),
        transform=caption_axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.7,
        color="0.25",
    )
    fig.suptitle("Finite-sliding convergent flow", fontsize=13)
    fig.savefig(FIGURE_PATH, dpi=240, bbox_inches="tight")
    fig.savefig(FIGURE_PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def write_report(
    table: pd.DataFrame,
    validation: pd.DataFrame | None,
    eulerian_audit: pd.DataFrame | None,
) -> None:
    """Write a concise interpretation and convergence/censoring audit."""
    valid = table[table["valid"].astype(bool)].copy()
    negative = valid[valid["minimum_resolved_J"] < -0.05]
    unresolved = valid[np.abs(valid["minimum_resolved_J"]) <= 0.05]
    raw_negative = valid[valid["minimum_raw_J"] < -0.05]
    caps = int(table["safety_cap_trajectory_count"].fillna(0).sum())
    has_counterflow = (
        valid["eulerian_significant_counterflow_area_fraction"] > 0.0
    )
    is_overturned = valid["minimum_resolved_J"] < -0.05
    overturn_without_counterflow = int(
        np.count_nonzero(is_overturned & ~has_counterflow)
    )
    counterflow_without_overturn = int(
        np.count_nonzero(has_counterflow & ~is_overturned)
    )
    ablation_correlation = float(
        np.corrcoef(
            valid["minimum_resolved_J"],
            valid["peak_required_ablation_m_per_yr"],
        )[0, 1]
    )
    counterflow_correlation = float(
        np.corrcoef(
            valid["minimum_resolved_J"],
            valid["eulerian_significant_counterflow_area_fraction"],
        )[0, 1]
    )
    isolated_minimum_count = int(
        np.count_nonzero(
            (valid["minimum_resolved_J"] < -0.05)
            & (
                valid["resolved_per_source_minimum_J_1st_percentile"]
                > 0.05
            )
        )
    )
    slope_prediction = valid["maximum_short_flank_slope"] > 0.917
    slope_mismatch_count = int(
        np.count_nonzero(slope_prediction != is_overturned)
    )
    robust_overturn = (
        valid["resolved_per_source_minimum_J_1st_percentile"] < -0.05
    )
    robust_slope_prediction = valid["maximum_short_flank_slope"] > 1.10
    robust_slope_mismatch_count = int(
        np.count_nonzero(robust_slope_prediction != robust_overturn)
    )
    lines = [
        "# Steady convergent-flow parameter map",
        "",
        "## Diagnostic",
        "",
        "For each fixed geometry, the full-Stokes velocity is solved once and held fixed. The equal-age tangent is then carried along surface-origin characteristics by",
        "",
        r"\[\frac{d\mathbf X}{da}=\mathbf v(\mathbf X),\qquad \frac{d\mathbf T}{da}=(\nabla\mathbf v)\mathbf T,\qquad J=T_x.\]",
        "",
        "Every characteristic is followed to its first physical bed, surface, or divide exit. The 120 kyr ceiling is only a reported safety cap, not an analysis horizon. Particle age parameterizes a characteristic, but neither the glacier nor deposited cohorts are advanced in time. The plotted resolved map asks whether a retained path overturns before entering the stated ablating-surface buffer; the raw to-exit minimum remains in the CSV. No 2 or 8 kyr reporting choice is involved.",
        "",
        "The source interval is fixed a priori to the central 5--95% of positive surface accumulation flux in each basin. The primary minimum also excludes the final two vertical mesh cells (16.7 m) below an ablating surface, where gradient interpolation and a one-path extremum are least reliable. The raw full-residence minimum and its surface clearance remain in the CSV.",
        "",
        "This source trim is a physical definition, not merely a convergence setting. As accumulation approaches zero at a source-basin endpoint, residence age becomes singular or extremely long, so an untrimmed global minimum is not stable. The plotted phase map is explicitly conditional on the central 5--95% accumulation-flux convention; trim sensitivity is reported separately below.",
        "",
        "The effect is first-order near onset. At q=7 and Delta b/H=2.50, the resolved minimum is -12.68, -0.150, and +0.147 under the 1--99%, 5--95%, and 10--90% trims; q=7 and Delta b/H=2.25 also changes sign. By contrast, the strongly overturned q=1/8, Delta b/H=3.5 case remains negative (-44.3, -3.70, -1.30). Source trim therefore dominates transition-boundary uncertainty, while the strong-overturn region is robust.",
        "",
        "## Result",
        "",
        f"The screen contains {len(valid)} valid cells. Using an a-priori numerical classification, J_min<-0.05 is resolved overturning, J_min>0.05 is resolved ordered, and |J_min|<=0.05 is unresolved. The resolved diagnostic overturns in {len(negative)} cells and is unresolved in {len(unresolved)}; the unbuffered diagnostic overturns in {len(raw_negative)}. The most negative resolved value is {valid['minimum_resolved_J'].min():.3f}, while the most negative across-case 1st percentile of per-source minima is {valid['resolved_per_source_minimum_J_1st_percentile'].min():.3f}. In {isolated_minimum_count} mapped cells the single-source minimum crosses -0.05 while the per-source 1st percentile remains above +0.05, flagging onset controlled by a very narrow retained-source interval. Across the grid, {caps} trajectories reached the safety cap.",
        "",
        "## Flow-mechanics comparison",
        "",
        f"Peak required ablation ranges from {valid['peak_required_ablation_m_per_yr'].min():.3f} to {valid['peak_required_ablation_m_per_yr'].max():.3f} m/yr and generally strengthens as relief increases. Its gridwise Pearson correlation with J_min is {ablation_correlation:.3f}. The significant-counterflow panel measures the fraction of sampled area moving opposite the local descent toward the basin center, after requiring speed greater than 1% of the domain maximum; its correlation with J_min is {counterflow_correlation:.3f}.",
        "",
        f"Counter-slope pockets are not themselves the overturn criterion: {counterflow_without_overturn} cells contain significant counterflow but are not resolved as overturned, while {overturn_without_counterflow} overturned cells contain no significant counterflow at this threshold. The panels therefore support cumulative strain along the retained characteristic—not ablation strength or counterflow alone—as the direct mechanism controlling J.",
        "",
        "Across this sampled family, significant counterflow appears on the ordered side of the phase boundary and is present in every overturned cell: it precedes overturning as roughness/asymmetry increases, but is far from sufficient for overturning.",
        "",
        f"A compact geometric summary nevertheless emerges. Let S_max be the maximum absolute bed slope on the shorter flank, S_max=(pi/2) Delta b/L_short. The empirical threshold S_max approximately 0.917 (42.5 degrees) reproduces the isolated-minimum classification in {len(valid) - slope_mismatch_count}/{len(valid)} sampled cells; S_max approximately 1.10 (47.7 degrees) reproduces the broader-source criterion p01(J_min)<-0.05 in {len(valid) - robust_slope_mismatch_count}/{len(valid)} cells. These are phase-map summaries, not universal local overturn criteria: J still accumulates along the full upstream path.",
        "",
    ]
    if len(negative):
        first_by_q = (
            negative.groupby("right_to_left_length_ratio")["relief_to_thickness_ratio"]
            .min()
            .sort_index()
        )
        lines += [
            "The first sampled relief ratio with resolved overturning at each affected q is:",
            "",
            "| q | first sampled Delta b/H with J<-0.05 |",
            "|---:|---:|",
        ]
        lines.extend(f"| {q:g} | {relief:g} |" for q, relief in first_by_q.items())
        lines.append("")
    if validation is not None and len(validation):
        usable = validation[validation["valid"].astype(bool)]
        spread_rows = []
        for check, subset in usable.groupby("check"):
            category = (
                "definition sensitivity"
                if check in ("outflow clearance", "source flux trim")
                else "numerical convergence"
            )
            minimum_spread = (
                subset.groupby("case")["minimum_resolved_J"]
                .agg(lambda values: float(np.ptp(values)))
                .max()
            )
            percentile_spread = (
                subset.groupby("case")[
                    "resolved_per_source_minimum_J_1st_percentile"
                ]
                .agg(lambda values: float(np.ptp(values)))
                .max()
            )
            spread_rows.append(
                (category, check, minimum_spread, percentile_spread)
            )
        lines += [
            "## Numerical checks",
            "",
            "Transition and reciprocal-counterflow cases use 120x16, 180x24, and 240x32 Stokes meshes; the extreme case uses 180x24, 240x32, and 300x40. All mesh comparisons hold physical clearance fixed at 16.7 m. Additional checks span 10--25 m clearances; 41--161 sources per basin; 12.5--50 yr RK4 steps; 1--99%, 5--95%, and 10--90% source-flux trims; and 1201x101 versus 1801x151 velocity samplers in transition and extreme cases.",
            "",
            "Maximum within-case ranges are kept separate for numerical convergence and for deliberate changes to the diagnostic definition:",
            "",
            "| category | check | range of isolated minimum J | range of per-source p01 J |",
            "|---|---|---:|---:|",
        ]
        lines.extend(
            f"| {category} | {check} | {minimum_spread:.3g} | {percentile_spread:.3g} |"
            for category, check, minimum_spread, percentile_spread in spread_rows
        )
        lines += [
            "",
            "The isolated minimum is intentionally retained for the existence question, but the 1st percentile distinguishes a broad overturning source band from a one-source onset. Individual variants and all exit/counterflow diagnostics are in the validation CSV.",
            "",
            "Applying the 16.7 m clearance only beneath ablating surface, rather than globally, changes J_min in 38/336 cells (maximum absolute change 0.0801) but changes no ordered/unresolved/overturned classifications.",
            "",
        ]
        transition = usable[usable["case"] == "transition overturned"]
        extreme = usable[usable["case"] == "extended extreme"]

        def diagnostic_range(frame, check, column):
            values = frame.loc[frame["check"] == check, column]
            return float(np.ptp(values)) if len(values) else np.nan

        lines += [
            f"At the transition, the isolated-minimum ranges are {diagnostic_range(transition, 'mesh at fixed clearance', 'minimum_resolved_J'):.3g} across the fixed-clearance mesh test, {diagnostic_range(transition, 'velocity sampler', 'minimum_resolved_J'):.3g} across velocity samplers, and {diagnostic_range(transition, 'time step', 'minimum_resolved_J'):.3g} across time steps; its sign classification is stable under those numerical changes.",
            "",
            f"The saturated extreme magnitude is not converged as a precise value: q=12, Delta b/H=4 spans {diagnostic_range(extreme, 'time step', 'minimum_resolved_J'):.3g} in J_min across time steps and {diagnostic_range(extreme, 'velocity sampler', 'minimum_resolved_J'):.3g} across velocity samplers (p01 ranges {diagnostic_range(extreme, 'velocity sampler', 'resolved_per_source_minimum_J_1st_percentile'):.3g}). Its negative sign is stable and the figure clips J below -1, so the extreme colors support robust occurrence and order of magnitude, not a converged amplitude.",
            "",
        ]
    if eulerian_audit is not None and len(eulerian_audit):
        ordered = eulerian_audit.iloc[0]
        audit_values = ", ".join(
            f"{row['case']}: min {row['minimum_resolved_J']:.2f}, p01 {row['stationary_resolved_J_1st_percentile']:.2f}"
            for _, row in eulerian_audit.iterrows()
        )
        lines += [
            "## Why the global stationary-DG field is not mapped",
            "",
            "A stationary Eulerian upwind-DG tangent solve was also tested with the correct curved-surface boundary value `T=(1,s_x)`, transported source labels, separate source-basin masks, and the same two-cell outflow buffer. It is reliable in the single-basin elliptical tests, but not here: the two inflow basins meet at a strongly compressed internal separatrix, where numerical branch mixing creates large spurious negative extrema.",
            "",
            f"The audit gives {audit_values}. In particular, the independently ordered q={ordered['right_to_left_length_ratio']:g}, Delta b/H={ordered['relief_to_thickness_ratio']:g} case is falsely classified as strongly overturned. The Eulerian extrema are therefore rejected rather than plotted. Direct steady characteristics preserve basin identity and remain well posed through horizontal counterflow.",
            "",
        ]
    lines += [
        "## Outputs",
        "",
        "- `data/steady_tangent_convergent_parameter_map.csv`",
        "- `data/steady_tangent_convergent_parameter_map_validation.csv`",
        "- `data/steady_tangent_convergent_eulerian_audit.csv`",
        "- `data/steady_tangent_convergent_parameter_map_metadata.json`",
        "",
        "No manuscript source is modified by this analysis.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


def write_metadata(
    table: pd.DataFrame,
    validation: pd.DataFrame | None,
    eulerian_audit: pd.DataFrame | None,
    runtime: float,
) -> None:
    """Save enough provenance to distinguish this map from fixed-age studies."""
    valid = table[table["valid"].astype(bool)]
    isolated_overturn = valid["minimum_resolved_J"] < -0.05
    robust_overturn = (
        valid["resolved_per_source_minimum_J_1st_percentile"] < -0.05
    )
    isolated_slope_prediction = valid["maximum_short_flank_slope"] > 0.917
    robust_slope_prediction = valid["maximum_short_flank_slope"] > 1.10
    metadata = {
        "diagnostic": "steady variational tangent carried to first physical exit",
        "mapped_quantity": "minimum resolved T_x",
        "source_flux_quantiles": list(SOURCE_FLUX_QUANTILES),
        "source_trim_interpretation": (
            "defining convention: near-zero-accumulation source endpoints "
            "have singular or extremely long residence age"
        ),
        "surface_outflow_buffer_m": 2.0 * THICKNESS / SCREEN_MESH[1],
        "outflow_buffer_application": "only beneath ablating surface",
        "global_vs_ablation_only_buffer_changed_cell_count": 38,
        "global_vs_ablation_only_buffer_maximum_absolute_J_change": 0.08006457108106746,
        "global_vs_ablation_only_buffer_classification_change_count": 0,
        "screen_mesh": list(SCREEN_MESH),
        "source_count_per_family": SOURCE_COUNT,
        "time_step_yr": TIME_STEP_YR,
        "maximum_age_safety_cap_yr": MAXIMUM_AGE_YR,
        "valid_cells": int(table["valid"].astype(bool).sum()),
        "resolved_overturning_cells": int(
            np.count_nonzero(
                table.loc[
                    table["valid"].astype(bool), "minimum_resolved_J"
                ]
                < -0.05
            )
        ),
        "unresolved_cells": int(
            np.count_nonzero(
                np.abs(
                    table.loc[
                        table["valid"].astype(bool), "minimum_resolved_J"
                    ]
                )
                <= 0.05
            )
        ),
        "safety_cap_trajectory_count": int(
            table["safety_cap_trajectory_count"].fillna(0).sum()
        ),
        "validation_rows": int(len(validation)) if validation is not None else 0,
        "eulerian_audit_rows": (
            int(len(eulerian_audit)) if eulerian_audit is not None else 0
        ),
        "eulerian_phase_map_rejected": True,
        "eulerian_rejection_reason": (
            "source-basin separatrix compression contaminates stationary "
            "DG tangent extrema"
        ),
        "empirical_short_flank_slope_threshold_isolated_minimum": 0.917,
        "empirical_short_flank_angle_threshold_isolated_minimum_degrees": float(
            np.degrees(np.arctan(0.917))
        ),
        "empirical_short_flank_slope_threshold_robust_p01": 1.10,
        "empirical_short_flank_angle_threshold_robust_p01_degrees": float(
            np.degrees(np.arctan(1.10))
        ),
        "isolated_slope_threshold_mismatch_count": int(
            np.count_nonzero(isolated_slope_prediction != isolated_overturn)
        ),
        "robust_slope_threshold_mismatch_count": int(
            np.count_nonzero(robust_slope_prediction != robust_overturn)
        ),
        "last_render_runtime_s": runtime,
        "summed_direct_screen_case_runtime_s": float(
            table.loc[
                table["solution_provenance"]
                != "exact_horizontal_reflection",
                "runtime_s",
            ].sum()
        ),
        "summed_validation_case_runtime_s": (
            float(validation["runtime_s"].sum())
            if validation is not None
            else 0.0
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if arguments.plot_only:
        table = pd.read_csv(DATA_PATH)
        validation = pd.read_csv(VALIDATION_PATH) if VALIDATION_PATH.exists() else None
        eulerian_audit = (
            pd.read_csv(EULERIAN_AUDIT_PATH)
            if EULERIAN_AUDIT_PATH.exists()
            else None
        )
    else:
        table = run_screen(max(1, arguments.workers))
        validation = None if arguments.skip_validation else run_validation()
        eulerian_audit = run_eulerian_audit()
    table = add_derived_geometry_columns(table)
    table.to_csv(DATA_PATH, index=False)
    make_figure(table)
    write_report(table, validation, eulerian_audit)
    write_metadata(
        table,
        validation,
        eulerian_audit,
        time.perf_counter() - started,
    )
    print(f"Saved {DATA_PATH}")
    print(f"Saved {FIGURE_PATH}")
    print(f"Saved {FIGURE_PDF_PATH}")


if __name__ == "__main__":
    main()
