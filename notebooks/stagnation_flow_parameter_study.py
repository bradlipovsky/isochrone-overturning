"""Geometry phase map for the finite-sliding convergent-flow model.

The two coordinates separate overall roughness from left/right asymmetry:

* ``relief / thickness`` controls the amplitude of the V-shaped bed and
  surface while ice thickness remains fixed; and
* ``right length / left length`` changes the relative flank slopes while the
  total horizontal span remains fixed.

Viscosity, basal drag (and hence mobility), ice thickness, total span,
boundary conditions, mesh size, marker count, and marker time step are held
fixed.  Each cell reports the minimum horizontal material-map derivative
``J = dx / dx0`` over the two surface-source families and their full residence
times.  ``J`` is obtained from the variational equation
``dF/dt = grad(u) F`` along each trajectory, so it does not rely on finite
differences between the last surviving markers at a surface exit.

The Stokes problem and all scalar diagnostics are invariant under horizontal
reflection, which maps ``q = L_right / L_left`` to ``1 / q``.  The screen is
therefore solved for ``q >= 1`` and copied exactly to reciprocal ``q < 1``
cells, with left/right and signed-coordinate fields reflected explicitly.
Selected direct ``q < 1`` solves in the validation table test this symmetry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import stagnation_flow_linear_sliding as convergent


OUTPUT_DIRECTORY = Path("notebooks/stagnation_flow_results")
TABLE_PATH = OUTPUT_DIRECTORY / "convergent_flow_geometry_phase_map.csv"
VALIDATION_PATH = (
    OUTPUT_DIRECTORY / "convergent_flow_geometry_phase_map_validation.csv"
)
FIGURE_STEM = OUTPUT_DIRECTORY / "convergent_flow_geometry_phase_map"

TOTAL_LENGTH = 6_000.0
THICKNESS = 200.0
DIRECT_LENGTH_RATIOS = np.r_[np.arange(1.0, 11.0, 1.0), 12.0]
LENGTH_RATIOS = np.sort(
    np.r_[1.0 / DIRECT_LENGTH_RATIOS[1:], DIRECT_LENGTH_RATIOS]
)
RELIEF_RATIOS = np.arange(0.25, 4.01, 0.25)
SCREEN_MESH = (180, 24)


def set_geometry(length_ratio: float, relief_ratio: float) -> None:
    """Set one geometry while preserving total length and ice thickness."""
    convergent.LEFT_LENGTH = TOTAL_LENGTH / (1.0 + length_ratio)
    convergent.RIGHT_LENGTH = TOTAL_LENGTH - convergent.LEFT_LENGTH
    convergent.RELIEF = relief_ratio * THICKNESS
    convergent.THICKNESS = THICKNESS


def material_map_metrics(family: dict) -> dict:
    """Summarize the variational tangent map for one source family."""
    source = family["source_x"]
    flat = int(np.nanargmin(family["Jx"]))
    time_index, source_index = np.unravel_index(flat, family["Jx"].shape)
    neighbor = family["Jx_neighbor"]
    negative = np.isfinite(family["Jx"]) & (family["Jx"] < 0.0)
    clearance = (1.0 - family["sigma"]) * THICKNESS
    if np.any(negative):
        first_time = int(np.flatnonzero(np.any(negative, axis=1))[0])
        first_sources = np.flatnonzero(negative[first_time])
        first_source = int(
            first_sources[np.argmin(family["Jx"][first_time, first_sources])]
        )
        first_negative_age = float(family["times"][first_time])
        first_negative_clearance = float(
            (1.0 - family["sigma"][first_time, first_source]) * THICKNESS
        )
        first_negative_value = float(family["Jx"][first_time, first_source])
    else:
        first_negative_age = np.nan
        first_negative_clearance = np.nan
        first_negative_value = np.nan
    interior_negative = negative & (clearance >= 10.0)
    if np.any(interior_negative):
        first_interior_time = int(
            np.flatnonzero(np.any(interior_negative, axis=1))[0]
        )
        first_interior_sources = np.flatnonzero(
            interior_negative[first_interior_time]
        )
        first_interior_source = int(
            first_interior_sources[
                np.argmin(
                    family["Jx"][first_interior_time, first_interior_sources]
                )
            ]
        )
        first_interior_age = float(family["times"][first_interior_time])
        first_interior_clearance = float(
            clearance[first_interior_time, first_interior_source]
        )
    else:
        first_interior_age = np.nan
        first_interior_clearance = np.nan
    return {
        "minimum_Jx": float(family["Jx"][time_index, source_index]),
        "age_at_minimum_Jx_yr": float(family["times"][time_index]),
        "source_x_at_minimum_Jx_m": float(source[source_index]),
        "active_markers_at_minimum": int(
            np.count_nonzero(family["active"][time_index])
        ),
        "surface_clearance_at_minimum_m": float(
            (1.0 - family["sigma"][time_index, source_index]) * THICKNESS
        ),
        "first_negative_age_yr": first_negative_age,
        "minimum_Jx_at_first_negative_age": first_negative_value,
        "surface_clearance_at_first_negative_m": first_negative_clearance,
        "first_negative_age_at_least_10m_below_surface_yr": first_interior_age,
        "clearance_at_first_negative_at_least_10m_below_surface_m": (
            first_interior_clearance
        ),
        "neighbor_difference_minimum_Jx": (
            float(np.nanmin(neighbor))
            if np.any(np.isfinite(neighbor))
            else np.nan
        ),
    }


def flow_dynamics_metrics(result: dict) -> dict:
    """Diagnose mass balance, shear, and genuine counter-slope flow."""
    diagnostics = result["diagnostics"]
    x = result["x"]
    X = result["X"]
    Z = result["Z"]
    u = result["values"][..., 0]

    # The horizontal shear term enters the tangent equation directly through
    # dF_x/dt = u_x F_x + u_z F_z.  Sample away from the outermost rows and
    # columns, where one-sided interpolation derivatives are least accurate.
    sample_X = X[2:-2:4, 3:-3:5]
    sample_Z = Z[2:-2:4, 3:-3:5]
    points = np.column_stack((sample_X.ravel(), sample_Z.ravel()))
    gradients = result["geometry"]["velocity_gradient"](points)
    horizontal_shear = np.abs(gradients[:, 0, 1])
    horizontal_compression = np.maximum(-gradients[:, 0, 0], 0.0)

    # "Counterflow" is defined relative to the local bed/surface descent:
    # u < 0 on the left flank or u > 0 on the right flank.  Exclude only a
    # narrow 30 m strip at the non-differentiable join and require at least
    # one percent of the domain-wide maximum |u| for the area fraction.
    away_from_join = np.abs(X) >= 0.005 * TOTAL_LENGTH
    counter = away_from_join & (
        ((X < 0.0) & (u < 0.0)) | ((X > 0.0) & (u > 0.0))
    )
    maximum_abs_u = float(np.nanmax(np.abs(u)))
    significant_counter = counter & (
        np.abs(u) >= 0.01 * max(maximum_abs_u, 1.0e-12)
    )
    maximum_counter_speed = (
        float(np.nanmax(np.abs(u[counter]))) if np.any(counter) else 0.0
    )

    # Locate the internal surface-u zero.  Its displacement from x=0 is the
    # width of the surface counter-slope reach on the long flank.
    surface_u = result["geometry"]["surface_velocity"](x)[:, 0]
    crossings = np.flatnonzero(
        (surface_u[:-1] >= 0.0) & (surface_u[1:] <= 0.0)
    )
    crossings = crossings[(crossings > 2) & (crossings < len(x) - 4)]
    roots = []
    for index in crossings:
        denominator = surface_u[index + 1] - surface_u[index]
        if abs(denominator) > 1.0e-14:
            roots.append(
                x[index]
                - surface_u[index]
                * (x[index + 1] - x[index])
                / denominator
            )
    confluence_x = min(roots, key=abs) if roots else np.nan

    return {
        "peak_required_ablation_m_per_yr": float(
            -diagnostics["minimum_surface_mass_balance"]
        ),
        "peak_accumulation_m_per_yr": float(
            diagnostics["maximum_surface_mass_balance"]
        ),
        "maximum_upwelling_m_per_yr": float(
            diagnostics["maximum_nodal_w"]
        ),
        "maximum_downwelling_m_per_yr": float(
            -diagnostics["minimum_nodal_w"]
        ),
        "maximum_horizontal_shear_rate_per_yr": float(
            np.nanmax(horizontal_shear)
        ),
        "maximum_horizontal_compression_rate_per_yr": float(
            np.nanmax(horizontal_compression)
        ),
        "surface_confluence_x_m": float(confluence_x),
        "surface_counter_slope_reach_m": float(abs(confluence_x)),
        "maximum_counterflow_speed_m_per_yr": maximum_counter_speed,
        "significant_counterflow_area_fraction": float(
            np.count_nonzero(significant_counter)
            / np.count_nonzero(away_from_join)
        ),
    }


def trajectory_path_metrics(result: dict) -> dict:
    """Count basin crossings, horizontal turns, and sampled recirculation."""
    trajectory_count = 0
    crossing_count = 0
    reversal_count = 0
    return_after_crossing_count = 0
    closed_count = 0
    maximum_overshoot = 0.0
    for family in result["families"]:
        for source_index, source_x in enumerate(family["source_x"]):
            positions = family["positions"][:, source_index]
            positions = positions[np.all(np.isfinite(positions), axis=1)]
            if len(positions) < 3:
                continue
            dx = np.diff(positions[:, 0])
            resolved_dx = dx[np.abs(dx) > 0.01]
            if not len(resolved_dx):
                continue
            trajectory_count += 1
            directions = np.sign(resolved_dx)
            if np.any(directions[1:] != directions[:-1]):
                reversal_count += 1
            crossed = positions[:, 0] * source_x < 0.0
            if np.any(crossed):
                crossing_count += 1
                first_crossing = int(np.flatnonzero(crossed)[0])
                overshoot = np.maximum(
                    -np.sign(source_x) * positions[:, 0], 0.0
                )
                maximum_overshoot = max(
                    maximum_overshoot, float(np.max(overshoot))
                )
                after_crossing_dx = np.diff(positions[first_crossing:, 0])
                if np.any(after_crossing_dx * np.sign(source_x) > 0.01):
                    return_after_crossing_count += 1
            if np.linalg.norm(positions[-1] - positions[0]) < 50.0:
                closed_count += 1
    denominator = max(trajectory_count, 1)
    return {
        "sampled_trajectory_count": trajectory_count,
        "basin_crossing_trajectory_fraction": crossing_count / denominator,
        "horizontal_reversal_trajectory_fraction": (
            reversal_count / denominator
        ),
        "return_after_crossing_trajectory_fraction": (
            return_after_crossing_count / denominator
        ),
        "closed_or_recirculating_trajectory_count": closed_count,
        "maximum_basin_overshoot_m": maximum_overshoot,
    }


def summarize_result(result: dict) -> dict:
    """Return scalar phase-map diagnostics and conservative validity tests."""
    diagnostics = result["diagnostics"]
    family_metrics = {
        family["name"]: material_map_metrics(family)
        for family in result["families"]
    }
    dynamics = flow_dynamics_metrics(result)
    short_length = min(convergent.LEFT_LENGTH, convergent.RIGHT_LENGTH)
    left_peak_slope = (
        0.5 * np.pi * convergent.RELIEF / convergent.LEFT_LENGTH
    )
    right_peak_slope = (
        0.5 * np.pi * convergent.RELIEF / convergent.RIGHT_LENGTH
    )
    minimum = float(diagnostics.get("minimum_Jx", np.nan))
    minimum_family = min(
        family_metrics,
        key=lambda name: family_metrics[name]["minimum_Jx"],
        default=None,
    )
    negative_families = [
        (name, metrics)
        for name, metrics in family_metrics.items()
        if np.isfinite(metrics["first_negative_age_yr"])
    ]
    first_negative_family, first_negative_metrics = (
        min(
            negative_families,
            key=lambda item: item[1]["first_negative_age_yr"],
        )
        if negative_families
        else (None, {})
    )
    interior_negative_metrics = [
        metrics
        for metrics in family_metrics.values()
        if np.isfinite(
            metrics["first_negative_age_at_least_10m_below_surface_yr"]
        )
    ]
    first_interior_negative = (
        min(
            interior_negative_metrics,
            key=lambda metrics: metrics[
                "first_negative_age_at_least_10m_below_surface_yr"
            ],
        )
        if interior_negative_metrics
        else {}
    )
    fixed_age_minima = {}
    for age in (3000.0, 4000.0):
        candidates = []
        negative_clearances = []
        for family in result["families"]:
            time_index = int(np.argmin(np.abs(family["times"] - age)))
            values = family["Jx"][time_index]
            if np.any(np.isfinite(values)):
                source_index = int(np.nanargmin(values))
                clearance = float(
                    (1.0 - family["sigma"][time_index, source_index])
                    * THICKNESS
                )
                candidates.append((float(values[source_index]), clearance))
                negative = np.isfinite(values) & (values < 0.0)
                if np.any(negative):
                    negative_clearances.extend(
                        (
                            (1.0 - family["sigma"][time_index, negative])
                            * THICKNESS
                        ).tolist()
                    )
        fixed_age_minima[age] = (
            min(candidates, key=lambda item: item[0])
            if candidates
            else (np.nan, np.nan)
        )
        fixed_age_minima[(age, "deepest_negative")] = (
            max(negative_clearances) if negative_clearances else np.nan
        )
    two_sources = len(result["families"]) == 2
    energy_ok = abs(diagnostics["relative_energy_residual"]) < 1.0e-3
    basal_ok = diagnostics["rms_basal_normal_speed"] < 1.0e-3
    valid = (
        np.isfinite(minimum)
        and two_sources
        and energy_ok
        and basal_ok
    )
    summary = {
        "valid": bool(valid),
        "status": "valid" if valid else "failed_diagnostic",
        "minimum_Jx": minimum,
        "left_source_minimum_Jx": family_metrics.get("left_flank", {}).get(
            "minimum_Jx", np.nan
        ),
        "right_source_minimum_Jx": family_metrics.get("right_flank", {}).get(
            "minimum_Jx", np.nan
        ),
        "minimum_Jx_neighbor_difference": min(
            (
                metrics["neighbor_difference_minimum_Jx"]
                for metrics in family_metrics.values()
            ),
            default=np.nan,
        ),
        "minimum_Jx_family": minimum_family,
        "age_at_minimum_Jx_yr": family_metrics.get(minimum_family, {}).get(
            "age_at_minimum_Jx_yr", np.nan
        ),
        "source_x_at_minimum_Jx_m": family_metrics.get(
            minimum_family, {}
        ).get("source_x_at_minimum_Jx_m", np.nan),
        "active_markers_at_minimum_Jx": family_metrics.get(
            minimum_family, {}
        ).get("active_markers_at_minimum", np.nan),
        "surface_clearance_at_minimum_Jx_m": family_metrics.get(
            minimum_family, {}
        ).get("surface_clearance_at_minimum_m", np.nan),
        "first_negative_family": first_negative_family,
        "first_negative_age_yr": first_negative_metrics.get(
            "first_negative_age_yr", np.nan
        ),
        "minimum_Jx_at_first_negative_age": first_negative_metrics.get(
            "minimum_Jx_at_first_negative_age", np.nan
        ),
        "surface_clearance_at_first_negative_m": first_negative_metrics.get(
            "surface_clearance_at_first_negative_m", np.nan
        ),
        "first_negative_age_at_least_10m_below_surface_yr": (
            first_interior_negative.get(
                "first_negative_age_at_least_10m_below_surface_yr", np.nan
            )
        ),
        "clearance_at_first_negative_at_least_10m_below_surface_m": (
            first_interior_negative.get(
                "clearance_at_first_negative_at_least_10m_below_surface_m",
                np.nan,
            )
        ),
        "minimum_Jx_at_3000yr": fixed_age_minima[3000.0][0],
        "clearance_at_minimum_Jx_at_3000yr_m": fixed_age_minima[3000.0][1],
        "deepest_negative_clearance_at_3000yr_m": fixed_age_minima[
            (3000.0, "deepest_negative")
        ],
        "minimum_Jx_at_4000yr": fixed_age_minima[4000.0][0],
        "clearance_at_minimum_Jx_at_4000yr_m": fixed_age_minima[4000.0][1],
        "deepest_negative_clearance_at_4000yr_m": fixed_age_minima[
            (4000.0, "deepest_negative")
        ],
        "overturned_samples": diagnostics.get("overturned_samples", np.nan),
        "maximum_age_before_exit_yr": diagnostics.get(
            "maximum_age_before_exit", np.nan
        ),
        "minimum_path_speed_m_per_yr": diagnostics.get(
            "minimum_speed", np.nan
        ),
        "relative_energy_residual": diagnostics[
            "relative_energy_residual"
        ],
        "rms_basal_normal_speed_m_per_yr": diagnostics[
            "rms_basal_normal_speed"
        ],
        "accumulation_intervals": diagnostics["accumulation_intervals"],
        "left_flank_peak_slope": left_peak_slope,
        "right_flank_peak_slope": right_peak_slope,
        "short_flank_peak_slope": max(left_peak_slope, right_peak_slope),
        "short_flank_dimensionless_center_curvature": (
            THICKNESS
            * 0.5
            * np.pi**2
            * convergent.RELIEF
            / short_length**2
        ),
    }
    summary.update(dynamics)
    return summary


def run_case(
    length_ratio: float,
    relief_ratio: float,
    nx: int,
    nz: int,
) -> tuple[dict, dict | None]:
    """Solve and trace one geometry, preserving failures as table rows."""
    set_geometry(length_ratio, relief_ratio)
    row = {
        "right_to_left_length_ratio": length_ratio,
        "relief_to_thickness_ratio": relief_ratio,
        "left_length_m": convergent.LEFT_LENGTH,
        "right_length_m": convergent.RIGHT_LENGTH,
        "relief_m": convergent.RELIEF,
        "thickness_m": THICKNESS,
        "total_length_m": TOTAL_LENGTH,
        "nx": nx,
        "nz": nz,
        "viscosity_Pa_yr": convergent.VISCOSITY,
        "basal_drag_Pa_yr_per_m": convergent.BASAL_DRAG,
        "sliding_mobility_m_per_Pa_yr": 1.0 / convergent.BASAL_DRAG,
        "source_count_per_family": 81,
        "time_step_yr": 50.0,
        "sampler_nx": 1201,
        "sampler_nsigma": 101,
        "solution_provenance": "direct_solve",
        "error": "",
    }
    try:
        result = convergent.diagnose_resolution(nx, nz, trace=True)
        # The base solver's historical names assumed q > 1.  Location-based
        # names remain correct for both direct reciprocal validation cases.
        for family in result["families"]:
            family["name"] = (
                "left_flank"
                if np.nanmean(family["source_x"]) < 0.0
                else "right_flank"
            )
        row.update(summarize_result(result))
        return row, result
    except Exception as error:  # retain invalid cells rather than hiding them
        row.update(
            {
                "valid": False,
                "status": f"{type(error).__name__}",
                "minimum_Jx": np.nan,
                "error": str(error),
            }
        )
        traceback.print_exc()
        return row, None


def screen_worker(arguments: tuple[float, float, int, int]) -> dict:
    """Process-safe wrapper that returns only serializable diagnostics."""
    length_ratio, relief_ratio, nx, nz = arguments
    row, _ = run_case(length_ratio, relief_ratio, nx, nz)
    # A regular sampling point can land exactly on an element-search edge in
    # an otherwise valid curved mesh.  Retry only that point-location failure
    # with one additional horizontal column; neighboring 179/181-column tests
    # agree to 5e-5 in J for the sole screen cell that triggers this path.
    if (
        not bool(row.get("valid", False))
        and row.get("status") == "PointNotInDomainError"
    ):
        row, _ = run_case(length_ratio, relief_ratio, nx + 1, nz)
        if bool(row.get("valid", False)):
            row["solution_provenance"] = "direct_solve_mesh_edge_retry"
    return row


def reflected_row(row: dict) -> dict:
    """Reflect a q>1 scalar result to its exact reciprocal geometry."""
    reflected = dict(row)
    ratio = float(row["right_to_left_length_ratio"])
    if ratio <= 1.0:
        raise ValueError("Only q > 1 rows have a distinct reciprocal")
    reflected["right_to_left_length_ratio"] = 1.0 / ratio
    reflected["left_length_m"] = row["right_length_m"]
    reflected["right_length_m"] = row["left_length_m"]
    reflected["solution_provenance"] = "exact_horizontal_reflection"
    reflected["source_x_at_minimum_Jx_m"] = -row.get(
        "source_x_at_minimum_Jx_m", np.nan
    )
    reflected["surface_confluence_x_m"] = -row.get(
        "surface_confluence_x_m", np.nan
    )
    reflected["left_source_minimum_Jx"] = row.get(
        "right_source_minimum_Jx", np.nan
    )
    reflected["right_source_minimum_Jx"] = row.get(
        "left_source_minimum_Jx", np.nan
    )
    reflected["left_flank_peak_slope"] = row.get(
        "right_flank_peak_slope", np.nan
    )
    reflected["right_flank_peak_slope"] = row.get(
        "left_flank_peak_slope", np.nan
    )
    family_swap = {
        "left_flank": "right_flank",
        "right_flank": "left_flank",
    }
    for column in ("minimum_Jx_family", "first_negative_family"):
        reflected[column] = family_swap.get(row.get(column), row.get(column))
    return reflected


def run_screen(workers: int = 1) -> pd.DataFrame:
    """Solve q >= 1 and fill q < 1 by exact horizontal reflection."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = [
        (length_ratio, relief_ratio, SCREEN_MESH[0], SCREEN_MESH[1])
        for relief_ratio in RELIEF_RATIOS
        for length_ratio in DIRECT_LENGTH_RATIOS
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_cases = {
            executor.submit(screen_worker, case): case for case in cases
        }
        for future in as_completed(future_cases):
            case = future_cases[future]
            row = future.result()
            rows.append(row)
            if case[0] > 1.0:
                rows.append(reflected_row(row))
            print(
                "PHASE MAP COMPLETE "
                f"Lr/Ll={case[0]:g}, relief/H={case[1]:g}: "
                f"{row['status']}, J={row.get('minimum_Jx', np.nan):.6g}",
                flush=True,
            )
            checkpoint = pd.DataFrame(rows).sort_values(
                ["relief_to_thickness_ratio", "right_to_left_length_ratio"]
            )
            checkpoint.to_csv(TABLE_PATH, index=False)
    return pd.DataFrame(rows).sort_values(
        ["relief_to_thickness_ratio", "right_to_left_length_ratio"]
    )


def rerun_material_map(result: dict, source_count: int, time_step: float) -> dict:
    """Repeat the marker calculation on a fixed velocity solution."""
    metrics = []
    fixed_age_minima = {3000.0: [], 4000.0: []}
    fixed_age_negative_clearances = {3000.0: [], 4000.0: []}
    for original in result["families"]:
        sources = np.linspace(
            original["source_x"][0], original["source_x"][-1], source_count
        )
        repeated = convergent.trace_source_family(
            result["sample"],
            result["geometry"],
            sources,
            time_step=time_step,
        )
        metrics.append(material_map_metrics(repeated))
        for age in fixed_age_minima:
            time_index = int(np.argmin(np.abs(repeated["times"] - age)))
            values = repeated["Jx"][time_index]
            if np.any(np.isfinite(values)):
                source_index = int(np.nanargmin(values))
                clearance = float(
                    (1.0 - repeated["sigma"][time_index, source_index])
                    * THICKNESS
                )
                fixed_age_minima[age].append(
                    (float(values[source_index]), clearance)
                )
                negative = np.isfinite(values) & (values < 0.0)
                if np.any(negative):
                    fixed_age_negative_clearances[age].extend(
                        (
                            (1.0 - repeated["sigma"][time_index, negative])
                            * THICKNESS
                        ).tolist()
                    )
    minimum = min(metrics, key=lambda item: item["minimum_Jx"])
    negative_metrics = [
        item for item in metrics if np.isfinite(item["first_negative_age_yr"])
    ]
    first_negative = (
        min(negative_metrics, key=lambda item: item["first_negative_age_yr"])
        if negative_metrics
        else {}
    )
    interior_negative_metrics = [
        item
        for item in metrics
        if np.isfinite(
            item["first_negative_age_at_least_10m_below_surface_yr"]
        )
    ]
    first_interior_negative = (
        min(
            interior_negative_metrics,
            key=lambda item: item[
                "first_negative_age_at_least_10m_below_surface_yr"
            ],
        )
        if interior_negative_metrics
        else {}
    )
    fixed_3000 = (
        min(fixed_age_minima[3000.0], key=lambda item: item[0])
        if fixed_age_minima[3000.0]
        else (np.nan, np.nan)
    )
    fixed_4000 = (
        min(fixed_age_minima[4000.0], key=lambda item: item[0])
        if fixed_age_minima[4000.0]
        else (np.nan, np.nan)
    )
    return {
        "minimum_Jx": minimum["minimum_Jx"],
        "minimum_Jx_neighbor_difference": min(
            item["neighbor_difference_minimum_Jx"] for item in metrics
        ),
        "age_at_minimum_Jx_yr": minimum["age_at_minimum_Jx_yr"],
        "source_x_at_minimum_Jx_m": minimum["source_x_at_minimum_Jx_m"],
        "active_markers_at_minimum_Jx": minimum[
            "active_markers_at_minimum"
        ],
        "surface_clearance_at_minimum_Jx_m": minimum[
            "surface_clearance_at_minimum_m"
        ],
        "first_negative_age_yr": first_negative.get(
            "first_negative_age_yr", np.nan
        ),
        "minimum_Jx_at_first_negative_age": first_negative.get(
            "minimum_Jx_at_first_negative_age", np.nan
        ),
        "surface_clearance_at_first_negative_m": first_negative.get(
            "surface_clearance_at_first_negative_m", np.nan
        ),
        "first_negative_age_at_least_10m_below_surface_yr": (
            first_interior_negative.get(
                "first_negative_age_at_least_10m_below_surface_yr", np.nan
            )
        ),
        "clearance_at_first_negative_at_least_10m_below_surface_m": (
            first_interior_negative.get(
                "clearance_at_first_negative_at_least_10m_below_surface_m",
                np.nan,
            )
        ),
        "minimum_Jx_at_3000yr": fixed_3000[0],
        "clearance_at_minimum_Jx_at_3000yr_m": fixed_3000[1],
        "deepest_negative_clearance_at_3000yr_m": max(
            fixed_age_negative_clearances[3000.0], default=np.nan
        ),
        "minimum_Jx_at_4000yr": fixed_4000[0],
        "clearance_at_minimum_Jx_at_4000yr_m": fixed_4000[1],
        "deepest_negative_clearance_at_4000yr_m": max(
            fixed_age_negative_clearances[4000.0], default=np.nan
        ),
    }


def run_validation() -> pd.DataFrame:
    """Refine transition/extreme cases and test reciprocal symmetry."""
    rows = []
    cases = (
        ("symmetric_high_relief", 1.0, 4.00),
        ("transition_positive", 7.0, 2.25),
        ("transition_negative", 7.0, 2.50),
        ("reciprocal_left_direct", 0.125, 3.50),
        ("reciprocal_right_direct", 8.0, 3.50),
        ("extended_extreme", 12.0, 4.00),
    )
    for name, length_ratio, relief_ratio in cases:
        meshes = (
            ((180, 24), (240, 32), (300, 40))
            if name == "extended_extreme"
            else ((120, 16), (180, 24), (240, 32))
        )
        for nx, nz in meshes:
            print(
                f"\nVALIDATION {name}: Lr/Ll={length_ratio:g}, "
                f"relief/H={relief_ratio:g}, mesh={nx}x{nz}",
                flush=True,
            )
            row, result = run_case(length_ratio, relief_ratio, nx, nz)
            row["case"] = name
            row["marker_variant"] = "81 sources, 50 yr"
            if result is not None:
                row.update(trajectory_path_metrics(result))
            rows.append(row)
            if result is not None and (nx, nz) == meshes[-1]:
                variants = ((81, 25.0), (161, 25.0), (321, 25.0), (321, 12.5))
                for source_count, time_step in variants:
                    refined = row.copy()
                    refined["source_count_per_family"] = source_count
                    refined["time_step_yr"] = time_step
                    refined["marker_variant"] = (
                        f"{source_count} sources, {time_step:g} yr"
                    )
                    try:
                        refined.update(
                            rerun_material_map(result, source_count, time_step)
                        )
                        refined["valid"] = np.isfinite(refined["minimum_Jx"])
                        refined["status"] = (
                            "valid"
                            if refined["valid"]
                            else "failed_diagnostic"
                        )
                    except Exception as error:
                        refined["valid"] = False
                        refined["status"] = type(error).__name__
                        refined["minimum_Jx"] = np.nan
                        refined["error"] = str(error)
                    rows.append(refined)
                if name == "extended_extreme":
                    for sampler_nx, sampler_nsigma in ((601, 51), (1601, 151)):
                        sampled = row.copy()
                        sampled["source_count_per_family"] = 161
                        sampled["time_step_yr"] = 25.0
                        sampled["sampler_nx"] = sampler_nx
                        sampled["sampler_nsigma"] = sampler_nsigma
                        sampled["marker_variant"] = (
                            f"gradient grid {sampler_nx}x{sampler_nsigma}"
                        )
                        try:
                            new_sample, new_geometry, *_ = convergent.make_sampler(
                                result["velocity"],
                                nx_sample=sampler_nx,
                                nz_sample=sampler_nsigma,
                            )
                            resampled_result = dict(result)
                            resampled_result["sample"] = new_sample
                            resampled_result["geometry"] = new_geometry
                            sampled.update(
                                rerun_material_map(
                                    resampled_result,
                                    source_count=161,
                                    time_step=25.0,
                                )
                            )
                            sampled["valid"] = np.isfinite(sampled["minimum_Jx"])
                            sampled["status"] = (
                                "valid"
                                if sampled["valid"]
                                else "failed_diagnostic"
                            )
                        except Exception as error:
                            sampled["valid"] = False
                            sampled["status"] = type(error).__name__
                            sampled["minimum_Jx"] = np.nan
                            sampled["error"] = str(error)
                        rows.append(sampled)
            pd.DataFrame(rows).to_csv(VALIDATION_PATH, index=False)
    return pd.DataFrame(rows)


def coordinate_edges(
    values: np.ndarray, *, logarithmic: bool = False
) -> np.ndarray:
    """Cell edges for linearly or logarithmically displayed centers."""
    if logarithmic:
        midpoints = np.sqrt(values[:-1] * values[1:])
        return np.r_[
            values[0] ** 2 / midpoints[0],
            midpoints,
            values[-1] ** 2 / midpoints[-1],
        ]
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.r_[
        values[0] - (midpoints[0] - values[0]),
        midpoints,
        values[-1] + (values[-1] - midpoints[-1]),
    ]


def draw_geometry_schematic(axis: plt.Axes) -> None:
    """Draw the exact model shape and define its four geometric parameters."""
    # Dimensionless values are chosen only to keep the schematic legible.  The
    # curves use the same piecewise sin^2 geometry as the finite-element mesh.
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
    axis.plot(x, bed, color="0.12", linewidth=1.5, zorder=3)
    axis.plot(x, surface, color="#2C6E9B", linewidth=1.5, zorder=3)
    axis.text(
        2.45,
        1.41,
        "surface",
        color="#2C6E9B",
        fontsize=8.5,
        ha="right",
        va="bottom",
    )
    axis.text(
        1.70,
        np.interp(1.70, x, bed) - 0.10,
        r"bed $b(x)$",
        color="0.15",
        fontsize=8.5,
        ha="center",
    )

    dimension_style = {
        "arrowstyle": "<->",
        "color": "0.12",
        "linewidth": 1.1,
        "shrinkA": 0.0,
        "shrinkB": 0.0,
    }
    guide_style = {"color": "0.45", "linewidth": 0.8, "linestyle": ":"}

    # Horizontal flank lengths are measured from the confluence at x=0.
    length_y = -0.34
    for location, upper_y in (
        (-left_length, relief),
        (0.0, 0.0),
        (right_length, relief),
    ):
        axis.plot(
            (location, location),
            (length_y - 0.03, upper_y),
            **guide_style,
            zorder=0,
        )
    axis.annotate(
        "",
        xy=(0.0, length_y),
        xytext=(-left_length, length_y),
        arrowprops=dimension_style,
        annotation_clip=False,
    )
    axis.annotate(
        "",
        xy=(right_length, length_y),
        xytext=(0.0, length_y),
        arrowprops=dimension_style,
        annotation_clip=False,
    )
    axis.text(
        -0.5 * left_length,
        length_y - 0.07,
        r"$L_{\mathrm{left}}$",
        ha="center",
        va="top",
        fontsize=10,
    )
    axis.text(
        0.5 * right_length,
        length_y - 0.07,
        r"$L_{\mathrm{right}}$",
        ha="center",
        va="top",
        fontsize=10,
    )

    # Relief is the vertical difference between the central minimum and the
    # equal-height outer endpoints.
    relief_x = -left_length - 0.20
    axis.plot((relief_x, -left_length), (0.0, 0.0), **guide_style)
    axis.plot((relief_x, -left_length), (relief, relief), **guide_style)
    axis.annotate(
        "",
        xy=(relief_x, relief),
        xytext=(relief_x, 0.0),
        arrowprops=dimension_style,
        annotation_clip=False,
    )
    axis.text(
        relief_x - 0.06,
        0.5 * relief,
        r"$\Delta b$",
        ha="right",
        va="center",
        fontsize=10,
    )

    # Ice thickness is the vertical bed-to-surface distance everywhere.
    thickness_x = 0.28
    local_bed = float(np.interp(thickness_x, x, bed))
    local_surface = local_bed + thickness
    axis.plot(
        (thickness_x - 0.08, thickness_x + 0.08),
        (local_bed, local_bed),
        **guide_style,
    )
    axis.plot(
        (thickness_x - 0.08, thickness_x + 0.08),
        (local_surface, local_surface),
        **guide_style,
    )
    axis.annotate(
        "",
        xy=(thickness_x, local_surface),
        xytext=(thickness_x, local_bed),
        arrowprops=dimension_style,
    )
    axis.text(
        thickness_x + 0.09,
        0.5 * (local_bed + local_surface),
        r"$H$",
        ha="left",
        va="center",
        fontsize=10,
    )

    # Put the defining equation and scale warning below the drawing, following
    # the convention used in the elliptical-trough parameter figure.
    axis.text(
        0.5,
        -0.60,
        (
            r"$b(x)=\Delta b\,\sin^2[\pi |x|/(2L_{\rm side})]$,  "
            r"$s(x)=b(x)+H$"
        ),
        ha="center",
        va="top",
        fontsize=8.7,
        color="0.2",
    )
    axis.text(
        0.5,
        -0.74,
        (
            r"$L_{\rm side}=L_{\rm left}$ for $x<0$ and "
            r"$L_{\rm right}$ for $x>0$; vertical scale exaggerated"
        ),
        ha="center",
        va="top",
        fontsize=7.6,
        color="0.35",
        style="italic",
    )
    axis.set(
        xlim=(-1.43, 2.68),
        ylim=(-0.88, 1.55),
        title="(a) Geometry and parameter definitions",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def make_figure(table: pd.DataFrame) -> None:
    """Plot geometry, overturning, ablation, and tangent-driving shear."""
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
            if bool(row.get("valid", False)) and np.isfinite(row.get(column)):
                values[iy, ix] = row[column]
        return values

    minimum_J = values_for("minimum_Jx")
    ablation = values_for("peak_required_ablation_m_per_yr")
    shear = 1.0e3 * values_for("maximum_horizontal_shear_rate_per_yr")
    finite_J = minimum_J[np.isfinite(minimum_J)]
    if not finite_J.size:
        raise RuntimeError("No valid phase-map cells to plot")

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("0.75")
    norm = TwoSlopeNorm(
        vmin=float(np.nanmin(finite_J)),
        vcenter=0.0,
        vmax=float(np.nanmax(finite_J)),
    )

    fig = plt.figure(figsize=(13.2, 9.6), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(1.0, 1.17),
        height_ratios=(1.0, 1.0, 0.08),
    )
    schematic_axis = fig.add_subplot(grid[0, 0])
    j_axis = fig.add_subplot(grid[0, 1])
    ablation_axis = fig.add_subplot(grid[1, 0])
    shear_axis = fig.add_subplot(grid[1, 1])
    caption_axis = fig.add_subplot(grid[2, :])
    caption_axis.axis("off")
    draw_geometry_schematic(schematic_axis)

    j_image = j_axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(minimum_J),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="0.88",
        linewidth=0.25,
    )
    j_axis.set(
        ylabel=r"bed-relief ratio $\Delta b/H$",
        title=r"(b) Minimum tangent-map $J=\partial x/\partial x_0$",
    )
    j_colorbar = fig.colorbar(j_image, ax=j_axis, pad=0.02)
    j_colorbar.set_label(r"minimum variational $J$ over full residence")

    diagnostic_specs = (
        (
            ablation_axis,
            ablation,
            "magma",
            "(c) Peak required surface ablation",
            r"$\max(-\dot a,0)$ (m yr$^{-1}$)",
        ),
        (
            shear_axis,
            shear,
            "viridis",
            r"(d) Peak horizontal shear $|\partial u/\partial z|$",
            r"maximum shear ($10^{-3}$ yr$^{-1}$)",
        ),
    )
    for axis, values, color_map, title, label in diagnostic_specs:
        image = axis.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(values),
            cmap=color_map,
            shading="flat",
            edgecolors="0.88",
            linewidth=0.25,
        )
        axis.set(
            xlabel=(
                r"flank-length ratio "
                r"$q=L_{\mathrm{right}}/L_{\mathrm{left}}$"
            ),
            ylabel=r"bed-relief ratio $\Delta b/H$",
            title=title,
        )
        colorbar = fig.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label(label)

    tick_values = np.array((1 / 12, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 12))
    tick_labels = (
        r"$1/12$",
        r"$1/8$",
        r"$1/4$",
        r"$1/2$",
        r"$1$",
        r"$2$",
        r"$4$",
        r"$8$",
        r"$12$",
    )
    for axis in (j_axis, ablation_axis, shear_axis):
        axis.set_xscale("log")
        axis.set_xlim(x_edges[0], x_edges[-1])
        axis.set_ylim(y_edges[0], y_edges[-1])
        axis.set_xticks(tick_values, tick_labels)
        axis.set_yticks(np.arange(0.5, 4.01, 0.5))
        axis.minorticks_off()
    caption_axis.text(
        0.5,
        0.5,
        (
            r"Fixed: $L_{\rm left}+L_{\rm right}=6$ km, "
            r"$H=200$ m, "
            r"$C=10^{-5}$ m Pa$^{-1}$ yr$^{-1}$"
            r", $\eta=2.5\times10^8$ Pa yr, boundary treatment, and resolution.  "
            r"Cells at $q<1$ are exact horizontal reflections of reciprocal "
            r"direct solves at $q>1$."
        ),
        transform=caption_axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color="0.2",
    )
    fig.suptitle("Finite-sliding convergent flow", fontsize=14)
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=240)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only", action="store_true", help="redraw from the existing CSV"
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="omit selected refinements"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent worker processes for the phase-map grid",
    )
    arguments = parser.parse_args()
    if arguments.plot_only:
        table = pd.read_csv(TABLE_PATH)
    else:
        table = run_screen(max(1, arguments.workers))
        if not arguments.skip_validation:
            run_validation()
    make_figure(table)
    print(f"\nSaved {TABLE_PATH} and {FIGURE_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
