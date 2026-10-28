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
LENGTH_RATIOS = np.arange(1.0, 10.0, 1.0)
RELIEF_RATIOS = np.arange(0.25, 2.51, 0.25)
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


def summarize_result(result: dict) -> dict:
    """Return scalar phase-map diagnostics and conservative validity tests."""
    diagnostics = result["diagnostics"]
    family_metrics = {
        family["name"]: material_map_metrics(family)
        for family in result["families"]
    }
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
    return {
        "valid": bool(valid),
        "status": "valid" if valid else "failed_diagnostic",
        "minimum_Jx": minimum,
        "left_source_minimum_Jx": family_metrics.get("left_steep", {}).get(
            "minimum_Jx", np.nan
        ),
        "right_source_minimum_Jx": family_metrics.get("right_gentle", {}).get(
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
    }


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
        "error": "",
    }
    try:
        result = convergent.diagnose_resolution(nx, nz, trace=True)
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
    return row


def run_screen(workers: int = 1) -> pd.DataFrame:
    """Run the full geometry grid and checkpoint after every solve."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = [
        (length_ratio, relief_ratio, SCREEN_MESH[0], SCREEN_MESH[1])
        for relief_ratio in RELIEF_RATIOS
        for length_ratio in LENGTH_RATIOS
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_cases = {
            executor.submit(screen_worker, case): case for case in cases
        }
        for future in as_completed(future_cases):
            case = future_cases[future]
            row = future.result()
            rows.append(row)
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
    fixed_3000 = min(fixed_age_minima[3000.0], key=lambda item: item[0])
    fixed_4000 = min(fixed_age_minima[4000.0], key=lambda item: item[0])
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
    """Refine a baseline, a contour-bracketing pair, and an extreme case."""
    rows = []
    cases = (
        ("baseline", 5.0, 0.75),
        ("transition_positive", 7.0, 2.25),
        ("transition_negative", 7.0, 2.50),
        ("overturned", 9.0, 2.50),
    )
    meshes = ((120, 16), (180, 24), (240, 32))
    for name, length_ratio, relief_ratio in cases:
        for nx, nz in meshes:
            print(
                f"\nVALIDATION {name}: Lr/Ll={length_ratio:g}, "
                f"relief/H={relief_ratio:g}, mesh={nx}x{nz}",
                flush=True,
            )
            row, result = run_case(length_ratio, relief_ratio, nx, nz)
            row["case"] = name
            row["marker_variant"] = "81 sources, 50 yr"
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
                if name == "overturned":
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


def coordinate_edges(values: np.ndarray) -> np.ndarray:
    """Cell edges for a one-dimensional array of centers."""
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
        r"surface $s(x)=b(x)+H$",
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

    axis.text(
        0.02,
        0.97,
        (
            r"$b(x)=\Delta b\,\sin^2[\pi |x|/(2L_{\rm side})]$"
            "\n"
            r"$L_{\rm side}=L_{\rm left}$ for $x<0$; "
            r"$L_{\rm side}=L_{\rm right}$ for $x>0$"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="0.25",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 1.5,
        },
    )
    axis.text(
        0.98,
        0.62,
        "schematic; vertical scale exaggerated",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="0.35",
        style="italic",
    )
    axis.set(
        xlim=(-1.43, 2.68),
        ylim=(-0.49, 1.55),
        title="(a) Geometry and parameter definitions",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def make_figure(table: pd.DataFrame) -> None:
    """Plot the model geometry and minimum J phase map side by side."""
    ratios = np.sort(table["right_to_left_length_ratio"].unique())
    reliefs = np.sort(table["relief_to_thickness_ratio"].unique())
    values = np.full((len(reliefs), len(ratios)), np.nan)
    invalid = np.ones_like(values, dtype=bool)
    for _, row in table.iterrows():
        iy = int(np.flatnonzero(reliefs == row["relief_to_thickness_ratio"])[0])
        ix = int(np.flatnonzero(ratios == row["right_to_left_length_ratio"])[0])
        if bool(row.get("valid", False)) and np.isfinite(row["minimum_Jx"]):
            values[iy, ix] = row["minimum_Jx"]
            invalid[iy, ix] = False

    finite = values[np.isfinite(values)]
    if not finite.size:
        raise RuntimeError("No valid phase-map cells to plot")
    limit = max(1.0, float(np.max(np.abs(finite))))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.75")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig = plt.figure(figsize=(13.2, 5.9), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.05, 1.75),
        height_ratios=(1.0, 0.07),
    )
    schematic_axis = fig.add_subplot(grid[0, 0])
    axis = fig.add_subplot(grid[0, 1])
    caption_axis = fig.add_subplot(grid[1, :])
    caption_axis.axis("off")
    draw_geometry_schematic(schematic_axis)
    image = axis.pcolormesh(
        coordinate_edges(ratios),
        coordinate_edges(reliefs),
        np.ma.masked_invalid(values),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="0.85",
        linewidth=0.35,
    )
    if np.nanmin(values) < 0.0 < np.nanmax(values):
        contour = axis.contour(
            ratios,
            reliefs,
            values,
            levels=(0.0,),
            colors="black",
            linewidths=1.4,
        )
        axis.clabel(contour, fmt={0.0: "J = 0"}, fontsize=9)
    invalid_y, invalid_x = np.where(invalid)
    if len(invalid_x):
        axis.scatter(
            ratios[invalid_x],
            reliefs[invalid_y],
            marker="x",
            color="0.2",
            linewidth=1.2,
            label="invalid solve/diagnostic",
        )
    axis.set(
        xlabel=r"flank-length ratio $L_{\mathrm{right}}/L_{\mathrm{left}}$",
        ylabel=r"bed-relief ratio $\Delta b/H$",
        title=r"(b) Minimum tangent-map $J=\partial x/\partial x_0$",
        xticks=ratios,
        yticks=reliefs,
    )
    if len(invalid_x):
        axis.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            loc="upper left",
        )
    colorbar = fig.colorbar(image, ax=axis, extend="neither")
    colorbar.set_label(r"minimum variational $J$ over full marker residence")
    caption_axis.text(
        0.5,
        0.5,
        (
            r"Phase map fixes $L_{\rm left}+L_{\rm right}=6$ km, "
            r"$H=200$ m, "
            r"$C=10^{-5}$ m Pa$^{-1}$ yr$^{-1}$"
            r", $\eta=2.5\times10^8$ Pa yr, boundary treatment, and resolution."
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
