"""Restartable steady-characteristic sweep over convergent-basin aspect.

The finite basin has four independent nondimensional controls under the
linear sliding law: flank-length ratio ``q``, relief ratio ``Delta b/H``,
total aspect ``L/H``, and sliding number ``beta H/eta``.  This sweep holds
``H=200 m`` (and therefore ``beta H/eta=0.08``) fixed while varying ``L/H``.
Only ``q>=1`` is solved because exchanging the two flanks is an exact
horizontal reflection that leaves the minimum tangent diagnostic unchanged.

All panels use the validated 25 yr characteristic step.  New aspect cases use
a 480 kyr safety ceiling so that the longest, gentlest basins reach a physical
exit; the reused, uncensored L/H=30 baseline retains its original 120 kyr
ceiling.  In both cases the ceiling is a censoring guard, not an analysis
horizon.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
from pathlib import Path
import time

import numpy as np
import pandas as pd


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_DIRECTORY = NOTEBOOK_DIRECTORY.parent
CORE_PATH = NOTEBOOK_DIRECTORY / "steady_tangent_convergent_parameter_map.py"
DATA_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_aspect_parameter_map.csv"
)
VALIDATION_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_aspect_validation.csv"
)
LEGACY_DATA_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_parameter_map.csv"
)

METHOD_VERSION = "convergent-steady-v3-aspect-central90-ablation-buffer"
THICKNESS_M = 200.0
ASPECT_RATIOS = np.array((15.0, 20.0, 30.0, 45.0, 60.0))
LENGTH_RATIOS = np.r_[np.arange(1.0, 11.0), 12.0]
RELIEF_RATIOS = np.arange(0.25, 4.01, 0.25)
SCREEN_MESH = (180, 24)
REFERENCE_LENGTH_M = 6_000.0
TIME_STEP_YR = 25.0
MAXIMUM_AGE_YR = 480_000.0


def load_core_module():
    """Load the existing validated characteristic implementation."""
    specification = importlib.util.spec_from_file_location(
        "steady_tangent_convergent_core", CORE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import {CORE_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def case_key(aspect_ratio: float, length_ratio: float, relief_ratio: float):
    """Stable rounded key for one grid cell."""
    return (
        round(float(aspect_ratio), 10),
        round(float(length_ratio), 10),
        round(float(relief_ratio), 10),
    )


def add_sweep_columns(row: dict, aspect_ratio: float) -> dict:
    """Attach the independent aspect coordinate and method provenance."""
    result = dict(row)
    result["total_length_to_thickness_ratio"] = float(aspect_ratio)
    result["diagnostic_method_version"] = METHOD_VERSION
    return result


def seed_reference_aspect() -> pd.DataFrame:
    """Reuse the already validated L/H=30 direct solves when available."""
    if not LEGACY_DATA_PATH.exists():
        return pd.DataFrame()
    table = pd.read_csv(LEGACY_DATA_PATH)
    required = {
        "right_to_left_length_ratio",
        "relief_to_thickness_ratio",
        "thickness_m",
        "total_length_m",
        "solution_provenance",
        "valid",
        "status",
        "nx",
        "nz",
        "source_count_per_family",
        "source_flux_lower_quantile",
        "source_flux_upper_quantile",
        "time_step_yr",
        "surface_outflow_buffer_m",
        "safety_cap_trajectory_count",
    }
    if not required.issubset(table.columns):
        return pd.DataFrame()
    mask = (
        np.isclose(table["thickness_m"], THICKNESS_M)
        & np.isclose(table["total_length_m"], REFERENCE_LENGTH_M)
        & (table["right_to_left_length_ratio"] >= 1.0)
        & table["solution_provenance"].astype(str).str.startswith("direct_solve")
        & table["valid"].astype(str).str.lower().eq("true")
        & table["status"].astype(str).eq("valid")
        & np.isclose(table["nx"], SCREEN_MESH[0])
        & np.isclose(table["nz"], SCREEN_MESH[1])
        & np.isclose(table["source_count_per_family"], 81)
        & np.isclose(table["source_flux_lower_quantile"], 0.05)
        & np.isclose(table["source_flux_upper_quantile"], 0.95)
        & np.isclose(table["time_step_yr"], TIME_STEP_YR)
        & np.isclose(table["surface_outflow_buffer_m"], THICKNESS_M / 12.0)
        & np.isclose(table["safety_cap_trajectory_count"].fillna(np.inf), 0.0)
    )
    seeded = table.loc[mask].copy()
    if len(seeded) != len(LENGTH_RATIOS) * len(RELIEF_RATIOS):
        return pd.DataFrame()
    seeded["total_length_to_thickness_ratio"] = 30.0
    seeded["diagnostic_method_version"] = METHOD_VERSION
    seeded["sampler_nx"] = 1201
    seeded["sampler_nz"] = 101
    return seeded


def load_restart_table() -> pd.DataFrame:
    """Load current output or initialize it from the validated baseline."""
    if DATA_PATH.exists():
        table = pd.read_csv(DATA_PATH)
    else:
        table = seed_reference_aspect()
    if table.empty:
        return table
    for column in (
        "total_length_to_thickness_ratio",
        "right_to_left_length_ratio",
        "relief_to_thickness_ratio",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    reference_mask = np.isclose(
        table["total_length_to_thickness_ratio"], 30.0
    )
    table.loc[
        reference_mask & table["sampler_nx"].isna(), "sampler_nx"
    ] = 1201
    table.loc[
        reference_mask & table["sampler_nz"].isna(), "sampler_nz"
    ] = 101
    table = (
        table.drop_duplicates(
            subset=(
                "total_length_to_thickness_ratio",
                "right_to_left_length_ratio",
                "relief_to_thickness_ratio",
            ),
            keep="last",
        )
        .sort_values(
            [
                "total_length_to_thickness_ratio",
                "relief_to_thickness_ratio",
                "right_to_left_length_ratio",
            ]
        )
        .reset_index(drop=True)
    )
    return table


def run_worker(arguments: tuple[float, float, float, int, int]) -> dict:
    """Solve one independent aspect-map cell in a worker process."""
    aspect_ratio, length_ratio, relief_ratio, nx, nz = arguments
    core = load_core_module()
    core.THICKNESS = THICKNESS_M
    core.TOTAL_LENGTH = float(aspect_ratio) * THICKNESS_M
    core.MAXIMUM_AGE_YR = MAXIMUM_AGE_YR
    row = None
    attempts = (
        (int(nx), 1201),
        (int(nx), 1199),
        (int(nx), 1203),
        (int(nx) + 1, 1201),
    )
    for attempted_nx, sampler_nx in attempts:
        row, _ = core.run_case(
            float(length_ratio),
            float(relief_ratio),
            attempted_nx,
            int(nz),
            time_step=TIME_STEP_YR,
            sampler_nx=sampler_nx,
        )
        if row.get("status") != "PointNotInDomainError":
            if (attempted_nx, sampler_nx) != (int(nx), 1201) and row.get(
                "valid", False
            ):
                row["solution_provenance"] = (
                    "direct_solve_point_location_retry_"
                    f"nx_{attempted_nx}_sampler_{sampler_nx}"
                )
            break
    if row is None:
        raise RuntimeError("mesh-edge retry loop produced no result")
    return add_sweep_columns(row, aspect_ratio)


def write_restart_table(rows: list[dict]) -> pd.DataFrame:
    """De-duplicate, sort, and atomically rewrite the restart table."""
    table = pd.DataFrame(rows)
    table = (
        table.drop_duplicates(
            subset=(
                "total_length_to_thickness_ratio",
                "right_to_left_length_ratio",
                "relief_to_thickness_ratio",
            ),
            keep="last",
        )
        .sort_values(
            [
                "total_length_to_thickness_ratio",
                "relief_to_thickness_ratio",
                "right_to_left_length_ratio",
            ]
        )
        .reset_index(drop=True)
    )
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATA_PATH.with_suffix(".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(DATA_PATH)
    return table


def row_is_complete(row: dict) -> bool:
    """Accept only uncensored successful rows as restart-complete."""
    try:
        safety_caps = float(row.get("safety_cap_trajectory_count", np.nan))
    except (TypeError, ValueError):
        return False
    return (
        row.get("diagnostic_method_version") == METHOD_VERSION
        and str(row.get("status")) == "valid"
        and np.isfinite(safety_caps)
        and safety_caps == 0.0
    )


def run_sweep(workers: int) -> pd.DataFrame:
    """Complete every missing grid cell, preserving restartable output."""
    existing = load_restart_table()
    rows = existing.to_dict("records") if not existing.empty else []
    completed = {
        case_key(
            row["total_length_to_thickness_ratio"],
            row["right_to_left_length_ratio"],
            row["relief_to_thickness_ratio"],
        )
        for row in rows
        if row_is_complete(row)
    }
    cases = [
        (aspect, q, relief, SCREEN_MESH[0], SCREEN_MESH[1])
        for aspect in ASPECT_RATIOS
        for relief in RELIEF_RATIOS
        for q in LENGTH_RATIOS
        if case_key(aspect, q, relief) not in completed
    ]
    print(
        f"Aspect sweep: {len(completed)} retained, {len(cases)} to run",
        flush=True,
    )
    if not cases:
        return write_restart_table(rows)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(run_worker, case): case for case in cases}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            aspect, q, relief, _, _ = futures[future]
            row = future.result()
            rows.append(row)
            write_restart_table(rows)
            elapsed = time.perf_counter() - started
            print(
                "ASPECT MAP COMPLETE "
                f"{completed_count}/{len(cases)}; L/H={aspect:g}, "
                f"q={q:g}, relief/H={relief:g}: {row.get('status')}, "
                f"J={row.get('minimum_resolved_J', np.nan):.5g}; "
                f"elapsed={elapsed / 60.0:.1f} min",
                flush=True,
            )
    return write_restart_table(rows)


def run_validation() -> pd.DataFrame:
    """Check the re-entrant and long-residence sentinels."""
    core = load_core_module()
    scenarios = (
        ("short-aspect low-relief overturn", 15.0, 12.0, 0.25),
        ("short-aspect high-relief ordered", 15.0, 12.0, 4.0),
        ("intermediate-aspect closest-to-zero ordered", 45.0, 12.0, 4.0),
        ("long-aspect long-residence ordered", 60.0, 12.0, 0.25),
    )
    variants = (
        ("production", 180, 24, 25.0, 1201, 101),
        ("mesh refined", 240, 32, 25.0, 1201, 101),
        ("time-step refined", 180, 24, 12.5, 1201, 101),
        ("sampler refined", 180, 24, 25.0, 1801, 151),
    )
    if VALIDATION_PATH.exists():
        existing = pd.read_csv(VALIDATION_PATH)
        rows = existing.to_dict("records")
    else:
        rows = []
    completed = {
        (str(row.get("validation_scenario")), str(row.get("validation_variant")))
        for row in rows
        if str(row.get("status")) == "valid"
    }
    fixed_clearance = 2.0 * THICKNESS_M / SCREEN_MESH[1]
    for scenario, aspect_ratio, q_value, relief_ratio in scenarios:
        core.THICKNESS = THICKNESS_M
        core.TOTAL_LENGTH = aspect_ratio * THICKNESS_M
        core.MAXIMUM_AGE_YR = MAXIMUM_AGE_YR
        for variant, nx, nz, time_step, sampler_nx, sampler_nz in variants:
            if (scenario, variant) in completed:
                continue
            row, _ = core.run_case(
                q_value,
                relief_ratio,
                nx,
                nz,
                time_step=time_step,
                outflow_buffer_m=fixed_clearance,
                sampler_nx=sampler_nx,
                sampler_nz=sampler_nz,
            )
            row = add_sweep_columns(row, aspect_ratio)
            row.update(
                {
                    "validation_scenario": scenario,
                    "validation_variant": variant,
                }
            )
            rows.append(row)
            print(
                f"VALIDATION {scenario}; {variant}: {row.get('status')}, "
                f"J={row.get('minimum_resolved_J', np.nan):.6g}",
                flush=True,
            )
    table = (
        pd.DataFrame(rows)
        .drop_duplicates(
            subset=("validation_scenario", "validation_variant"), keep="last"
        )
        .sort_values(["validation_scenario", "validation_variant"])
        .reset_index(drop=True)
    )
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(VALIDATION_PATH, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate_only:
        table = run_validation()
        print(f"Wrote {VALIDATION_PATH} ({len(table)} variants)")
        return
    table = run_sweep(arguments.workers)
    expected = len(ASPECT_RATIOS) * len(LENGTH_RATIOS) * len(RELIEF_RATIOS)
    print(f"Wrote {DATA_PATH} ({len(table)}/{expected} cells)")


if __name__ == "__main__":
    main()
