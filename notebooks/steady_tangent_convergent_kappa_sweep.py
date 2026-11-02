"""Restartable sweep over bed--surface coupling in a convergent basin.

The geometry holds total length ``L=6000 m`` and domain-mean vertical ice
thickness ``H=200 m`` fixed while varying

    kappa = Delta s / Delta b.

For the piecewise-sine flank shape ``f`` with exact domain mean 1/2,

    b = Delta b f,
    s = H + (1-kappa) Delta b / 2 + kappa Delta b f.

Thus kappa=1 exactly recovers the constant-thickness calculation and kappa=0
is the stagnant hydrostatic limit.  Only kappa>0 is solved.  The RK4 step and
safety age scale as 1/kappa, preserving approximately the validated spatial
and finite-strain increment as the velocity scale decreases.
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
    / "data/steady_tangent_convergent_kappa_parameter_map.csv"
)
BASELINE_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_aspect_parameter_map.csv"
)

METHOD_VERSION = (
    "convergent-steady-v4-kappa-meanH-central90-local-ablation-buffer"
)
MEAN_THICKNESS_M = 200.0
TOTAL_LENGTH_M = 6_000.0
KAPPA_VALUES = np.array((0.10, 0.25, 0.50, 0.75, 1.00))
LENGTH_RATIOS = np.r_[np.arange(1.0, 11.0), 12.0]
RELIEF_RATIOS = np.arange(0.25, 4.01, 0.25)
SCREEN_MESH = (180, 24)
SOURCE_COUNT = 81
BASE_TIME_STEP_YR = 25.0
BASE_MAXIMUM_AGE_YR = 240_000.0
SOURCE_FLUX_QUANTILES = (0.05, 0.95)


def load_core_module():
    """Load the validated steady-characteristic implementation."""
    specification = importlib.util.spec_from_file_location(
        "steady_tangent_convergent_kappa_core", CORE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import {CORE_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.THICKNESS = MEAN_THICKNESS_M
    module.TOTAL_LENGTH = TOTAL_LENGTH_M
    module.PRINT_CASE_TRACEBACKS = False
    return module


def case_key(kappa: float, length_ratio: float, relief_ratio: float):
    """Stable rounded key for one grid cell."""
    return tuple(
        round(float(value), 10)
        for value in (kappa, length_ratio, relief_ratio)
    )


def minimum_thickness_m(kappa: float, relief_ratio: float) -> float:
    """Exact minimum thickness for the mean-thickness parameterization."""
    return MEAN_THICKNESS_M * (
        1.0 - 0.5 * abs(1.0 - kappa) * relief_ratio
    )


def add_method_columns(row: dict, kappa: float) -> dict:
    """Attach sweep provenance to a direct or analytically invalid row."""
    result = dict(row)
    result["surface_to_bed_relief_ratio_kappa"] = float(kappa)
    result["total_length_to_thickness_ratio"] = (
        TOTAL_LENGTH_M / MEAN_THICKNESS_M
    )
    result["diagnostic_method_version"] = METHOD_VERSION
    return result


def seed_kappa_one() -> pd.DataFrame:
    """Reuse the already validated constant-thickness L/H=30 slice."""
    if not BASELINE_PATH.exists():
        return pd.DataFrame()
    table = pd.read_csv(BASELINE_PATH)
    required = {
        "total_length_to_thickness_ratio",
        "right_to_left_length_ratio",
        "relief_to_thickness_ratio",
        "thickness_m",
        "total_length_m",
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
        np.isclose(table["total_length_to_thickness_ratio"], 30.0)
        & np.isclose(table["thickness_m"], MEAN_THICKNESS_M)
        & np.isclose(table["total_length_m"], TOTAL_LENGTH_M)
        & (table["right_to_left_length_ratio"] >= 1.0)
        & table["valid"].astype(str).str.lower().eq("true")
        & table["status"].astype(str).eq("valid")
        & table["nx"].isin((SCREEN_MESH[0], SCREEN_MESH[0] + 1))
        & np.isclose(table["nz"], SCREEN_MESH[1])
        & np.isclose(table["source_count_per_family"], SOURCE_COUNT)
        & np.isclose(table["source_flux_lower_quantile"], 0.05)
        & np.isclose(table["source_flux_upper_quantile"], 0.95)
        & np.isclose(table["time_step_yr"], BASE_TIME_STEP_YR)
        & np.isclose(
            table["surface_outflow_buffer_m"],
            2.0 * MEAN_THICKNESS_M / SCREEN_MESH[1],
        )
        & np.isclose(
            table["safety_cap_trajectory_count"].fillna(np.inf), 0.0
        )
    )
    seeded = table.loc[mask].copy()
    expected = len(LENGTH_RATIOS) * len(RELIEF_RATIOS)
    if len(seeded) != expected:
        return pd.DataFrame()
    seeded["surface_to_bed_relief_ratio_kappa"] = 1.0
    seeded["surface_relief_m"] = seeded["relief_m"]
    seeded["mean_thickness_m"] = MEAN_THICKNESS_M
    for column in (
        "center_thickness_m",
        "divide_thickness_m",
        "minimum_thickness_m",
        "maximum_thickness_m",
    ):
        seeded[column] = MEAN_THICKNESS_M
    seeded["surface_outflow_buffer_fraction_of_local_thickness"] = (
        seeded["surface_outflow_buffer_m"] / MEAN_THICKNESS_M
    )
    seeded["minimum_local_surface_outflow_buffer_m"] = seeded[
        "surface_outflow_buffer_m"
    ]
    seeded["maximum_local_surface_outflow_buffer_m"] = seeded[
        "surface_outflow_buffer_m"
    ]
    seeded["source_accumulation_threshold_m_per_yr"] = 1.0e-5
    seeded["path_counterflow_velocity_threshold_m_per_yr"] = 1.0e-5
    seeded["sampler_sigma_lower"] = 0.002
    seeded["sampler_sigma_upper"] = 0.999
    seeded["maximum_short_surface_slope"] = seeded[
        "maximum_short_flank_slope"
    ]
    seeded["diagnostic_method_version"] = METHOD_VERSION
    seeded["solution_provenance"] = "reused_validated_kappa_one"
    return seeded


def load_restart_table() -> pd.DataFrame:
    """Load a prior restart or initialize it with the kappa=1 baseline."""
    table = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else seed_kappa_one()
    if table.empty:
        return table
    for column in (
        "surface_to_bed_relief_ratio_kappa",
        "right_to_left_length_ratio",
        "relief_to_thickness_ratio",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if "sampler_sigma_lower" not in table:
        table["sampler_sigma_lower"] = 0.002
    else:
        table["sampler_sigma_lower"] = table["sampler_sigma_lower"].fillna(
            0.002
        )
    if "sampler_sigma_upper" not in table:
        table["sampler_sigma_upper"] = 0.999
    else:
        table["sampler_sigma_upper"] = table["sampler_sigma_upper"].fillna(
            0.999
        )
    if "path_counterflow_velocity_threshold_m_per_yr" not in table:
        table["path_counterflow_velocity_threshold_m_per_yr"] = (
            1.0e-5 * table["surface_to_bed_relief_ratio_kappa"].abs()
        )
    else:
        table["path_counterflow_velocity_threshold_m_per_yr"] = table[
            "path_counterflow_velocity_threshold_m_per_yr"
        ].fillna(
            1.0e-5 * table["surface_to_bed_relief_ratio_kappa"].abs()
        )
    invalid = table["status"].astype(str).eq(
        "invalid_nonpositive_thickness"
    )
    table.loc[invalid, "solution_provenance"] = (
        "analytical_invalid_nonpositive_thickness"
    )
    return (
        table.drop_duplicates(
            subset=(
                "surface_to_bed_relief_ratio_kappa",
                "right_to_left_length_ratio",
                "relief_to_thickness_ratio",
            ),
            keep="last",
        )
        .sort_values(
            [
                "surface_to_bed_relief_ratio_kappa",
                "relief_to_thickness_ratio",
                "right_to_left_length_ratio",
            ]
        )
        .reset_index(drop=True)
    )


def run_worker(arguments: tuple[float, float, float, int, int]) -> dict:
    """Solve one independent kappa-map cell with a mesh-edge retry."""
    kappa, length_ratio, relief_ratio, nx, nz = arguments
    core = load_core_module()
    time_step = BASE_TIME_STEP_YR / kappa
    maximum_age = BASE_MAXIMUM_AGE_YR / kappa
    attempts = (
        (int(nx), 1201, 0.002, 0.999),
        (int(nx), 1201, 0.005, 0.999),
        (int(nx), 1201, 0.005, 0.990),
        (int(nx), 1201, 0.010, 0.980),
        (int(nx), 1201, 0.010, 0.999),
        (int(nx), 1199, 0.010, 0.999),
        (int(nx) + 1, 1201, 0.010, 0.999),
        (int(nx), 1201, 0.020, 0.995),
        (int(nx), 1201, 0.020, 0.950),
    )
    row = None
    for (
        attempted_nx,
        sampler_nx,
        sampler_sigma_lower,
        sampler_sigma_upper,
    ) in attempts:
        row, _ = core.run_case(
            float(length_ratio),
            float(relief_ratio),
            attempted_nx,
            int(nz),
            source_count=SOURCE_COUNT,
            time_step=time_step,
            outflow_buffer_m=2.0 * MEAN_THICKNESS_M / nz,
            source_flux_quantiles=SOURCE_FLUX_QUANTILES,
            sampler_nx=sampler_nx,
            sampler_nz=101,
            sampler_sigma_lower=sampler_sigma_lower,
            sampler_sigma_upper=sampler_sigma_upper,
            kappa=float(kappa),
            maximum_age_yr=maximum_age,
        )
        if row.get("status") != "PointNotInDomainError":
            if (
                (
                    attempted_nx,
                    sampler_nx,
                    sampler_sigma_lower,
                    sampler_sigma_upper,
                )
                != (int(nx), 1201, 0.002, 0.999)
                and row.get("valid", False)
            ):
                row["solution_provenance"] = (
                    "direct_solve_point_location_retry_"
                    f"nx_{attempted_nx}_sampler_{sampler_nx}_"
                    f"sigma_{sampler_sigma_lower:g}_"
                    f"{sampler_sigma_upper:g}"
                )
            break
    if row is None:
        raise RuntimeError("mesh-edge retry loop produced no result")
    return add_method_columns(row, kappa)


def write_restart_table(rows: list[dict]) -> pd.DataFrame:
    """De-duplicate, sort, and atomically rewrite the restart table."""
    table = pd.DataFrame(rows)
    table = (
        table.drop_duplicates(
            subset=(
                "surface_to_bed_relief_ratio_kappa",
                "right_to_left_length_ratio",
                "relief_to_thickness_ratio",
            ),
            keep="last",
        )
        .sort_values(
            [
                "surface_to_bed_relief_ratio_kappa",
                "relief_to_thickness_ratio",
                "right_to_left_length_ratio",
            ]
        )
        .reset_index(drop=True)
    )
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = DATA_PATH.with_suffix(".tmp")
    table.to_csv(temporary_path, index=False)
    temporary_path.replace(DATA_PATH)
    return table


def is_complete(row: pd.Series) -> bool:
    """Return whether a restart row is resolved or analytically invalid."""
    if str(row.get("diagnostic_method_version", "")) != METHOD_VERSION:
        return False
    status = str(row.get("status", ""))
    valid = str(row.get("valid", "")).lower() == "true" or bool(
        row.get("valid", False) is True
    )
    return valid or status == "invalid_nonpositive_thickness"


def run_screen(workers: int) -> pd.DataFrame:
    """Run missing cells and save after every completed future."""
    table = load_restart_table()
    rows = table.to_dict("records") if not table.empty else []
    complete = {
        case_key(
            row["surface_to_bed_relief_ratio_kappa"],
            row["right_to_left_length_ratio"],
            row["relief_to_thickness_ratio"],
        )
        for _, row in table.iterrows()
        if is_complete(row)
    }
    tasks = []
    for kappa in KAPPA_VALUES:
        for relief_ratio in RELIEF_RATIOS:
            for length_ratio in LENGTH_RATIOS:
                key = case_key(kappa, length_ratio, relief_ratio)
                if key not in complete:
                    tasks.append(
                        (
                            float(kappa),
                            float(length_ratio),
                            float(relief_ratio),
                            *SCREEN_MESH,
                        )
                    )
    print(
        f"kappa sweep: {len(complete)} complete, {len(tasks)} pending, "
        f"{workers} workers",
        flush=True,
    )
    if not tasks:
        return write_restart_table(rows)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(run_worker, task): task for task in tasks
        }
        for completed, future in enumerate(as_completed(future_to_task), 1):
            task = future_to_task[future]
            try:
                row = future.result()
            except Exception as error:
                kappa, length_ratio, relief_ratio, nx, nz = task
                row = add_method_columns(
                    {
                        "right_to_left_length_ratio": length_ratio,
                        "relief_to_thickness_ratio": relief_ratio,
                        "mean_thickness_m": MEAN_THICKNESS_M,
                        "total_length_m": TOTAL_LENGTH_M,
                        "minimum_thickness_m": minimum_thickness_m(
                            kappa, relief_ratio
                        ),
                        "nx": nx,
                        "nz": nz,
                        "valid": False,
                        "status": type(error).__name__,
                        "error": str(error),
                        "minimum_resolved_J": np.nan,
                        "minimum_raw_J": np.nan,
                    },
                    kappa,
                )
            rows.append(row)
            table = write_restart_table(rows)
            if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"{completed}/{len(tasks)} in {elapsed / 60.0:.1f} min; "
                    f"last kappa={task[0]:g}, q={task[1]:g}, "
                    f"relief/H={task[2]:g}, status={row.get('status')}",
                    flush=True,
                )
    return table


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_screen(max(1, arguments.workers))


if __name__ == "__main__":
    main()
