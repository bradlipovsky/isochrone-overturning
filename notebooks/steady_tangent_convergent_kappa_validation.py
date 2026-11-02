"""Targeted validation for the convergent-basin kappa sweep."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
from pathlib import Path

import pandas as pd


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_DIRECTORY = NOTEBOOK_DIRECTORY.parent
CORE_PATH = NOTEBOOK_DIRECTORY / "steady_tangent_convergent_parameter_map.py"
OUTPUT_PATH = (
    REPOSITORY_DIRECTORY
    / "data/steady_tangent_convergent_kappa_validation.csv"
)
MEAN_THICKNESS_M = 200.0
TOTAL_LENGTH_M = 6_000.0


def load_core():
    specification = importlib.util.spec_from_file_location(
        "steady_tangent_convergent_kappa_validation_core", CORE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import {CORE_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.THICKNESS = MEAN_THICKNESS_M
    module.TOTAL_LENGTH = TOTAL_LENGTH_M
    module.PRINT_CASE_TRACEBACKS = False
    return module


def task(
    scenario,
    variant,
    kappa,
    q,
    relief,
    nx=180,
    nz=24,
    time_step_factor=1.0,
    source_count=81,
    sampler_nx=1201,
    sampler_nz=101,
    sigma_lower=0.002,
    sigma_upper=0.999,
):
    return {
        "scenario": scenario,
        "variant": variant,
        "kappa": kappa,
        "q": q,
        "relief": relief,
        "nx": nx,
        "nz": nz,
        "time_step": 25.0 / kappa * time_step_factor,
        "source_count": source_count,
        "sampler_nx": sampler_nx,
        "sampler_nz": sampler_nz,
        "sigma_lower": sigma_lower,
        "sigma_upper": sigma_upper,
    }


TASKS = (
    task("low-kappa onset", "screen", 0.10, 8, 1.50),
    task(
        "low-kappa onset", "half time step", 0.10, 8, 1.50,
        time_step_factor=0.5,
    ),
    task(
        "low-kappa onset", "fine mesh and sampler", 0.10, 8, 1.50,
        nx=240, nz=32, time_step_factor=0.5,
        sampler_nx=1801, sampler_nz=151,
    ),
    task(
        "low-kappa onset", "161 sources", 0.10, 8, 1.50,
        source_count=161,
    ),
    task("low-kappa strong", "screen", 0.10, 12, 1.25),
    task(
        "low-kappa strong", "half time step", 0.10, 12, 1.25,
        time_step_factor=0.5,
    ),
    task(
        "low-kappa strong", "fine mesh and sampler", 0.10, 12, 1.25,
        nx=240, nz=32, time_step_factor=0.5,
        sampler_nx=1801, sampler_nz=151,
    ),
    task(
        "thin positive-thickness edge", "inset 0.5 percent", 0.10, 12, 2.00,
        sigma_lower=0.005,
    ),
    task(
        "thin positive-thickness edge", "inset 1 percent", 0.10, 12, 2.00,
        sigma_lower=0.010,
    ),
    task(
        "thin positive-thickness edge", "fine mesh", 0.10, 12, 2.00,
        nx=240, nz=32, time_step_factor=0.5,
        sampler_nx=1801, sampler_nz=151, sigma_lower=0.005,
    ),
    task(
        "high-relief sampler retry", "inset 2--95 percent", 0.50, 8, 3.75,
        sigma_lower=0.020, sigma_upper=0.950,
    ),
    task(
        "high-relief sampler retry", "inset 3--94 percent", 0.50, 8, 3.75,
        sigma_lower=0.030, sigma_upper=0.940,
    ),
    task(
        "high-relief sampler retry", "fine mesh", 0.50, 8, 3.75,
        nx=240, nz=32, time_step_factor=0.5,
        sampler_nx=1801, sampler_nz=151,
        sigma_lower=0.020, sigma_upper=0.950,
    ),
    task("kappa 0.75 extreme", "screen", 0.75, 12, 3.75),
    task(
        "kappa 0.75 extreme", "fine mesh and time step", 0.75, 12, 3.75,
        nx=240, nz=32, time_step_factor=0.5,
        sampler_nx=1801, sampler_nz=151,
    ),
)


def run_task(settings):
    core = load_core()
    row, _ = core.run_case(
        settings["q"],
        settings["relief"],
        settings["nx"],
        settings["nz"],
        source_count=settings["source_count"],
        time_step=settings["time_step"],
        outflow_buffer_m=2.0 * MEAN_THICKNESS_M / settings["nz"],
        sampler_nx=settings["sampler_nx"],
        sampler_nz=settings["sampler_nz"],
        sampler_sigma_lower=settings["sigma_lower"],
        sampler_sigma_upper=settings["sigma_upper"],
        kappa=settings["kappa"],
        maximum_age_yr=240_000.0 / settings["kappa"],
    )
    row.update(settings)
    return row


def main():
    rows = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_task, settings): settings for settings in TASKS}
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(
                f"{completed}/{len(TASKS)} {row['scenario']} / {row['variant']}: "
                f"{row['status']}, J={row.get('minimum_resolved_J')}",
                flush=True,
            )
    table = pd.DataFrame(rows).sort_values(["scenario", "variant"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
