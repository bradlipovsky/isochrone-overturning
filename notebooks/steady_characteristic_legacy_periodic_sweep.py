"""Lifetime steady-characteristic audit of the legacy periodic bed screen.

This revisits the 15 geometries defined in
``realistic_geometry_parameter_search.py`` without a time-dependent cohort.
The Stokes solve, exact-surface source selection, equal-age tangent equation,
and individual first-exit stopping rule are imported from the current
sinusoidal-bed diagnostic.  Only the bed profile changes.  The basal law is
therefore the same passive linear sliding law, with beta = 1e5 Pa yr m^-1.

The CSV is checkpointed after every completed case and can be restarted.
At most two isolated Firedrake workers are allowed deliberately.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cached_property
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from realistic_geometry_parameter_search import GEOMETRIES
from steady_characteristic_sinusoidal_parameter_map import (
    BASAL_DRAG,
    DIAGNOSTIC_METHOD_VERSION,
    NITSCHE_PENALTY,
    SURFACE_SLOPE,
    VISCOSITY,
    solve_parameter_case,
)


METHOD_VERSION = "legacy-periodic-steady-characteristic-v1"
PRODUCTION = {
    "mesh_nx": 72,
    "mesh_nz": 18,
    "sampler_nx": 481,
    "sampler_nz": 161,
    "source_count": 321,
    "time_step": 10.0,
    "maximum_age": 1_000_000.0,
}
VALIDATION_INDICES = (4, 6, 11)
VALIDATION_LEVELS = (
    ("coarse mesh", 48, 12, 481, 161, 321, 10.0),
    ("fine mesh", 96, 24, 481, 161, 321, 10.0),
    ("fine mesh + trace", 96, 24, 641, 241, 481, 5.0),
)


def periodic_distance(x, center, length):
    """Shortest signed distance on the periodic model cell."""
    return (
        (np.asarray(x, dtype=float) - center + 0.5 * length) % length
        - 0.5 * length
    )


@dataclass(frozen=True)
class LegacyPeriodicGeometry:
    """Adapter exposing one legacy bed to the current periodic diagnostic."""

    index: int

    @property
    def definition(self):
        return GEOMETRIES[self.index]

    @property
    def name(self):
        return str(self.definition["name"])

    @property
    def wavelength(self):
        return float(self.definition["length"])

    @property
    def phase_origin(self):
        return 0.0

    def thickness(self, x):
        x = np.asarray(x, dtype=float)
        result = np.full_like(x, float(self.definition["background thickness"]))
        for item in self.definition["features"]:
            distance = periodic_distance(x, item["center"], self.wavelength)
            width = np.where(
                distance < 0.0, item["width left"], item["width right"]
            )
            result += item["amplitude"] * np.exp(
                -(np.abs(distance) / width) ** item["exponent"]
            )
        return result

    def thickness_slope(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        for item in self.definition["features"]:
            distance = periodic_distance(x, item["center"], self.wavelength)
            width = np.where(
                distance < 0.0, item["width left"], item["width right"]
            )
            absolute = np.abs(distance)
            exponent = float(item["exponent"])
            shape = np.exp(-(absolute / width) ** exponent)
            derivative = (
                -exponent
                * (absolute / width) ** (exponent - 1.0)
                / width
                * np.sign(distance)
            )
            result += item["amplitude"] * shape * derivative
        return result

    def bed_height(self, x):
        return -self.thickness(x)

    @cached_property
    def _sample(self):
        x = np.linspace(0.0, self.wavelength, 200_001, endpoint=False)
        return x, self.thickness(x)

    @property
    def mean_thickness(self):
        return float(np.mean(self._sample[1]))

    @property
    def minimum_thickness(self):
        return float(np.min(self._sample[1]))

    @property
    def maximum_thickness(self):
        return float(np.max(self._sample[1]))

    @property
    def relief(self):
        return self.maximum_thickness - self.minimum_thickness

    @property
    def amplitude(self):
        return 0.5 * self.relief

    @property
    def amplitude_ratio(self):
        return self.amplitude / self.mean_thickness

    @property
    def wavelength_ratio(self):
        return self.wavelength / self.mean_thickness

    @property
    def peak_bed_slope(self):
        return float(np.max(np.abs(self.thickness_slope(self._sample[0]))))


def task_tuple(index, resolution, controls):
    return (
        int(index),
        str(resolution),
        int(controls["mesh_nx"]),
        int(controls["mesh_nz"]),
        int(controls["sampler_nx"]),
        int(controls["sampler_nz"]),
        int(controls["source_count"]),
        float(controls["time_step"]),
        float(controls["maximum_age"]),
    )


def validation_tasks(maximum_age):
    tasks = []
    for index in VALIDATION_INDICES:
        for label, mx, mz, sx, sz, sources, step in VALIDATION_LEVELS:
            controls = {
                "mesh_nx": mx,
                "mesh_nz": mz,
                "sampler_nx": sx,
                "sampler_nz": sz,
                "source_count": sources,
                "time_step": step,
                "maximum_age": maximum_age,
            }
            tasks.append(task_tuple(index, label, controls))
    return tasks


def task_key(task):
    return tuple(task[:-1])


def row_key(row):
    return (
        int(row["legacy geometry index"]),
        str(row["resolution"]),
        int(row["mesh nx"]),
        int(row["mesh nz"]),
        int(row["sampler nx"]),
        int(row["sampler nz"]),
        int(row["candidate source count"]),
        float(row["characteristic time step (yr)"]),
    )


def solve_task(task):
    index, resolution, mx, mz, sx, sz, sources, step, maximum_age = task
    geometry = LegacyPeriodicGeometry(index)
    row = solve_parameter_case(
        geometry,
        mesh_nx=mx,
        mesh_nz=mz,
        vertical_grading_power=1.0,
        sampler_nx=sx,
        sampler_nz=sz,
        sampler_vertical_grading_power=1.0,
        source_count=sources,
        retained_flux_fraction=0.80,
        time_step=step,
        maximum_age=maximum_age,
        minimum_surface_signal_to_basal_leakage=10.0,
    )
    return {
        "legacy diagnostic method version": METHOD_VERSION,
        "legacy geometry index": index,
        "geometry": geometry.name,
        "resolution": resolution,
        **row,
    }


def sorted_table(rows):
    return pd.DataFrame(rows).sort_values(
        [
            "legacy geometry index",
            "mesh nx",
            "sampler nx",
            "characteristic time step (yr)",
        ]
    ).reset_index(drop=True)


def write_metadata(table, output, mode):
    successful = table.loc[table["status"].eq("success")]
    minimum = None
    if len(successful):
        row = successful.loc[successful["minimum J"].idxmin()]
        minimum = {
            "geometry": str(row["geometry"]),
            "minimum J": float(row["minimum J"]),
            "raw minimum J to exit": float(row["raw minimum J to exit"]),
            "minimum mechanism C before reversal": float(
                row["minimum mechanism C before reversal"]
            ),
        }
    metadata = {
        "method": METHOD_VERSION,
        "imported characteristic method": DIAGNOSTIC_METHOD_VERSION,
        "mode": mode,
        "row count": int(len(table)),
        "status counts": {
            str(k): int(v) for k, v in table["status"].value_counts().items()
        },
        "minimum successful case": minimum,
        "physics": {
            "viscosity (Pa yr)": VISCOSITY,
            "basal drag (Pa yr/m)": BASAL_DRAG,
            "surface slope (rad)": SURFACE_SLOPE,
            "Nitsche penalty": NITSCHE_PENALTY,
        },
        "source convention": "central 80% of positive accumulation flux in every interval",
        "stopping convention": "each characteristic stops at its own first surface/bed exit",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_tasks(tasks, output, mode, jobs):
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if output.exists():
        rows = pd.read_csv(output).to_dict("records")
    # Keep the last checkpoint for each control tuple.  Safety-censored and
    # failed cases are retried, so raising --maximum-age can finish them.
    rows_by_key = {row_key(row): row for row in rows}
    rows = list(rows_by_key.values())
    completed = {
        row_key(row)
        for row in rows
        if str(row.get("status", ""))
        not in {
            "characteristic safety-cap censoring",
            "numerical characteristic failure",
            "numerical bed exit",
        }
        and not str(row.get("status", "")).startswith("failed:")
    }
    pending = [task for task in tasks if task_key(task) not in completed]
    print(f"{len(rows)} completed; {len(pending)} pending", flush=True)

    def record(row):
        key = row_key(row)
        rows[:] = [existing for existing in rows if row_key(existing) != key]
        rows.append(row)
        table = sorted_table(rows)
        table.to_csv(output, index=False, na_rep="NA")
        write_metadata(table, output, mode)
        print(
            f"{row['geometry']}: {row['status']}; "
            f"J={row.get('minimum J', np.nan):.6g}; "
            f"raw={row.get('raw minimum J to exit', np.nan):.6g}",
            flush=True,
        )

    if jobs == 1:
        for task in pending:
            record(solve_task(task))
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=jobs) as pool:
            for row in pool.imap_unordered(solve_task, pending):
                record(row)
    table = sorted_table(rows)
    write_metadata(table, output, mode)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("production", "validation"), default="production")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--maximum-age", type=float, default=PRODUCTION["maximum_age"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index", type=int, action="append")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 2:
        parser.error("--jobs must be 1 or 2")
    root = Path(__file__).resolve().parents[1]
    if args.mode == "production":
        indices = args.index if args.index is not None else range(len(GEOMETRIES))
        controls = dict(PRODUCTION, maximum_age=args.maximum_age)
        tasks = [task_tuple(i, "production", controls) for i in indices]
        default = root / "data" / "steady_characteristic_legacy_periodic_sweep.csv"
    else:
        if args.index is not None:
            parser.error("--index is only available in production mode")
        tasks = validation_tasks(args.maximum_age)
        default = root / "data" / "steady_characteristic_legacy_periodic_validation.csv"
    run_tasks(tasks, args.output or default, args.mode, args.jobs)


if __name__ == "__main__":
    main()
