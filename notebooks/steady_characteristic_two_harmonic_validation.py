"""Focused convergence checks for the smooth two-harmonic bed sweep.

The production sweep deliberately reaches the thin-neck limit ``A/H=0.95``
and the very short fundamental wavelength ``lambda/H=1/12``.  Those are the
two numerically most demanding directions: the first creates slow paths and
the second gives the second harmonic a wavelength of only ``H/24``.  This
script repeats the important intermediate-, short-, and long-wave cases with
finer meshes, samplers, source spacing, and characteristic time steps.  It
also checks the lowest-J ``A/H=0.85`` shape-scan cell, the raw-minimum wave,
the short-wave grading switch, and source-grid/coordinate-gauge sensitivity
near the ``q=1/4`` shape transition.

The output is a small CSV audit table; no manuscript source is touched.
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

import steady_characteristic_two_harmonic_parameter_sweep as sweep


VALIDATION_METHOD_VERSION = (
    "two-harmonic-focused-convergence-v1__" + sweep.METHOD_VERSION
)


def geometry(
    mean_thickness,
    amplitude_ratio,
    wavelength_ratio,
    second_harmonic_ratio=0.125,
    phase_fraction=0.75,
    phase_origin_fraction=0.0,
):
    return sweep.TwoHarmonicGeometry(
        wavelength=mean_thickness * wavelength_ratio,
        amplitude=mean_thickness * amplitude_ratio,
        mean_thickness=mean_thickness,
        second_harmonic_ratio=second_harmonic_ratio,
        relative_phase=2.0 * np.pi * phase_fraction,
        phase_origin=mean_thickness * wavelength_ratio * phase_origin_fraction,
    )


def scenarios():
    """Return paired production/refined cases at the critical limits."""
    rows = []

    def add(
        name,
        h,
        amplitude_ratio,
        wavelength_ratio,
        refined=False,
        second_harmonic_ratio=0.125,
        phase_fraction=0.75,
        maximum_age=None,
        resolution_label=None,
        grading_power=None,
        source_count=None,
        phase_origin_fraction=0.0,
    ):
        task = sweep.task_for_geometry(
            geometry(
                h,
                amplitude_ratio,
                wavelength_ratio,
                second_harmonic_ratio=second_harmonic_ratio,
                phase_fraction=phase_fraction,
                phase_origin_fraction=phase_origin_fraction,
            ),
            sweep.MAXIMUM_AGE,
        )
        if refined:
            task.update(
                mesh_nx=max(192, int(task["mesh_nx"])),
                mesh_nz=max(144, int(task["mesh_nz"])),
                sampler_nx=max(769, int(task["sampler_nx"])),
                sampler_nz=max(481, int(task["sampler_nz"])),
                source_count=max(481, int(task["source_count"])),
                time_step=min(6.25, float(task["time_step"])),
                maximum_age=max(1_500_000.0, float(task["maximum_age"])),
            )
            if wavelength_ratio <= 1.0 / 12.0 + 1.0e-12:
                task["mesh_nz"] = max(160, int(task["mesh_nz"]))
                task["sampler_nz"] = max(641, int(task["sampler_nz"]))
        if maximum_age is not None:
            task["maximum_age"] = float(maximum_age)
        if grading_power is not None:
            task["mesh_vertical_grading_power"] = float(grading_power)
            task["sampler_vertical_grading_power"] = float(grading_power)
        if source_count is not None:
            task["source_count"] = int(source_count)
        task["validation scenario"] = name
        task["validation resolution"] = resolution_label or (
            "refined" if refined else "production"
        )
        rows.append(task)

    for refined in (False, True):
        add(
            "shape-scan minimum",
            160.0,
            0.85,
            3.0,
            refined=refined,
        )
        add(
            "thin-neck intermediate wave",
            50.0,
            0.95,
            3.0,
            refined=refined,
        )
        add(
            "thin-neck raw-minimum wave",
            50.0,
            0.95,
            2.0,
            refined=refined,
        )
        add(
            "thin-neck shortest wave",
            160.0,
            0.95,
            1.0 / 12.0,
            refined=refined,
        )
        add(
            "thin-neck long wave",
            50.0,
            0.95,
            12.0,
            refined=refined,
        )
        add(
            "slow near-transition harmonic shape",
            160.0,
            0.95,
            3.0,
            refined=refined,
            second_harmonic_ratio=0.24,
            phase_fraction=0.5,
            maximum_age=5_000_000.0,
        )

    # The long-wave thin-neck case exposed a false sign change when quadratic
    # endpoint grading was applied outside its short-wave range.  Retain a
    # middle quasi-uniform mesh as a direct convergence check.
    task = sweep.task_for_geometry(geometry(50.0, 0.95, 12.0), 1_500_000.0)
    task.update(
        mesh_nx=160,
        mesh_nz=120,
        sampler_nx=769,
        sampler_nz=481,
        source_count=481,
        time_step=6.25,
    )
    task["validation scenario"] = "thin-neck long wave"
    task["validation resolution"] = "intermediate mesh"
    rows.append(task)

    # Audit both sides of the p=2 to p=1 switch so that the numerical policy
    # itself cannot masquerade as a wavelength transition.
    add(
        "grading switch lambda/H=1/2",
        160.0,
        0.85,
        0.5,
        resolution_label="default p=1",
    )
    add(
        "grading switch lambda/H=1/2",
        160.0,
        0.85,
        0.5,
        resolution_label="graded p=2",
        grading_power=2.0,
    )
    add(
        "grading switch lambda/H=1/2, thin neck",
        160.0,
        0.95,
        0.5,
        resolution_label="default p=1",
    )
    add(
        "grading switch lambda/H=1/2, thin neck",
        160.0,
        0.95,
        0.5,
        resolution_label="graded p=2",
        grading_power=2.0,
    )
    add(
        "grading switch lambda/H=1/4, thin neck",
        160.0,
        0.95,
        0.25,
        resolution_label="default p=2",
    )
    add(
        "grading switch lambda/H=1/4, thin neck",
        160.0,
        0.95,
        0.25,
        resolution_label="uniform p=1",
        grading_power=1.0,
    )
    add(
        "grading switch lambda/H=1",
        160.0,
        0.85,
        1.0,
        resolution_label="default p=1",
    )
    add(
        "grading switch lambda/H=1",
        160.0,
        0.85,
        1.0,
        resolution_label="graded p=2",
        grading_power=2.0,
    )
    add(
        "grading switch lambda/H=1, q=1",
        160.0,
        0.85,
        1.0,
        second_harmonic_ratio=1.0,
        resolution_label="default p=1",
    )
    add(
        "grading switch lambda/H=1, q=1",
        160.0,
        0.85,
        1.0,
        second_harmonic_ratio=1.0,
        resolution_label="graded p=2",
        grading_power=2.0,
    )

    add(
        "slow near-transition harmonic shape",
        160.0,
        0.95,
        3.0,
        second_harmonic_ratio=0.24,
        phase_fraction=0.5,
        maximum_age=5_000_000.0,
        resolution_label="source-count audit",
        source_count=481,
    )
    add(
        "slow near-transition harmonic shape",
        160.0,
        0.95,
        3.0,
        second_harmonic_ratio=0.24,
        phase_fraction=0.5,
        phase_origin_fraction=0.25,
        maximum_age=5_000_000.0,
        resolution_label="quarter-wave gauge",
    )
    return rows


def solve(task):
    task = dict(task)
    scenario = task.pop("validation scenario")
    resolution = task.pop("validation resolution")
    row = sweep.solve_task(task)
    row["validation scenario"] = scenario
    row["validation resolution"] = resolution
    row["validation method version"] = VALIDATION_METHOD_VERSION
    return row


def validation_key_from_task(task):
    return (
        str(task["validation scenario"]),
        str(task["validation resolution"]),
        sweep.task_key(task),
    )


def validation_key_from_row(row):
    return (
        str(row["validation scenario"]),
        str(row["validation resolution"]),
        sweep.row_key(row),
    )


def reusable_rows(output):
    if not output.exists():
        return {}
    table = pd.read_csv(output)
    required = {
        "validation method version",
        "validation scenario",
        "validation resolution",
        "status",
        "minimum J",
        "mesh vertical grading power",
        "sampler vertical grading power",
        "bed phase origin (m; coordinate gauge)",
    }
    if not required.issubset(table.columns):
        return {}
    reusable = {}
    for _, row in table.iterrows():
        status = str(row["status"])
        complete = status == "success" and all(
            pd.notna(row.get(column))
            for column in (
                "maximum horizontal step (wavelengths)",
                "mechanism C at minimum J",
                "maximum relative streamfunction identity residual",
            )
        )
        if (
            complete
            and row["validation method version"] == VALIDATION_METHOD_VERSION
        ):
            reusable[validation_key_from_row(row)] = row.to_dict()
    return reusable


def run(output, jobs):
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = scenarios()
    reusable = reusable_rows(output)
    rows = []
    pending = []
    for task in tasks:
        row = reusable.get(validation_key_from_task(task))
        if row is None:
            pending.append(task)
        else:
            rows.append(row)
    print(
        f"Validation grid: {len(tasks)} cases, {len(rows)} reused, "
        f"{len(pending)} pending.",
        flush=True,
    )
    if jobs == 1:
        solved = map(solve, pending)
        pool = None
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=jobs)
        solved = pool.imap(solve, pending, chunksize=1)
    try:
        for number, row in enumerate(solved, start=1):
            rows.append(row)
            print(
                f"{number}/{len(pending)} {row['validation scenario']} "
                f"({row['validation resolution']}): "
                f"status={row['status']}, J={row.get('minimum J')}",
                flush=True,
            )
            pd.DataFrame(rows).sort_values(
                ["validation scenario", "validation resolution"]
            ).to_csv(output, index=False, na_rep="NA")
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()
    table = pd.DataFrame(rows).sort_values(
        ["validation scenario", "validation resolution"]
    )
    table.to_csv(output, index=False, na_rep="NA")
    return table


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository
            / "data"
            / "steady_characteristic_two_harmonic_validation.csv"
        ),
    )
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_arguments()
    table = run(args.output, args.jobs)
    print(f"Wrote {args.output} ({len(table)} rows).")


if __name__ == "__main__":
    main()
