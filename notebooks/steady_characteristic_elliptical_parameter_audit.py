"""Numerical audit for the direct steady-characteristic parameter map."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from realistic_geometry_elliptical_family import solve_stokes
from realistic_geometry_elliptical_parameter_map import geometry_from_thicknesses
from steady_characteristic_elliptical_parameter_map import (
    CLASSIFICATION_COLUMN,
    DIRECT_MINIMUM_COLUMN,
    TRACED_TRIM_FRACTION,
    solve_parameter_case,
    trim_minimum_column,
)
from steady_tangent_elliptical_validation import (
    integrate_tangent,
    make_variational_sampler,
)


CASES = (
    ("near-zero 10/70", 10.0, 70.0),
    ("near-zero 30/80", 30.0, 80.0),
    ("clear overturn", 70.0, 150.0),
    ("slow ordered", 70.0, 270.0),
    ("counterflow", 180.0, 240.0),
    ("legacy first-exit censoring", 40.0, 130.0),
    ("former DG edge 40/100", 40.0, 100.0),
    ("former DG edge 50/230", 50.0, 230.0),
    ("former DG edge 60/240", 60.0, 240.0),
    ("no deposition source", 160.0, 270.0),
)

BASELINE = {
    "resolution": 10.0,
    "time_step": 10.0,
    "source_count": 161,
    "sampler_nx": 601,
    "sampler_nz": 181,
    "maximum_age": 120_000.0,
}


def configurations(case):
    """Return baseline plus one-at-a-time numerical perturbations."""
    values = [("baseline", BASELINE.copy())]
    for step in (20.0, 5.0):
        config = BASELINE.copy()
        config["time_step"] = step
        values.append((f"time step {step:g} yr", config))
    if "near-zero" in case or "DG edge" in case:
        config = BASELINE.copy()
        config["time_step"] = 2.5
        values.append(("time step 2.5 yr", config))
    for count in (81, 321):
        config = BASELINE.copy()
        config["source_count"] = count
        values.append((f"candidate sources {count}", config))
    for nx, nz in ((301, 91), (901, 271)):
        config = BASELINE.copy()
        config["sampler_nx"] = nx
        config["sampler_nz"] = nz
        values.append((f"sampler {nx}x{nz}", config))
    for resolution in (7.5, 5.0):
        config = BASELINE.copy()
        config["resolution"] = resolution
        values.append((f"mesh {resolution:g} m", config))
    return values


def audit_worker(payload):
    """Run one configuration in a spawn worker."""
    case, boundary, center, check, config = payload
    task = (
        boundary,
        center,
        config["resolution"],
        config["time_step"],
        config["source_count"],
        TRACED_TRIM_FRACTION,
        config["sampler_nx"],
        config["sampler_nz"],
        config["maximum_age"],
    )
    return {
        "case": case,
        "check": check,
        **solve_parameter_case(task),
    }


def add_adaptive_comparisons(table):
    """Compare each baseline minimum-source path with adaptive integration."""
    table = table.copy()
    for case, boundary, center in CASES:
        selected = (table["case"] == case) & (table["check"] == "baseline")
        if not np.any(selected):
            continue
        index = table.index[selected][0]
        row = table.loc[index]
        if row["status"] != "success":
            continue
        geometry = geometry_from_thicknesses(boundary, center)
        velocity, _ = solve_stokes(geometry, BASELINE["resolution"])
        _, sample_vector, sample_gradient = make_variational_sampler(
            geometry,
            velocity,
            nx=BASELINE["sampler_nx"],
            nz=BASELINE["sampler_nz"],
        )
        trace = integrate_tangent(
            geometry,
            sample_vector,
            sample_gradient,
            float(row["source x0 at minimum J (m)"]),
            maximum_step=2.0,
        )
        adaptive = trace["minimum direct T_x"]
        table.at[index, "adaptive minimum J at selected source"] = adaptive
        table.at[index, "fixed-step minus adaptive minimum J"] = (
            row[DIRECT_MINIMUM_COLUMN] - adaptive
        )
        table.at[index, "adaptive first J=0 age (yr)"] = trace[
            "first T_x=0 age (yr)"
        ]
        table.at[index, "adaptive exit age (yr)"] = trace["exit age (yr)"]
        table.at[index, "adaptive exit kind"] = trace["exit kind"]
        table.at[index, "adaptive minimum path u (m/yr)"] = trace[
            "minimum path u (m/yr)"
        ]
        table.at[index, "adaptive horizontal reversal count"] = trace[
            "horizontal velocity reversal count"
        ]
    return table


def add_case_sensitivity(table):
    """Flag sign changes across numerical checks, separately from trim."""
    table = table.copy()
    table["numerical-check sign change"] = False
    table["numerical-check classification change"] = False
    table["numerical-check minimum J range"] = np.nan
    for case, group in table.groupby("case"):
        values = pd.to_numeric(
            group.loc[group["status"] == "success", DIRECT_MINIMUM_COLUMN],
            errors="coerce",
        ).dropna()
        if len(values):
            sign_change = bool(np.any(values < 0.0) and np.any(values >= 0.0))
            value_range = float(values.max() - values.min())
        else:
            sign_change = False
            value_range = np.nan
        classifications = set(
            group.loc[
                group["status"] == "success", CLASSIFICATION_COLUMN
            ].dropna()
        )
        classification_change = len(classifications) > 1
        selected = table["case"] == case
        table.loc[selected, "numerical-check sign change"] = sign_change
        table.loc[
            selected, "numerical-check classification change"
        ] = classification_change
        table.loc[selected, "numerical-check minimum J range"] = value_range

    interpretations = []
    for _, row in table.iterrows():
        if row["status"] != "success":
            interpretations.append("not evaluated")
        elif bool(row["sign changes across 70/80/90% trims"]):
            interpretations.append("unresolved: trim-sensitive sign")
        elif bool(row["numerical-check classification change"]):
            interpretations.append(
                "unresolved: numerical-check classification change"
            )
        else:
            interpretations.append(row[CLASSIFICATION_COLUMN])
    table["audit interpretation"] = interpretations
    return table


def run_audit(jobs):
    """Run all one-at-a-time checks and adaptive baseline comparisons."""
    payloads = []
    for case, boundary, center in CASES:
        for check, config in configurations(case):
            payloads.append((case, boundary, center, check, config))
    if jobs == 1:
        results = map(audit_worker, payloads)
        pool = None
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=jobs, maxtasksperchild=20)
        results = pool.imap(audit_worker, payloads, chunksize=1)
    rows = []
    try:
        for index, row in enumerate(results, start=1):
            rows.append(row)
            print(
                f"Audit {index}/{len(payloads)}: {row['case']}, "
                f"{row['check']}, {row['status']}",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    table = pd.DataFrame(rows)
    table = add_adaptive_comparisons(table)
    return add_case_sensitivity(table)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "data"
            / "steady_characteristic_elliptical_parameter_map_audit.csv"
        ),
    )
    args = parser.parse_args()
    table = run_audit(args.jobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, na_rep="NA")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
