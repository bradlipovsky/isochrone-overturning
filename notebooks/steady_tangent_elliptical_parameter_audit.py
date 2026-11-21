"""Mesh and characteristic audit for the stationary symmetric parameter map.

Six geometries sample ordered, overturned, counterflow, transition, and
no-deposition regimes.  The Eulerian stationary tangent metric is recomputed
at 10, 7.5, and 5 m.  For the counterflow and transition cases, an adaptive
steady characteristic is launched from the surface source attached to the
Eulerian minimum-J cell and integrated to its first physical exit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from realistic_geometry_elliptical_family import side_locations, solve_stokes
from realistic_geometry_elliptical_parameter_map import geometry_from_thicknesses
from steady_tangent_elliptical_parameter_map import (
    MAIN_PROVENANCE_THRESHOLD,
    PROVENANCE_THRESHOLDS,
    RAW_FIRST_PERCENTILE_COLUMN,
    RAW_MINIMUM_COLUMN,
    central_surface_source_interval,
    minimum_tangent_j,
    solve_stationary_parameter_fields,
)
from steady_tangent_elliptical_validation import (
    integrate_tangent,
    make_variational_sampler,
)


CASES = (
    ("ordered", 70.0, 270.0, False),
    ("overturned", 70.0, 150.0, False),
    ("counterflow", 180.0, 240.0, True),
    ("censoring comparison", 30.0, 130.0, True),
    ("censoring mechanism", 40.0, 130.0, True),
    ("mask edge 40/100", 40.0, 100.0, True),
    ("mask edge 50/230", 50.0, 230.0, True),
    ("mask edge 60/240", 60.0, 240.0, True),
    ("no deposition source", 160.0, 270.0, False),
)
RESOLUTIONS = (10.0, 7.5, 5.0)
PRODUCTION_BUFFER_M = 20.0
PHYSICAL_BUFFERS_M = (10.0, 20.0, 30.0)
AUDIT_MINIMUM_COLUMN = (
    "minimum Eulerian stationary J (20 m outflow buffer)"
)
AUDIT_PERCENTILE_COLUMN = (
    "first percentile Eulerian stationary J (20 m outflow buffer)"
)


def production_minimum_location(
    fields, source_interval, geometry, resolution
):
    """Return the production-mask minimum and its transported source label."""
    coordinates = np.asarray(fields["coordinates"].dat.data_ro)
    velocity = np.asarray(fields["velocity"].dat.data_ro)
    provenance = np.asarray(fields["provenance"].dat.data_ro).reshape(-1)
    source_x = np.asarray(fields["source x"].dat.data_ro).reshape(-1)
    tangent_j = np.asarray(fields["tangent J"].dat.data_ro).reshape(-1)
    source_min, source_max = source_interval
    left_side, right_side = side_locations(geometry)
    distance = PRODUCTION_BUFFER_M
    outflow_buffer = (
        (
            (coordinates[:, 0] < left_side + distance)
            & (velocity[:, 0] < 0.0)
        )
        | (
            (coordinates[:, 0] > right_side - distance)
            & (velocity[:, 0] > 0.0)
        )
        | (
            (coordinates[:, 1] > -distance)
            & (velocity[:, 1] > 0.0)
        )
    )
    valid = (
        np.isfinite(provenance)
        & np.isfinite(source_x)
        & np.isfinite(tangent_j)
        & (provenance >= MAIN_PROVENANCE_THRESHOLD)
        & (source_x >= source_min)
        & (source_x <= source_max)
        & (coordinates[:, 1] < -0.25)
        & ~outflow_buffer
    )
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        raise RuntimeError("production stationary mask is empty")
    index = int(candidates[np.argmin(tangent_j[candidates])])
    return {
        "Eulerian minimum J": float(tangent_j[index]),
        "Eulerian minimum source x0 (m)": float(source_x[index]),
        "Eulerian minimum x (m)": float(coordinates[index, 0]),
        "Eulerian minimum z (m)": float(coordinates[index, 1]),
        "Eulerian minimum provenance": float(provenance[index]),
    }


def classify(value):
    """Apply the stated numerical resolution band around J=0."""
    if not np.isfinite(value):
        return "not evaluated"
    if value < -0.05:
        return "resolved overturn"
    if value > 0.05:
        return "resolved ordered"
    return "unresolved: |minimum J|<=0.05"


def run_audit():
    """Return one row for each geometry and mesh resolution."""
    rows = []
    for case, boundary, center, run_characteristic in CASES:
        geometry = geometry_from_thicknesses(boundary, center)
        for resolution in RESOLUTIONS:
            print(
                f"Auditing {case}: Hb={boundary:g}, Hc={center:g}, "
                f"h={resolution:g} m",
                flush=True,
            )
            started = perf_counter()
            velocity, stokes = solve_stokes(geometry, resolution)
            sample, sample_vector, sample_gradient = make_variational_sampler(
                geometry, velocity
            )
            source_interval, depositing_count = central_surface_source_interval(
                geometry, sample
            )
            row = {
                "case": case,
                "boundary thickness (m)": boundary,
                "center thickness (m)": center,
                "resolution (m)": resolution,
                "mesh cells": stokes["cells"],
                "depositing surface source count": depositing_count,
            }
            if source_interval is None:
                row |= {
                    "status": "no surface-deposition source",
                    AUDIT_MINIMUM_COLUMN: np.nan,
                    "Eulerian classification": "not evaluated",
                    "sign changes across available masks": False,
                    "raw/buffered sign disagreement": False,
                    "total audit runtime (s)": perf_counter() - started,
                }
                rows.append(row)
                continue

            fields = solve_stationary_parameter_fields(velocity)
            production = minimum_tangent_j(
                fields,
                source_interval,
                geometry,
                resolution,
                MAIN_PROVENANCE_THRESHOLD,
                PRODUCTION_BUFFER_M / resolution,
            )
            raw = minimum_tangent_j(
                fields,
                source_interval,
                geometry,
                resolution,
                MAIN_PROVENANCE_THRESHOLD,
                0.0,
            )
            location = production_minimum_location(
                fields, source_interval, geometry, resolution
            )
            row |= {
                "status": "success",
                AUDIT_MINIMUM_COLUMN: production["minimum"],
                AUDIT_PERCENTILE_COLUMN: production["first percentile"],
                RAW_MINIMUM_COLUMN: raw["minimum"],
                RAW_FIRST_PERCENTILE_COLUMN: raw["first percentile"],
                "Eulerian classification": classify(production["minimum"]),
                "fixed source minimum x0 (m)": source_interval[0],
                "fixed source maximum x0 (m)": source_interval[1],
            } | location

            sensitivity_values = []
            populated_masks = 0
            for threshold in PROVENANCE_THRESHOLDS:
                for buffer_m in PHYSICAL_BUFFERS_M:
                    column = (
                        "minimum Eulerian stationary J "
                        f"(provenance>={threshold:.2f}, buffer={buffer_m:g}m)"
                    )
                    try:
                        summary = minimum_tangent_j(
                            fields,
                            source_interval,
                            geometry,
                            resolution,
                            threshold,
                            buffer_m / resolution,
                        )
                        value = summary["minimum"]
                        populated_masks += 1
                        sensitivity_values.append(value)
                    except RuntimeError:
                        value = np.nan
                    row[column] = value
            sensitivity_values = np.asarray(sensitivity_values)
            row["sensitivity masks with retained cells"] = populated_masks
            mask_sign_changes = bool(
                np.any(sensitivity_values < 0.0)
                and np.any(sensitivity_values >= 0.0)
            )
            row["sign changes across available masks"] = mask_sign_changes
            row["raw/buffered sign disagreement"] = bool(
                (raw["minimum"] < 0.0) != (production["minimum"] < 0.0)
            )

            if run_characteristic:
                trace = integrate_tangent(
                    geometry,
                    sample_vector,
                    sample_gradient,
                    location["Eulerian minimum source x0 (m)"],
                    maximum_step=2.0,
                )
                row |= {
                    "adaptive characteristic minimum J": trace[
                        "minimum direct T_x"
                    ],
                    "adaptive characteristic classification": classify(
                        trace["minimum direct T_x"]
                    ),
                    "Eulerian minus characteristic minimum J": (
                        production["minimum"]
                        - trace["minimum direct T_x"]
                    ),
                    "adaptive characteristic exit age (yr)": trace[
                        "exit age (yr)"
                    ],
                    "adaptive characteristic exit kind": trace["exit kind"],
                    "adaptive characteristic minimum path u (m/yr)": trace[
                        "minimum path u (m/yr)"
                    ],
                    "adaptive characteristic horizontal reversal count": trace[
                        "horizontal velocity reversal count"
                    ],
                    "Eulerian/characteristic sign disagreement": bool(
                        (production["minimum"] < 0.0)
                        != (trace["minimum direct T_x"] < 0.0)
                    ),
                }
            row["total audit runtime (s)"] = perf_counter() - started
            rows.append(row)
    table = pd.DataFrame(rows)
    table["mesh sign changes across resolutions"] = False
    for case, group in table.groupby("case"):
        values = pd.to_numeric(
            group[AUDIT_MINIMUM_COLUMN], errors="coerce"
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        mesh_sign_changes = bool(
            np.any(values < 0.0) and np.any(values >= 0.0)
        )
        table.loc[
            table["case"] == case,
            "mesh sign changes across resolutions",
        ] = mesh_sign_changes

    interpretations = []
    for _, row in table.iterrows():
        if row["status"] != "success":
            interpretations.append("not evaluated")
        elif bool(row["sign changes across available masks"]):
            interpretations.append("unresolved: mask-sensitive sign")
        elif bool(row["mesh sign changes across resolutions"]):
            interpretations.append("unresolved: mesh-sensitive sign")
        else:
            interpretations.append(classify(row[AUDIT_MINIMUM_COLUMN]))
    table["audit interpretation"] = interpretations
    return table


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "data"
            / "steady_tangent_elliptical_parameter_map_audit.csv"
        ),
    )
    args = parser.parse_args()
    table = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, na_rep="NA")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
