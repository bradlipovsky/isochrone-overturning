"""Convergence checks for the realistic steady-geometry experiments."""

from pathlib import Path

import pandas as pd

import realistic_geometry_elliptical_family as elliptical
import realistic_geometry_parameter_search as periodic


REPOSITORY = Path(__file__).resolve().parents[1]


def elliptical_checks():
    """Check mesh, source-count, and time-step convergence of positive cases."""
    cache = {}

    def evaluate(index, resolution, time_step=1.0, source_count=401):
        key = (index, resolution, time_step, source_count)
        if key not in cache:
            print(
                "Elliptical:",
                elliptical.GEOMETRIES[index]["name"],
                f"spacing={resolution:g}, dt={time_step:g}, sources={source_count}",
                flush=True,
            )
            cache[key] = elliptical.run_case(
                elliptical.GEOMETRIES[index],
                resolution,
                maximum_age=3_200.0,
                time_step=time_step,
                source_count=source_count,
            )
        return cache[key]

    rows = []
    for index in (1, 4):
        for resolution in (15.0, 10.0, 7.5):
            row = elliptical.flatten_result(evaluate(index, resolution))
            rows.append(
                {
                    "check": "mesh",
                    "input time step (yr)": 1.0,
                    "input source count": 401,
                }
                | row
            )

    for source_count in (201, 401, 801):
        row = elliptical.flatten_result(
            evaluate(1, 10.0, time_step=1.0, source_count=source_count)
        )
        rows.append(
            {
                "check": "source count",
                "input time step (yr)": 1.0,
                "input source count": source_count,
            }
            | row
        )

    for time_step in (2.0, 1.0, 0.5):
        row = elliptical.flatten_result(
            evaluate(1, 10.0, time_step=time_step, source_count=401)
        )
        rows.append(
            {
                "check": "time step",
                "input time step (yr)": time_step,
                "input source count": 401,
            }
            | row
        )

    table = pd.DataFrame(rows).drop_duplicates(
        subset=[
            "check",
            "geometry",
            "target spacing (m)",
            "input time step (yr)",
            "input source count",
        ]
    )
    path = REPOSITORY / "data" / "realistic_geometry_elliptical_convergence.csv"
    table.to_csv(path, index=False, na_rep="NA")
    print("Saved", path, flush=True)


def elliptical_boundary_checks():
    """Move both hydrostatic cuts while preserving the narrow trough shape."""
    rows = []
    reference = elliptical.GEOMETRIES[1]
    for cut_fraction in (0.70, 0.75, 0.80, 0.85, 0.90):
        geometry = reference | {
            "name": f"narrow symmetric trough, cut={cut_fraction:.2f}",
            "cut fraction": cut_fraction,
        }
        print("Elliptical cut fraction:", cut_fraction, flush=True)
        result = elliptical.run_case(
            geometry,
            10.0,
            maximum_age=3_200.0,
            time_step=1.0,
            source_count=401,
        )
        rows.append(elliptical.flatten_result(result))
    table = pd.DataFrame(rows)
    path = (
        REPOSITORY
        / "data"
        / "realistic_geometry_elliptical_boundary_sensitivity.csv"
    )
    table.to_csv(path, index=False, na_rep="NA")
    print("Saved", path, flush=True)


def periodic_checks():
    """Remesh the three periodic beds closest to loss of order."""
    rows = []
    for index in (4, 6, 11):
        geometry = periodic.GEOMETRIES[index]
        for nx, nz in ((48, 12), (72, 18), (96, 24)):
            print(
                "Periodic:", geometry["name"], f"mesh={nx}x{nz}", flush=True
            )
            row = periodic.run_case(
                geometry,
                nx,
                nz,
                maximum_age=80_000.0,
                time_step=5.0,
                source_count=481,
            )
            rows.append(row)
    table = pd.DataFrame(rows)
    path = REPOSITORY / "data" / "realistic_geometry_periodic_convergence.csv"
    table.to_csv(path, index=False, na_rep="NA")
    print("Saved", path, flush=True)


if __name__ == "__main__":
    elliptical_checks()
    elliptical_boundary_checks()
    periodic_checks()
