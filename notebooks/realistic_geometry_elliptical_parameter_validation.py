"""Sensitivity checks for the symmetric-ellipse parameter map."""

from pathlib import Path

import pandas as pd

from realistic_geometry_elliptical_family import flatten_result, run_case
from realistic_geometry_elliptical_parameter_map import geometry_from_thicknesses


CHECKS = [
    # Mesh checks: the existing case and both sides of a nearby zero crossing.
    *(('mesh, existing case', 70.0, 150.0, dx, 1.0, 401, 4_000.0)
      for dx in (15.0, 10.0, 7.5)),
    *(('mesh, positive side', 60.0, 180.0, dx, 1.0, 401, 4_000.0)
      for dx in (15.0, 10.0, 7.5)),
    *(('mesh, negative side', 70.0, 180.0, dx, 1.0, 401, 4_000.0)
      for dx in (15.0, 10.0, 7.5)),
    # Source and time-step checks at the closest sampled fixed-interior zero.
    *(('source count, near zero', 70.0, 180.0, 10.0, 1.0, count, 4_000.0)
      for count in (201, 401, 801)),
    *(('time step, near zero', 70.0, 180.0, 10.0, dt, 401, 4_000.0)
      for dt in (2.0, 1.0, 0.5)),
    # Longer integrations for 4-kyr-ceiling cells on either side of contours.
    ('long horizon', 20.0, 150.0, 10.0, 1.0, 401, 8_000.0),
    ('long horizon', 40.0, 150.0, 10.0, 1.0, 401, 8_000.0),
    ('long horizon', 60.0, 180.0, 10.0, 1.0, 401, 8_000.0),
    ('long horizon', 70.0, 180.0, 10.0, 1.0, 401, 8_000.0),
    ('long horizon', 100.0, 210.0, 10.0, 1.0, 401, 8_000.0),
    ('long horizon', 120.0, 210.0, 10.0, 1.0, 401, 8_000.0),
    ('long horizon', 140.0, 240.0, 10.0, 1.0, 401, 8_000.0),
    ('long horizon', 70.0, 270.0, 10.0, 1.0, 401, 8_000.0),
]


def main():
    rows = []
    for check, boundary, center, resolution, time_step, sources, age in CHECKS:
        print(
            f"{check}: H_boundary={boundary:g}, H_center={center:g}, "
            f"dx={resolution:g}, dt={time_step:g}, N={sources}, age={age:g}",
            flush=True,
        )
        result = flatten_result(
            run_case(
                geometry_from_thicknesses(boundary, center),
                resolution,
                maximum_age=age,
                time_step=time_step,
                source_count=sources,
            )
        )
        rows.append(
            {
                "check": check,
                "boundary thickness input (m)": boundary,
                "center thickness input (m)": center,
                "input target spacing (m)": resolution,
                "input time step (yr)": time_step,
                "input source count": sources,
                "input maximum age (yr)": age,
            }
            | result
        )
        print(
            f"  raw J={result['minimum J']:.6f}; "
            f"fixed-interior J={result['minimum fixed-interior J']:.6f}; "
            f"interior end={result['fixed-interior evaluation end age (yr)']:.0f} yr",
            flush=True,
        )
    repository = Path(__file__).resolve().parents[1]
    pd.DataFrame(rows).to_csv(
        repository
        / "data"
        / "realistic_geometry_elliptical_parameter_validation.csv",
        index=False,
        na_rep="NA",
    )


if __name__ == "__main__":
    main()
