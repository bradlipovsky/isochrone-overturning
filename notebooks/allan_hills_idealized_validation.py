"""Mesh, sampler, and source-depth validation for the open-left case."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import allan_hills_idealized_flow as model


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data/allan_hills_idealized_validation.csv"
)

VARIANTS = (
    ("coarse", 80, 16, 241, 81, 5.0, 193, 0.0005),
    ("production", 160, 32, 501, 161, 5.0, 193, 0.0005),
    ("production-dense-depth", 160, 32, 501, 161, 5.0, 385, 0.0005),
    ("refined", 240, 48, 751, 241, 2.5, 193, 0.0005),
    ("buffer-0.001", 160, 32, 501, 161, 5.0, 193, 0.001),
    ("buffer-0.00025", 160, 32, 501, 201, 5.0, 193, 0.00025),
)


def settings(
    nx,
    nz,
    sample_nx,
    sample_nz,
    time_step,
    source_count,
    sigma_lower,
):
    return SimpleNamespace(
        nx=nx,
        nz=nz,
        sample_nx=sample_nx,
        sample_nz=sample_nz,
        sources=source_count,
        maximum_age=500_000.0,
        time_step=time_step,
        slope_limit=0.10,
        slope_count=41,
        sigma_lower=sigma_lower,
        sigma_upper=model.SAMPLER_SIGMA_UPPER,
        source_sigma_min=model.SOURCE_SIGMA_MIN,
        source_sigma_max=model.SOURCE_SIGMA_MAX,
        store_history=False,
    )


def main():
    model.require_firedrake()
    rows = []
    case = model.CASES[0]
    for (
        name,
        nx,
        nz,
        sample_nx,
        sample_nz,
        time_step,
        source_count,
        sigma_lower,
    ) in VARIANTS:
        result = model.run_case(
            case,
            settings(
                nx,
                nz,
                sample_nx,
                sample_nz,
                time_step,
                source_count,
                sigma_lower,
            ),
        )
        diagnostic = result["diagnostics"]
        horizontal = result["source frame"][
            result["source frame"]["is primary horizontal"]
        ]
        negative = horizontal[horizontal["minimum J"] < 0.0]
        rows.append(
            {
                "variant": name,
                "nx": nx,
                "nz": nz,
                "sample nx": sample_nx,
                "sample nz": sample_nz,
                "source count": source_count,
                "bed buffer sigma": sigma_lower,
                "requested time step (yr)": time_step,
                "actual time step (yr)": diagnostic[
                    "trajectory time step (yr)"
                ],
                "minimum J over all incoming slopes": diagnostic[
                    "minimum J over all incoming slopes"
                ],
                "minimum J, m0=0": diagnostic["minimum J, m0=0"],
                "minimum J, m0=m_right": diagnostic[
                    "minimum J, m0=m_right"
                ],
                "negative horizontal sources": len(negative),
                "minimum negative source sigma": (
                    negative["source sigma"].min()
                    if len(negative)
                    else float("nan")
                ),
                "maximum negative source sigma": (
                    negative["source sigma"].max()
                    if len(negative)
                    else float("nan")
                ),
                "source sigma at minimum J, m0=0": diagnostic[
                    "source sigma at minimum J, m0=0"
                ],
                "age at minimum J, m0=0 (yr)": diagnostic[
                    "age at minimum J, m0=0 (yr)"
                ],
                "maximum abs detF-1": diagnostic["maximum abs detF-1"],
                "maximum abs J budget closure": diagnostic[
                    "maximum abs J budget closure"
                ],
                "point-locator nearest-fill count": diagnostic[
                    "point-locator nearest-fill count"
                ],
                "age-capped paths": diagnostic["age-capped paths"],
                "nonfinite paths": diagnostic["nonfinite paths"],
                "bed-buffer terminations": diagnostic[
                    "bed-buffer terminations"
                ],
                "rms right-boundary w (m/yr)": diagnostic[
                    "rms right-boundary w (m/yr)"
                ],
                "relative flux closure": diagnostic[
                    "relative flux closure"
                ],
            }
        )
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
