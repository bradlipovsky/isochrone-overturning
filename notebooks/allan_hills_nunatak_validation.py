"""Diagnostic convergence tests for the true Allan Hills point margin."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import allan_hills_idealized_flow as core
import allan_hills_nunatak_margin as margin


ROOT = Path(__file__).resolve().parents[1]

VARIANTS = (
    {
        "variant": "production",
        "sample_nx": 501,
        "sample_nz": 161,
        "horizontal_margin": 0.5,
        "sigma_lower": 0.0005,
        "sigma_upper": 0.99,
        "sources": 193,
        "time_step": 20.0,
    },
    {
        "variant": "small-time-step",
        "sample_nx": 501,
        "sample_nz": 161,
        "horizontal_margin": 0.5,
        "sigma_lower": 0.0005,
        "sigma_upper": 0.99,
        "sources": 193,
        "time_step": 1.5,
    },
    {
        "variant": "dense-sampler-and-sources",
        "sample_nx": 751,
        "sample_nz": 241,
        "horizontal_margin": 0.5,
        "sigma_lower": 0.0005,
        "sigma_upper": 0.99,
        "sources": 385,
        "time_step": 20.0,
    },
    {
        "variant": "source-plane-20m-inside",
        "sample_nx": 501,
        "sample_nz": 161,
        "horizontal_margin": 10.0,
        "sigma_lower": 0.0005,
        "sigma_upper": 0.99,
        "sources": 193,
        "time_step": 20.0,
    },
    {
        "variant": "larger-terrain-buffer",
        "sample_nx": 501,
        "sample_nz": 161,
        "horizontal_margin": 0.5,
        "sigma_lower": 0.001,
        "sigma_upper": 0.985,
        "sources": 193,
        "time_step": 20.0,
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rheology",
        choices=("linear", "glen"),
        default="linear",
        help="Constitutive law: linear viscous (n=1) or isotropic Glen (n=3).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = "" if args.rheology == "linear" else f"_{args.rheology}"
    output = ROOT / f"data/allan_hills_nunatak{suffix}_validation.csv"
    margin.require_environment()
    configuration = margin.MarginConfiguration(
        "point_margin_validation",
        "True nunatak point margin validation",
        0.0,
        20.0,
        4.0,
        25.0,
        rheology=args.rheology,
    )
    _, velocity, solve_diagnostics = margin.solve_stokes(configuration)
    sampling_left = margin.x_at_thickness(
        configuration.diagnostic_thickness_cutoff
    )
    case = core.Case(
        configuration.key,
        configuration.title,
        True,
        "nunatak_margin",
        configuration.inflow_fraction,
    )
    slopes = np.linspace(-0.1, 0.1, 41)
    rows = []
    for variant in VARIANTS:
        print(f"Tracing {variant['variant']}", flush=True)
        sampler = core.make_sampler(
            core.CASES[0],
            velocity,
            nx_sample=variant["sample_nx"],
            nz_sample=variant["sample_nz"],
            horizontal_margin=variant["horizontal_margin"],
            sigma_lower=variant["sigma_lower"],
            sigma_upper=variant["sigma_upper"],
            domain_left=sampling_left,
            domain_right=margin.LENGTH,
            analytic_bed_function=margin.bed_elevation,
            analytic_surface_function=margin.surface_elevation,
        )
        trace = core.trace_incoming_material(
            case,
            sampler,
            slopes,
            variant["sources"],
            3_000_000.0,
            variant["time_step"],
            store_history=False,
            source_sigma_min=0.01,
            source_sigma_max=0.95,
        )
        frame = pd.DataFrame(trace["rows"])
        horizontal = frame[frame["is primary horizontal"]]
        negative = horizontal[horizontal["minimum J"] < 0.0]
        statuses = pd.Series(trace["exit status"]).value_counts()
        rows.append(
            {
                "variant": variant["variant"],
                "sample nx": variant["sample_nx"],
                "sample nz": variant["sample_nz"],
                "source count": variant["sources"],
                "source plane x (m)": (
                    margin.LENGTH - 2.0 * variant["horizontal_margin"]
                ),
                "sigma lower": variant["sigma_lower"],
                "sigma upper": variant["sigma_upper"],
                "requested time step (yr)": variant["time_step"],
                "actual time step (yr)": trace["time step"],
                "minimum J, m0=0": horizontal["minimum J"].min(),
                "minimum J for source sigma >= 0.05": horizontal.loc[
                    horizontal["source sigma"] >= 0.05, "minimum J"
                ].min(),
                "negative horizontal sources": len(negative),
                "maximum negative source sigma": (
                    negative["source sigma"].max()
                    if len(negative)
                    else np.nan
                ),
                "maximum abs detF-1": np.max(
                    trace["maximum determinant error"]
                ),
                "maximum abs J budget closure": frame[
                    "J budget closure at minimum"
                ].abs().max(),
                "left/cutoff exits": int(statuses.get("left_exit", 0)),
                "surface exits": int(statuses.get("surface_exit", 0)),
                "bed-buffer terminations": int(
                    statuses.get("bed_buffer", 0)
                ),
                "nonfinite paths": int(statuses.get("nonfinite", 0)),
                "age-capped paths": int(statuses.get("age_cap", 0)),
                "rms right-boundary w (m/yr)": solve_diagnostics[
                    "rms right-boundary w (m/yr)"
                ],
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
