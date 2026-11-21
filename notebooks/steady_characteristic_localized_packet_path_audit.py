"""Write the refined per-source audit for the localized-riegel fold case."""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd

import steady_characteristic_localized_packet_parameter_sweep as sweep
import steady_characteristic_sinusoidal_parameter_map as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grading-power", type=float, default=1.0)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    geometry = sweep.LocalizedPacketGeometry(
        wavelength=4.0 * sweep.MEAN_THICKNESS,
        amplitude=0.90 * sweep.MEAN_THICKNESS,
        mean_thickness=sweep.MEAN_THICKNESS,
        feature_width=(1.0 / 12.0) * sweep.MEAN_THICKNESS,
        feature_phase=0.5 * np.pi,
        feature_polarity=1,
    )
    task = sweep.task_for_geometry(
        geometry, "focused path audit", refined=True
    )
    task["mesh_vertical_grading_power"] = args.grading_power
    task["sampler_vertical_grading_power"] = args.grading_power
    _, velocity, stokes = core.solve_stokes(
        geometry,
        task["mesh_nx"],
        task["mesh_nz"],
        vertical_grading_power=task["mesh_vertical_grading_power"],
    )
    sample, sample_vector, sample_gradient, _ = core.make_variational_sampler(
        geometry,
        task["mesh_nx"],
        velocity,
        horizontal_samples=task["sampler_nx"],
        vertical_samples=task["sampler_nz"],
        vertical_grading_power=task["sampler_vertical_grading_power"],
    )
    source_data = core.retained_surface_sources(
        geometry,
        sample,
        source_count=task["source_count"],
        retained_flux_fraction=task["retained_flux_fraction"],
    )
    trajectories = core.trace_sources_to_exit(
        geometry,
        task["mesh_nx"],
        sample_vector,
        sample_gradient,
        source_data["retained sources"],
        time_step=task["time_step"],
        maximum_age=task["maximum_age"],
        surface_accumulation_tolerance=source_data["accumulation tolerance"],
    )
    trajectories.insert(0, "retained source order", np.arange(len(trajectories)))
    trajectories["distance from nearer retained-list end"] = np.minimum(
        trajectories["retained source order"],
        len(trajectories) - 1 - trajectories["retained source order"],
    )
    suffix = "_uniform" if args.grading_power == 1.0 else ""
    path = repository / "data" / f"steady_characteristic_localized_packet_refined{suffix}_paths.csv"
    trajectories.to_csv(path, index=False, na_rep="NA")

    candidates = pd.DataFrame(
        {
            "candidate x (m)": source_data["candidate sources"],
            "candidate phase": source_data["candidate sources"] / geometry.wavelength,
            "surface accumulation (m/yr)": source_data["accumulation"],
            "depositing": np.isin(
                source_data["candidate sources"], source_data["depositing sources"]
            ),
            "retained": np.isin(
                source_data["candidate sources"], source_data["retained sources"]
            ),
        }
    )
    candidate_path = repository / "data" / f"steady_characteristic_localized_packet_refined{suffix}_sources.csv"
    candidates.to_csv(candidate_path, index=False)

    minimum_row = trajectories.loc[
        trajectories["minimum J outside ablating-surface buffer"].idxmin()
    ]
    negative = trajectories[
        "minimum J outside ablating-surface buffer"
    ] < 0.0
    print(f"retained sources: {len(trajectories)}")
    print(f"sources with J < 0: {int(negative.sum())}")
    print(minimum_row.to_string())
    print(f"strong-boundary energy imbalance: {stokes['physical strong-boundary energy imbalance']}")
    print(f"Wrote {path}")
    print(f"Wrote {candidate_path}")


if __name__ == "__main__":
    main()
