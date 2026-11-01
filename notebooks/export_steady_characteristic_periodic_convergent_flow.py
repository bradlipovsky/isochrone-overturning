"""Export one representative periodic-convergence velocity field for plotting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import steady_characteristic_periodic_convergent as core


Q = 12.0
RELIEF_RATIO = 2.0
SMOOTHING_FRACTION = 0.10
DRIVING_SLOPE = 0.0
MESH = (360, 48)
SAMPLER = (1801, 241)


def main():
    repository = Path(__file__).resolve().parents[1]
    geometry = core.PeriodicConvergentGeometry(
        Q,
        RELIEF_RATIO,
        SMOOTHING_FRACTION,
        DRIVING_SLOPE,
    )
    _, velocity, stokes = core.solve_stokes(geometry, *MESH)
    _, sample_vector, _, sampler = core.make_variational_sampler(
        geometry,
        MESH[0],
        velocity,
        horizontal_samples=SAMPLER[0],
        vertical_samples=SAMPLER[1],
    )

    surface_phase = np.linspace(0.0, 1.0, 1201, endpoint=False)
    surface_x = surface_phase * geometry.wavelength
    surface_points = np.column_stack((surface_x, np.zeros_like(surface_x)))
    surface_velocity = sample_vector(surface_points)
    surface_slope = core.represented_relief_slope(
        surface_x, geometry, MESH[0]
    )
    surface_physical_w = (
        surface_velocity[:, 1] + surface_velocity[:, 0] * surface_slope
    )
    surface = pd.DataFrame(
        {
            "phase x/L": surface_phase,
            "x/H": surface_x / geometry.mean_thickness,
            "bed elevation/H": core.represented_relief(
                surface_x, geometry, MESH[0]
            )
            / geometry.mean_thickness,
            "surface u (m/yr)": surface_velocity[:, 0],
            "surface transformed W (m/yr)": surface_velocity[:, 1],
            "surface physical w (m/yr)": surface_physical_w,
            "required accumulation a=-W (m/yr)": -surface_velocity[:, 1],
        }
    )

    flow_phase = np.linspace(0.0, 1.0, 121, endpoint=False)
    vertical_fraction = np.linspace(0.10, 0.90, 9)
    phase_grid, fraction_grid = np.meshgrid(
        flow_phase, vertical_fraction, indexing="xy"
    )
    flow_x = phase_grid.ravel() * geometry.wavelength
    zeta = (fraction_grid.ravel() - 1.0) * geometry.mean_thickness
    flow_velocity = sample_vector(np.column_stack((flow_x, zeta)))
    flow_slope = core.represented_relief_slope(flow_x, geometry, MESH[0])
    flow_physical_w = flow_velocity[:, 1] + flow_velocity[:, 0] * flow_slope
    bed = core.represented_relief(flow_x, geometry, MESH[0])
    flow = pd.DataFrame(
        {
            "phase x/L": phase_grid.ravel(),
            "vertical fraction above bed": fraction_grid.ravel(),
            "x/H": flow_x / geometry.mean_thickness,
            "physical elevation/H": (
                bed + fraction_grid.ravel() * geometry.mean_thickness
            )
            / geometry.mean_thickness,
            "u (m/yr)": flow_velocity[:, 0],
            "transformed W (m/yr)": flow_velocity[:, 1],
            "physical w (m/yr)": flow_physical_w,
        }
    )

    data = repository / "data"
    surface_path = data / "steady_characteristic_periodic_convergent_surface.csv"
    flow_path = data / "steady_characteristic_periodic_convergent_flow.csv"
    metadata_path = (
        data / "steady_characteristic_periodic_convergent_flow_metadata.json"
    )
    surface.to_csv(surface_path, index=False)
    flow.to_csv(flow_path, index=False)
    metadata = {
        "description": (
            "Representative velocity field used in the periodic-convergence "
            "summary figure"
        ),
        "diagnostic method version": core.METHOD_VERSION,
        "q = L_right / L_left": Q,
        "bed relief / thickness": RELIEF_RATIO,
        "wrapped-Gaussian sigma / short flank length": SMOOTHING_FRACTION,
        "mean driving slope (rad)": DRIVING_SLOPE,
        "period (m)": geometry.wavelength,
        "thickness (m)": geometry.mean_thickness,
        "mesh": list(MESH),
        "sampler": list(SAMPLER),
        "stokes diagnostics": stokes,
        "sampler diagnostics": sampler,
        "surface rows": int(len(surface)),
        "flow rows": int(len(flow)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {surface_path}")
    print(f"Wrote {flow_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
