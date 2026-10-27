"""Search plausible finite-thickness beds for steady isochrone overturning.

The surface is flat in flow-following coordinates and the bed is periodic.  A
constant downslope component of gravity drives full-Stokes flow.  The bed is
impermeable and obeys the passive linear law u_t = tau_b / beta.  Surface mass
balance is diagnosed from the steady kinematic condition; only accumulating
parts of the surface supply the material-age map.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="Unable to import recommended hash")
warnings.filterwarnings("ignore", message="The .*Function.at.* method")
logging.getLogger("tsfc").setLevel(logging.ERROR)

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


DENSITY = 917.0  # kg m^-3
GRAVITY = 9.81  # m s^-2
VISCOSITY = 2.5e8  # Pa yr
SURFACE_SLOPE = 0.05  # rad
BASAL_DRAG = 1.0e5  # Pa yr m^-1; mobility C = 10^-5 m yr^-1 Pa^-1
NITSCHE_PENALTY = 3.0e6

SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def feature(center, amplitude, width_left, width_right=None, exponent=2.0):
    """One smooth thickness anomaly; positive amplitude is an overdeepening."""
    return {
        "center": center,
        "amplitude": amplitude,
        "width left": width_left,
        "width right": width_left if width_right is None else width_right,
        "exponent": exponent,
    }


# These beds span common valley forms: asymmetric overdeepenings, an
# overdeepening ending at a sill, and paired riegels bounding a basin.  All
# thicknesses are positive, and no boundary moves or prescribes ice velocity.
GEOMETRIES = [
    {
        "name": "asymmetric trough 15m 250x75",
        "length": 800.0,
        "background thickness": 15.0,
        "features": [feature(400.0, 200.0, 250.0, 75.0, 2.0)],
    },
    {
        "name": "asymmetric trough 25m 250x75",
        "length": 800.0,
        "background thickness": 25.0,
        "features": [feature(400.0, 200.0, 250.0, 75.0, 2.0)],
    },
    {
        "name": "asymmetric trough 50m 250x75",
        "length": 800.0,
        "background thickness": 50.0,
        "features": [feature(400.0, 200.0, 250.0, 75.0, 2.0)],
    },
    {
        "name": "reverse trough 25m 75x250",
        "length": 800.0,
        "background thickness": 25.0,
        "features": [feature(400.0, 200.0, 75.0, 250.0, 2.0)],
    },
    {
        "name": "steep asymmetric trough 10m",
        "length": 800.0,
        "background thickness": 10.0,
        "features": [feature(400.0, 250.0, 260.0, 55.0, 4.0)],
    },
    {
        "name": "steep asymmetric trough 25m",
        "length": 800.0,
        "background thickness": 25.0,
        "features": [feature(400.0, 250.0, 260.0, 55.0, 4.0)],
    },
    {
        "name": "steep reverse trough 25m",
        "length": 800.0,
        "background thickness": 25.0,
        "features": [feature(400.0, 250.0, 55.0, 260.0, 4.0)],
    },
    {
        "name": "narrow overdeepening 25m",
        "length": 800.0,
        "background thickness": 25.0,
        "features": [feature(400.0, 250.0, 90.0, 90.0, 2.0)],
    },
    {
        "name": "deep valley 25m",
        "length": 1000.0,
        "background thickness": 25.0,
        "features": [feature(500.0, 400.0, 300.0, 90.0, 2.0)],
    },
    {
        "name": "overdeepening and exit sill",
        "length": 900.0,
        "background thickness": 70.0,
        "features": [
            feature(390.0, 220.0, 230.0, 130.0, 2.0),
            feature(585.0, -55.0, 55.0, 40.0, 4.0),
        ],
    },
    {
        "name": "overdeepening and entry sill",
        "length": 900.0,
        "background thickness": 70.0,
        "features": [
            feature(510.0, 220.0, 130.0, 230.0, 2.0),
            feature(315.0, -55.0, 40.0, 55.0, 4.0),
        ],
    },
    {
        "name": "basin between narrow riegels",
        "length": 900.0,
        "background thickness": 190.0,
        "features": [
            feature(270.0, -165.0, 45.0, 45.0, 4.0),
            feature(630.0, -165.0, 45.0, 45.0, 4.0),
        ],
    },
    {
        "name": "basin between broad riegels",
        "length": 900.0,
        "background thickness": 190.0,
        "features": [
            feature(260.0, -160.0, 90.0, 55.0, 2.0),
            feature(640.0, -160.0, 55.0, 90.0, 2.0),
        ],
    },
    {
        "name": "trough sill doublet",
        "length": 800.0,
        "background thickness": 85.0,
        "features": [
            feature(330.0, 180.0, 180.0, 110.0, 2.0),
            feature(535.0, -70.0, 50.0, 35.0, 4.0),
        ],
    },
    {
        "name": "two asymmetric overdeepenings",
        "length": 1200.0,
        "background thickness": 30.0,
        "features": [
            feature(360.0, 190.0, 210.0, 65.0, 2.0),
            feature(835.0, 190.0, 210.0, 65.0, 2.0),
        ],
    },
]


def periodic_distance(x, center, length):
    return (np.asarray(x) - center + 0.5 * length) % length - 0.5 * length


def thickness(x, geometry):
    """Ice thickness in metres."""
    result = np.full_like(
        np.asarray(x, dtype=float), geometry["background thickness"]
    )
    for item in geometry["features"]:
        distance = periodic_distance(x, item["center"], geometry["length"])
        width = np.where(distance < 0.0, item["width left"], item["width right"])
        shape = np.exp(-(np.abs(distance) / width) ** item["exponent"])
        result += item["amplitude"] * shape
    return result


def represented_thickness(x, geometry, nx):
    """Piecewise-linear thickness represented by a mesh with nx columns."""
    mesh_x = np.linspace(0.0, geometry["length"], nx, endpoint=False)
    mesh_thickness = thickness(mesh_x, geometry)
    return np.interp(
        np.mod(x, geometry["length"]),
        np.r_[mesh_x, geometry["length"]],
        np.r_[mesh_thickness, mesh_thickness[0]],
    )


def make_mesh(geometry, nx, nz):
    """Map a periodic rectangle into a flat-surface, variable-bed glacier."""
    mesh = fd.PeriodicRectangleMesh(nx, nz, geometry["length"], 1.0, direction="x")
    reference = mesh.coordinates.dat.data_ro.copy()
    local_thickness = thickness(reference[:, 0], geometry)
    mesh.coordinates.dat.data[:, 1] = -local_thickness * (1.0 - reference[:, 1])
    mesh.clear_spatial_index()
    return mesh


def solve_stokes(geometry, nx, nz):
    """Solve gravity-driven Stokes flow with passive linear basal sliding."""
    mesh = make_mesh(geometry, nx, nz)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    solution = fd.Function(velocity_space * pressure_space)
    u, p = fd.split(solution)
    v, q = fd.TestFunctions(solution.function_space())
    body_force = fd.Constant(
        (
            DENSITY * GRAVITY * np.sin(SURFACE_SLOPE),
            -DENSITY * GRAVITY * np.cos(SURFACE_SLOPE),
        )
    )
    residual = (
        2.0 * VISCOSITY * fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v)))
        - p * fd.div(v)
        - q * fd.div(u)
        - fd.dot(body_force, v)
    ) * fd.dx

    normal = fd.FacetNormal(mesh)
    tangent = fd.as_vector((-normal[1], normal[0]))
    stress = 2.0 * VISCOSITY * fd.sym(fd.grad(u)) - p * fd.Identity(2)
    test_stress = 2.0 * VISCOSITY * fd.sym(fd.grad(v)) - q * fd.Identity(2)
    normal_speed = fd.dot(u, normal)
    test_normal_speed = fd.dot(v, normal)
    residual += (
        BASAL_DRAG * fd.dot(u, tangent) * fd.dot(v, tangent)
        - fd.dot(stress * normal, normal) * test_normal_speed
        - fd.dot(test_stress * normal, normal) * normal_speed
        + NITSCHE_PENALTY
        * VISCOSITY
        / fd.CellDiameter(mesh)
        * normal_speed
        * test_normal_speed
    ) * fd.ds(1)

    fd.solve(residual == 0, solution, solver_parameters=SOLVER_PARAMETERS)
    velocity, pressure = solution.subfunctions
    gravity_power = fd.assemble(fd.dot(body_force, velocity) * fd.dx)
    basal_dissipation = fd.assemble(
        BASAL_DRAG * fd.dot(velocity, tangent) ** 2 * fd.ds(1)
    )
    viscous_dissipation = fd.assemble(
        2.0
        * VISCOSITY
        * fd.inner(fd.sym(fd.grad(velocity)), fd.sym(fd.grad(velocity)))
        * fd.dx
    )
    bed_length = fd.assemble(fd.Constant(1.0) * fd.ds(1, domain=mesh))
    diagnostics = {
        "cells": mesh.num_cells(),
        "basal dissipation / gravity power": basal_dissipation / gravity_power,
        "relative energy residual": (
            gravity_power - basal_dissipation - viscous_dissipation
        )
        / gravity_power,
        "rms basal normal speed": np.sqrt(
            fd.assemble(fd.dot(velocity, normal) ** 2 * fd.ds(1)) / bed_length
        ),
    }
    return velocity, diagnostics


def make_sampler(
    geometry, nx, velocity, horizontal_samples=481, vertical_samples=141
):
    """Resample the finite-element velocity in terrain-following coordinates."""
    length = geometry["length"]
    dx = length / horizontal_samples
    x = (np.arange(horizontal_samples) + 0.5) * dx
    sigma = np.linspace(0.025, 0.999, vertical_samples)
    local_thickness = represented_thickness(x, geometry, nx)
    X = np.tile(x, (vertical_samples, 1))
    Z = -(1.0 - sigma[:, None]) * local_thickness[None, :]
    points = np.column_stack((X.ravel(), Z.ravel()))
    values = np.asarray(velocity.at(points, tolerance=1.0e-7)).reshape(
        Z.shape + (2,)
    )
    interpolators = {}
    periodic_x = np.r_[x[-1] - length, x, x[0] + length]
    for component, name in enumerate(("u", "w")):
        periodic_values = np.column_stack(
            (values[:, -1, component], values[..., component], values[:, 0, component])
        )
        interpolators[name] = RegularGridInterpolator(
            (sigma, periodic_x), periodic_values, bounds_error=False, fill_value=np.nan
        )

    def sample(name, x_value, z_value):
        x_values, z_values = np.broadcast_arrays(x_value, z_value)
        wrapped = np.mod(x_values, length)
        sigma_values = 1.0 + z_values / represented_thickness(
            wrapped, geometry, nx
        )
        points = np.column_stack((sigma_values.ravel(), wrapped.ravel()))
        result = interpolators[name](points).reshape(x_values.shape)
        return float(result) if result.ndim == 0 else result

    return sample, (
        float(np.nanmin(values[..., 0])),
        float(np.nanmax(values[..., 0])),
    )


def centered_order_jacobian(sources, states, valid):
    """Return centered d x(age; x0) / d x0 within valid source intervals."""
    jacobian = np.full(len(sources), np.nan)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        return jacobian
    source_spacing = np.min(np.diff(sources))
    breaks = (np.diff(valid_indices) > 1) | (
        np.diff(sources[valid_indices]) > 1.5 * source_spacing
    )
    groups = np.split(valid_indices, np.where(breaks)[0] + 1)
    for group in groups:
        if len(group) >= 7:
            interior = group[2:-2]
            jacobian[interior] = (
                states[interior + 1, 0] - states[interior - 1, 0]
            ) / (sources[interior + 1] - sources[interior - 1])
    return jacobian


def trace_surface_map(
    geometry,
    nx,
    sample,
    maximum_age=80_000.0,
    time_step=5.0,
    source_count=481,
):
    """Advect one cohort from accumulating surface and minimize its order."""
    length = geometry["length"]
    sources = np.linspace(0.0, length, source_count, endpoint=False)
    start_sigma = 0.995
    start_z = -(1.0 - start_sigma) * represented_thickness(
        sources, geometry, nx
    )
    accumulation = -sample("w", sources, start_z)
    keep = accumulation > 1.0e-7
    sources = sources[keep]
    accumulation = accumulation[keep]
    state = np.column_stack((sources, start_z[keep]))

    def velocity(values):
        return np.column_stack(
            (
                sample("u", values[:, 0], values[:, 1]),
                sample("w", values[:, 0], values[:, 1]),
            )
        )

    best = {
        "minimum J": np.inf,
        "age of minimum J (yr)": np.nan,
        "source at minimum J (m)": np.nan,
    }
    minimum_path_speed = np.inf
    steps = int(np.ceil(maximum_age / time_step))
    for step_index in range(steps + 1):
        age = step_index * time_step
        sigma = 1.0 + state[:, 1] / represented_thickness(
            state[:, 0], geometry, nx
        )
        valid = np.isfinite(state[:, 0]) & (sigma > 0.03) & (sigma < 0.996)
        jacobian = centered_order_jacobian(sources, state, valid)
        if np.any(np.isfinite(jacobian)):
            index = int(np.nanargmin(jacobian))
            if jacobian[index] < best["minimum J"]:
                best = {
                    "minimum J": float(jacobian[index]),
                    "age of minimum J (yr)": age,
                    "source at minimum J (m)": float(sources[index]),
                }
        if step_index == steps or np.count_nonzero(valid) < 7:
            break
        k1 = velocity(state)
        minimum_path_speed = min(
            minimum_path_speed, float(np.nanmin(k1[valid, 0]))
        )
        k2 = velocity(state + 0.5 * time_step * k1)
        k3 = velocity(state + 0.5 * time_step * k2)
        k4 = velocity(state + time_step * k3)
        state += time_step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    best.update(
        {
            "minimum path u (m/yr)": minimum_path_speed,
            "accumulating source count": int(len(sources)),
            "simulated age (yr)": float(age),
            "valid source count at end": int(np.count_nonzero(valid)),
            "accumulation min (m/yr)": float(np.min(accumulation)),
            "accumulation max (m/yr)": float(np.max(accumulation)),
        }
    )
    return best


def run_case(geometry, nx, nz, maximum_age, time_step, source_count):
    sampled_x = np.linspace(0.0, geometry["length"], 4001, endpoint=False)
    sampled_thickness = thickness(sampled_x, geometry)
    if np.min(sampled_thickness) <= 0.0:
        raise ValueError(f"Nonpositive thickness in {geometry['name']}")
    velocity, diagnostics = solve_stokes(geometry, nx, nz)
    sample, speed_range = make_sampler(geometry, nx, velocity)
    material_map = trace_surface_map(
        geometry,
        nx,
        sample,
        maximum_age=maximum_age,
        time_step=time_step,
        source_count=source_count,
    )
    return {
        "geometry": geometry["name"],
        "mesh": f"{nx} x {nz}",
        "minimum thickness (m)": float(np.min(sampled_thickness)),
        "maximum thickness (m)": float(np.max(sampled_thickness)),
        "minimum grid u (m/yr)": speed_range[0],
        "maximum grid u (m/yr)": speed_range[1],
    } | material_map | diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, action="append")
    parser.add_argument("--nx", type=int, default=60)
    parser.add_argument("--nz", type=int, default=15)
    parser.add_argument("--maximum-age", type=float, default=80_000.0)
    parser.add_argument("--time-step", type=float, default=5.0)
    parser.add_argument("--source-count", type=int, default=481)
    parser.add_argument(
        "--output-prefix",
        help="Optional path prefix for a CSV and compact PNG/PDF summary",
    )
    args = parser.parse_args()
    indices = args.index if args.index is not None else range(len(GEOMETRIES))
    results = []
    for index in indices:
        geometry = GEOMETRIES[index]
        print(f"Solving {index}: {geometry['name']}", flush=True)
        result = run_case(
            geometry,
            args.nx,
            args.nz,
            args.maximum_age,
            args.time_step,
            args.source_count,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    if args.output_prefix:
        table = pd.DataFrame(results)
        output_name = Path(args.output_prefix).name
        repository = Path(__file__).resolve().parents[1]
        csv_path = repository / "data" / f"{output_name}.csv"
        figure_prefix = repository / "figures" / output_name
        table.to_csv(csv_path, index=False, na_rep="NA")
        figure_height = max(3.5, 0.32 * len(table))
        figure, axis = plt.subplots(figsize=(7.2, figure_height), constrained_layout=True)
        plot_table = table.sort_values("minimum J")
        axis.barh(plot_table["geometry"], plot_table["minimum J"], color="0.35")
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set(
            xlabel=r"Minimum surface-map order $\partial x/\partial x_0$",
            ylabel="",
            title="Periodic finite-thickness bed screen",
        )
        figure.savefig(f"{figure_prefix}.png", dpi=220)
        figure.savefig(f"{figure_prefix}.pdf")


if __name__ == "__main__":
    main()
