"""Test steady overturning in smooth, finite-thickness glacier geometries."""

from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore", message="Unable to import recommended hash")
warnings.filterwarnings("ignore", message="The .*Function.at.* method")
logging.getLogger("tsfc").setLevel(logging.ERROR)

import firedrake as fd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.interpolate import RegularGridInterpolator

plt.rcParams["ps.fonttype"] = 42
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda value: f"{value:.6g}")


DENSITY = 917.0  # kg m^-3
GRAVITY = 9.81  # m s^-2
VISCOSITY = 2.5e8  # Pa yr
SURFACE_SLOPE = 0.05  # rad
BASAL_DRAG = 1.0e5  # Pa yr m^-1
NITSCHE_PENALTY = 3.0e6

GEOMETRIES = [
    {
        "name": "broad overdeepening",
        "kind": "bed trough",
        "length": 500.0,
        "background thickness": 50.0,
        "relief": 150.0,
    },
    {
        "name": "U-shaped overdeepening, 10 m margin",
        "kind": "bed trough",
        "length": 800.0,
        "shape scale": 180.0,
        "shape exponent": 8.0,
        "background thickness": 10.0,
        "relief": 150.0,
    },
    {
        "name": "U-shaped overdeepening, 25 m margin",
        "kind": "bed trough",
        "length": 800.0,
        "shape scale": 180.0,
        "shape exponent": 8.0,
        "background thickness": 25.0,
        "relief": 150.0,
    },
    {
        "name": "surface depression",
        "kind": "surface depression",
        "length": 800.0,
        "background thickness": 170.0,
        "relief": 130.0,
    },
]

CONVERGENCE_MESHES = [(72, 18), (108, 27), (144, 36)]

SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def geometry_shape(x, geometry):
    """One smooth feature per periodic model length."""
    length = geometry["length"]
    distance = (np.asarray(x) - 0.5 * length + 0.5 * length) % length - 0.5 * length
    if "shape exponent" in geometry:
        return np.exp(
            -(np.abs(distance) / geometry["shape scale"]) ** geometry["shape exponent"]
        )
    if "half width" not in geometry:
        return 0.5 * (1.0 + np.cos(2.0 * np.pi * distance / length))
    half_width = geometry["half width"]
    shape = np.zeros_like(distance, dtype=float)
    inside = np.abs(distance) < half_width
    shape[inside] = 0.5 * (1.0 + np.cos(np.pi * distance[inside] / half_width))
    return shape


def geometry_shape_slope(x, geometry):
    """Derivative of the periodic shape."""
    length = geometry["length"]
    distance = (np.asarray(x) - 0.5 * length + 0.5 * length) % length - 0.5 * length
    if "shape exponent" in geometry:
        exponent = geometry["shape exponent"]
        scale = geometry["shape scale"]
        shape = np.exp(-(np.abs(distance) / scale) ** exponent)
        return -shape * exponent * np.abs(distance) ** (exponent - 1.0) * np.sign(distance) / scale**exponent
    if "half width" not in geometry:
        return -np.pi / length * np.sin(2.0 * np.pi * distance / length)
    half_width = geometry["half width"]
    slope = np.zeros_like(distance, dtype=float)
    inside = np.abs(distance) < half_width
    slope[inside] = -0.5 * np.pi / half_width * np.sin(
        np.pi * distance[inside] / half_width
    )
    return slope


def surfaces(x, geometry):
    """Return bed and surface elevations in metres."""
    shape = geometry_shape(x, geometry)
    thickness = geometry["background thickness"]
    relief = geometry["relief"]
    if geometry["kind"] == "bed trough":
        bed = -thickness - relief * shape
        surface = np.zeros_like(shape)
    elif geometry["kind"] == "surface depression":
        bed = np.full_like(shape, -thickness)
        surface = -relief * shape
    else:
        raise ValueError(f"Unknown geometry {geometry['kind']}")
    return bed, surface


def surface_slope(x, geometry):
    if geometry["kind"] == "surface depression":
        return -geometry["relief"] * geometry_shape_slope(x, geometry)
    return np.zeros_like(np.asarray(x), dtype=float)


def make_mesh(geometry, nx, nz):
    """Map a periodic rectangle into the prescribed smooth geometry."""
    length = geometry["length"]
    mesh = fd.PeriodicRectangleMesh(nx, nz, length, 1.0, direction="x")
    reference = mesh.coordinates.dat.data_ro.copy()
    bed, surface = surfaces(reference[:, 0], geometry)
    mesh.coordinates.dat.data[:, 1] = bed + reference[:, 1] * (surface - bed)
    mesh.clear_spatial_index()
    return mesh


def solve_stokes(geometry, nx=72, nz=18):
    """Solve full Stokes flow with no penetration and linear basal drag."""
    mesh = make_mesh(geometry, nx, nz)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)
    u, p = fd.split(solution)
    v, q = fd.TestFunctions(mixed_space)
    strain = fd.sym(fd.grad(u))
    body_force = fd.Constant(
        (
            DENSITY * GRAVITY * np.sin(SURFACE_SLOPE),
            -DENSITY * GRAVITY * np.cos(SURFACE_SLOPE),
        )
    )
    residual = (
        2.0 * VISCOSITY * fd.inner(strain, fd.sym(fd.grad(v)))
        - p * fd.div(v)
        - q * fd.div(u)
        - fd.dot(body_force, v)
    ) * fd.dx

    normal = fd.FacetNormal(mesh)
    tangent = fd.as_vector((-normal[1], normal[0]))
    identity = fd.Identity(2)
    stress = 2.0 * VISCOSITY * fd.sym(fd.grad(u)) - p * identity
    test_stress = 2.0 * VISCOSITY * fd.sym(fd.grad(v)) - q * identity
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
    bed_length = fd.assemble(fd.Constant(1.0) * fd.ds(1, domain=mesh))
    basal_speed = fd.dot(velocity, tangent)
    gravity_power = fd.assemble(fd.dot(body_force, velocity) * fd.dx)
    basal_dissipation = fd.assemble(BASAL_DRAG * basal_speed**2 * fd.ds(1))
    viscous_dissipation = fd.assemble(
        2.0
        * VISCOSITY
        * fd.inner(fd.sym(fd.grad(velocity)), fd.sym(fd.grad(velocity)))
        * fd.dx
    )
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
    return mesh, velocity, diagnostics


def make_sampler(geometry, velocity, nx=361, nz=121):
    """Resample the finite-element velocity in terrain-following coordinates."""
    length = geometry["length"]
    spacing = length / nx
    x = (np.arange(nx) + 0.5) * spacing
    sigma = np.linspace(0.03, 0.995, nz)
    bed, surface = surfaces(x, geometry)
    X = np.tile(x, (nz, 1))
    Z = bed[None, :] + sigma[:, None] * (surface - bed)[None, :]
    points = np.column_stack((X.ravel(), Z.ravel()))
    values = np.asarray(velocity.at(points, tolerance=1.0e-7)).reshape(Z.shape + (2,))
    fields = {}
    for component, name in enumerate(("u", "w")):
        periodic_values = np.column_stack(
            (values[:, -1, component], values[..., component], values[:, 0, component])
        )
        fields[name] = RegularGridInterpolator(
            (sigma, np.r_[x[-1] - length, x, x[0] + length]),
            periodic_values,
            bounds_error=False,
            fill_value=np.nan,
        )

    def sample(name, x_value, z_value):
        x_values, z_values = np.broadcast_arrays(x_value, z_value)
        wrapped = np.mod(x_values, length)
        bed_values, surface_values = surfaces(wrapped, geometry)
        sigma_values = (z_values - bed_values) / (surface_values - bed_values)
        points = np.column_stack((sigma_values.ravel(), wrapped.ravel()))
        result = fields[name](points).reshape(x_values.shape)
        return float(result) if result.ndim == 0 else result

    return sample, (np.nanmin(values[..., 0]), np.nanmax(values[..., 0]))


def trace_material_map(geometry, sample, maximum_age=100_000.0, time_step=5.0, count=401):
    """Trace surface origins and retain the most strongly overturned isochrone."""
    length = geometry["length"]
    sources = np.linspace(0.0, length, count, endpoint=False)
    bed, surface = surfaces(sources, geometry)
    depth = 0.005 * (surface - bed)
    start_z = surface - depth
    surface_u = sample("u", sources, start_z)
    surface_w = sample("w", sources, start_z)
    accumulation = surface_u * surface_slope(sources, geometry) - surface_w
    keep = accumulation > 1.0e-6
    sources = sources[keep]
    state = np.column_stack((sources, start_z[keep]))

    def velocity(values):
        return np.column_stack(
            (sample("u", values[:, 0], values[:, 1]), sample("w", values[:, 0], values[:, 1]))
        )

    best = {
        "minimum J": np.inf,
        "age": np.nan,
        "sources": None,
        "states": None,
        "jacobian": None,
        "valid": None,
    }
    minimum_path_speed = np.inf
    ages = np.arange(0.0, maximum_age + time_step, time_step)
    for step_index, age in enumerate(ages):
        local_bed, local_surface = surfaces(state[:, 0], geometry)
        sigma = (state[:, 1] - local_bed) / (local_surface - local_bed)
        valid = np.isfinite(state[:, 0]) & (sigma > 0.032) & (sigma < 0.996)
        valid_indices = np.where(valid)[0]
        groups = np.split(valid_indices, np.where(np.diff(valid_indices) > 1)[0] + 1)
        jacobian = np.full(len(sources), np.nan)
        for group in groups:
            if len(group) >= 5:
                jacobian[group] = np.gradient(state[group, 0], sources[group])
        if np.any(np.isfinite(jacobian)):
            minimum = np.nanmin(jacobian)
            if minimum < best["minimum J"]:
                best = {
                    "minimum J": minimum,
                    "age": age,
                    "sources": sources.copy(),
                    "states": state.copy(),
                    "jacobian": jacobian.copy(),
                    "valid": valid.copy(),
                }
        if step_index == len(ages) - 1 or not np.any(valid):
            break
        k1 = velocity(state)
        minimum_path_speed = min(minimum_path_speed, np.nanmin(k1[valid, 0]))
        k2 = velocity(state + 0.5 * time_step * k1)
        k3 = velocity(state + 0.5 * time_step * k2)
        k4 = velocity(state + time_step * k3)
        state += time_step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    best["minimum path speed"] = minimum_path_speed
    best["accumulation range"] = (
        np.min(accumulation[keep]),
        np.max(accumulation[keep]),
    )
    return best


def run_case(geometry, nx, nz):
    """Solve one geometry and trace its surface-deposition map."""
    _, velocity, diagnostics = solve_stokes(geometry, nx, nz)
    sample, horizontal_speed_range = make_sampler(geometry, velocity)
    material_map = trace_material_map(geometry, sample)
    minimum_thickness = geometry["background thickness"]
    maximum_thickness = minimum_thickness
    if geometry["kind"] == "bed trough":
        maximum_thickness += geometry["relief"]
    else:
        minimum_thickness -= geometry["relief"]
    row = {
        "geometry": geometry["name"],
        "mesh": f"{nx} x {nz}",
        "minimum thickness (m)": minimum_thickness,
        "maximum thickness (m)": maximum_thickness,
        "minimum grid u (m/yr)": horizontal_speed_range[0],
        "minimum path u (m/yr)": material_map["minimum path speed"],
        "minimum J": material_map["minimum J"],
        "age of minimum J (yr)": material_map["age"],
        "accumulation min (m/yr)": material_map["accumulation range"][0],
        "accumulation max (m/yr)": material_map["accumulation range"][1],
    } | diagnostics
    return row, material_map


results = []
saved_cases = {}
for geometry in GEOMETRIES:
    print(f"Solving {geometry['name']}.", flush=True)
    row, material_map = run_case(geometry, *CONVERGENCE_MESHES[0])
    results.append(row)
    print(row, flush=True)
    saved_cases[geometry["name"]] = (geometry, material_map)

results = pd.DataFrame(results)
display(results)

reference_geometry = next(
    geometry
    for geometry in GEOMETRIES
    if geometry["name"] == "U-shaped overdeepening, 25 m margin"
)
convergence_rows = []
for nx, nz in CONVERGENCE_MESHES:
    if (nx, nz) == CONVERGENCE_MESHES[0]:
        convergence_rows.append(
            results[
                results["geometry"] == "U-shaped overdeepening, 25 m margin"
            ].iloc[0].to_dict()
        )
        continue
    print(f"Checking U-shaped overdeepening on a {nx} x {nz} mesh.", flush=True)
    row, _ = run_case(reference_geometry, nx, nz)
    convergence_rows.append(row)
convergence = pd.DataFrame(convergence_rows)
display(
    convergence[
        [
            "mesh",
            "cells",
            "minimum J",
            "age of minimum J (yr)",
            "minimum path u (m/yr)",
            "rms basal normal speed",
            "relative energy residual",
        ]
    ]
)

figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
selected = [
    "broad overdeepening",
    "U-shaped overdeepening, 10 m margin",
    "U-shaped overdeepening, 25 m margin",
    "surface depression",
]
for axis, case_name in zip(axes.ravel(), selected):
    geometry, material_map = saved_cases[case_name]
    length = geometry["length"]
    x = np.linspace(0.0, length, 601)
    bed, surface = surfaces(x, geometry)
    axis.fill_between(x, bed, np.min(bed) - 10.0, color="0.82")
    axis.plot(x, bed, color="0.25", linewidth=0.8)
    axis.plot(x, surface, color="0.25", linewidth=0.8)
    states = material_map["states"]
    valid = material_map["valid"] & np.isfinite(material_map["jacobian"])
    plot_points = states[valid].copy()
    plot_points[:, 0] = np.mod(plot_points[:, 0], length)
    order = np.argsort(plot_points[:, 0])
    plot_points = plot_points[order]
    plot_sources = material_map["sources"][valid][order]
    neighboring = np.diff(plot_points[:, 0]) < 0.05 * length
    segments = np.stack((plot_points[:-1], plot_points[1:]), axis=1)[neighboring]
    collection = LineCollection(segments, cmap="viridis", linewidth=1.8)
    collection.set_array(
        (0.5 * (plot_sources[:-1] + plot_sources[1:]))[neighboring]
    )
    axis.add_collection(collection)
    axis.set(
        xlim=(0.0, length),
        xlabel="Horizontal coordinate (m)",
        ylabel="Elevation (m)",
        title=(
            f"{case_name}\n"
            f"$J_{{min}}={material_map['minimum J']:.2f}$ at "
            f"{material_map['age'] / 1000:.2f} kyr"
        ),
    )

figure_directory = Path("../figures")
for suffix in ("eps", "pdf"):
    figure.savefig(figure_directory / f"dissipative_sliding_geometry_survey.{suffix}")
figure.savefig(figure_directory / "dissipative_sliding_geometry_survey.png", dpi=220)
for suffix in ("eps", "pdf"):
    figure.savefig(figure_directory / f"Figure8.{suffix}")
figure.savefig(figure_directory / "Figure8.png", dpi=220)
plt.show()
