"""Test a finite, positive-thickness family of elliptical glacier troughs.

The calculation uses gravity-driven full-Stokes flow, a traction-free surface,
hydrostatic traction on the two open lateral cuts, and the passive basal law
u_t = tau_b / beta.  It changes trough aspect ratio and asymmetry without
prescribing velocity anywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import tempfile
import warnings

warnings.filterwarnings("ignore", message="Unable to import recommended hash")
warnings.filterwarnings("ignore", message="The .*Function.at.* method")
logging.getLogger("tsfc").setLevel(logging.ERROR)

import firedrake as fd
import gmsh
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


DENSITY = 917.0  # kg m^-3
GRAVITY = 9.81  # m s^-2
VISCOSITY = 2.5e8  # Pa yr
SURFACE_SLOPE = 0.05  # rad
BASAL_DRAG = 1.0e5  # Pa yr m^-1; mobility C = 10^-5 m yr^-1 Pa^-1
NITSCHE_PENALTY = 3.0e4

SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


GEOMETRIES = [
    {
        "name": "circular reference",
        "vertical radius": 250.0,
        "center height": 100.0,
        "left radius": 250.0,
        "right radius": 250.0,
        "cut fraction": 0.80,
    },
    {
        "name": "narrow symmetric trough",
        "vertical radius": 250.0,
        "center height": 100.0,
        "left radius": 175.0,
        "right radius": 175.0,
        "cut fraction": 0.80,
    },
    {
        "name": "broad symmetric trough",
        "vertical radius": 250.0,
        "center height": 100.0,
        "left radius": 350.0,
        "right radius": 350.0,
        "cut fraction": 0.80,
    },
    {
        "name": "very broad symmetric trough",
        "vertical radius": 250.0,
        "center height": 100.0,
        "left radius": 500.0,
        "right radius": 500.0,
        "cut fraction": 0.80,
    },
    {
        "name": "steep entry gentle exit",
        "vertical radius": 250.0,
        "center height": 100.0,
        "left radius": 175.0,
        "right radius": 350.0,
        "cut fraction": 0.80,
    },
    {
        "name": "gentle entry steep exit",
        "vertical radius": 250.0,
        "center height": 100.0,
        "left radius": 350.0,
        "right radius": 175.0,
        "cut fraction": 0.80,
    },
    {
        "name": "deeper asymmetric trough",
        "vertical radius": 350.0,
        "center height": 150.0,
        "left radius": 245.0,
        "right radius": 490.0,
        "cut fraction": 0.80,
    },
    {
        "name": "shallower asymmetric trough",
        "vertical radius": 180.0,
        "center height": 80.0,
        "left radius": 126.0,
        "right radius": 252.0,
        "cut fraction": 0.80,
    },
]


def half_width(geometry, side):
    horizontal_radius = geometry[f"{side} radius"]
    vertical_radius = geometry["vertical radius"]
    center_height = geometry["center height"]
    return horizontal_radius * np.sqrt(
        1.0 - (center_height / vertical_radius) ** 2
    )


def side_locations(geometry):
    if "half domain" in geometry:
        return -geometry["half domain"], geometry["half domain"]
    cut = geometry["cut fraction"]
    return -cut * half_width(geometry, "left"), cut * half_width(
        geometry, "right"
    )


def bed_height(x, geometry):
    """Elevation of two elliptical half-arcs joined smoothly at the bottom."""
    x_values = np.asarray(x)
    horizontal_radius = np.where(
        x_values < 0.0, geometry["left radius"], geometry["right radius"]
    )
    inside = np.maximum(0.0, 1.0 - (x_values / horizontal_radius) ** 2)
    return geometry["center height"] - geometry["vertical radius"] * np.sqrt(
        inside
    )


def make_mesh(geometry, resolution):
    """Mesh the two elliptical bed arcs and retain exact quadratic geometry."""
    left_side, right_side = side_locations(geometry)
    left_bed_height = float(bed_height(left_side, geometry))
    right_bed_height = float(bed_height(right_side, geometry))
    center_height = geometry["center height"]
    vertical_radius = geometry["vertical radius"]
    bottom_height = center_height - vertical_radius

    mesh_file = tempfile.NamedTemporaryFile(suffix=".msh", delete=False).name
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.add("asymmetric_elliptical_trough")
    left_surface = gmsh.model.geo.addPoint(left_side, 0.0, 0.0, resolution)
    left_bed = gmsh.model.geo.addPoint(
        left_side, left_bed_height, 0.0, resolution
    )
    bottom = gmsh.model.geo.addPoint(0.0, bottom_height, 0.0, resolution)
    right_bed = gmsh.model.geo.addPoint(
        right_side, right_bed_height, 0.0, resolution
    )
    right_surface = gmsh.model.geo.addPoint(right_side, 0.0, 0.0, resolution)
    center = gmsh.model.geo.addPoint(0.0, center_height, 0.0, resolution)
    left_axis = gmsh.model.geo.addPoint(
        geometry["left radius"], center_height, 0.0, resolution
    )
    right_axis = gmsh.model.geo.addPoint(
        geometry["right radius"], center_height, 0.0, resolution
    )
    bed_left = gmsh.model.geo.addEllipseArc(
        left_bed, center, left_axis, bottom
    )
    bed_right = gmsh.model.geo.addEllipseArc(
        bottom, center, right_axis, right_bed
    )
    right = gmsh.model.geo.addLine(right_bed, right_surface)
    surface = gmsh.model.geo.addLine(right_surface, left_surface)
    left = gmsh.model.geo.addLine(left_surface, left_bed)
    loop = gmsh.model.geo.addCurveLoop(
        [bed_left, bed_right, right, surface, left]
    )
    area = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [bed_left, bed_right], 1)
    gmsh.model.addPhysicalGroup(1, [surface], 2)
    gmsh.model.addPhysicalGroup(1, [left], 3)
    gmsh.model.addPhysicalGroup(1, [right], 4)
    gmsh.model.addPhysicalGroup(2, [area], 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_file)
    gmsh.finalize()

    linear_mesh = fd.Mesh(mesh_file)
    coordinate_space = fd.VectorFunctionSpace(linear_mesh, "CG", 2)
    coordinates = fd.Function(coordinate_space).interpolate(
        fd.SpatialCoordinate(linear_mesh)
    )
    bed_nodes = fd.DirichletBC(
        coordinate_space, fd.Constant((0.0, 0.0)), 1
    ).nodes
    bed_coordinates = coordinates.dat.data[bed_nodes]
    is_left = bed_coordinates[:, 0] < 0.0
    horizontal_radii = np.where(
        is_left, geometry["left radius"], geometry["right radius"]
    )
    angles = np.arctan2(
        (bed_coordinates[:, 1] - center_height) / vertical_radius,
        bed_coordinates[:, 0] / horizontal_radii,
    )
    coordinates.dat.data[bed_nodes] = np.column_stack(
        (
            horizontal_radii * np.cos(angles),
            center_height + vertical_radius * np.sin(angles),
        )
    )
    return fd.Mesh(coordinates)


def solve_stokes(geometry, resolution):
    """Solve with linear basal drag and hydrostatic open-side traction."""
    mesh = make_mesh(geometry, resolution)
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
        / fd.Constant(resolution)
        * normal_speed
        * test_normal_speed
    ) * fd.ds(1)

    z = fd.SpatialCoordinate(mesh)[1]
    hydrostatic_pressure = -DENSITY * GRAVITY * np.cos(SURFACE_SLOPE) * z
    side_traction = -hydrostatic_pressure * normal
    side_measure = fd.ds(3, domain=mesh) + fd.ds(4, domain=mesh)
    residual -= fd.dot(side_traction, v) * side_measure
    fd.solve(residual == 0, solution, solver_parameters=SOLVER_PARAMETERS)
    velocity, pressure = solution.subfunctions

    gravity_power = fd.assemble(fd.dot(body_force, velocity) * fd.dx)
    left_boundary_power = fd.assemble(
        fd.dot(side_traction, velocity) * fd.ds(3, domain=mesh)
    )
    right_boundary_power = fd.assemble(
        fd.dot(side_traction, velocity) * fd.ds(4, domain=mesh)
    )
    boundary_power = left_boundary_power + right_boundary_power
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
        "gravity power": gravity_power,
        "left-boundary power": left_boundary_power,
        "right-boundary power": right_boundary_power,
        "side-boundary power": boundary_power,
        "basal dissipation / gravity power": basal_dissipation / gravity_power,
        "relative energy residual": (
            gravity_power + boundary_power - basal_dissipation - viscous_dissipation
        )
        / (abs(gravity_power) + abs(boundary_power)),
        "rms basal normal speed": np.sqrt(
            fd.assemble(fd.dot(velocity, normal) ** 2 * fd.ds(1)) / bed_length
        ),
    }
    return velocity, diagnostics


def make_sampler(geometry, velocity):
    """Sample the finite-element solution in bed-following coordinates."""
    left_side, right_side = side_locations(geometry)
    x = np.linspace(0.995 * left_side, 0.995 * right_side, 601)
    # Deposits start 0.05 m below the surface.  The sampling grid must extend
    # above that level even in the thickest geometry, or near-surface queries
    # become NaN and silently truncate the deposition interval.
    sigma = np.linspace(0.005, 0.99995, 181)
    bed = bed_height(x, geometry)
    X = np.tile(x, (len(sigma), 1))
    Z = bed[None, :] * (1.0 - sigma[:, None])
    points = np.column_stack((X.ravel(), Z.ravel()))
    values = np.asarray(velocity.at(points, tolerance=1.0e-7)).reshape(
        Z.shape + (2,)
    )
    fields = {
        name: RegularGridInterpolator(
            (sigma, x),
            values[..., component],
            bounds_error=False,
            fill_value=np.nan,
        )
        for component, name in enumerate(("u", "w"))
    }

    def sample(name, x_value, z_value):
        x_values, z_values = np.broadcast_arrays(x_value, z_value)
        local_bed = bed_height(x_values, geometry)
        local_sigma = (z_values - local_bed) / (-local_bed)
        points = np.column_stack((local_sigma.ravel(), x_values.ravel()))
        result = fields[name](points).reshape(x_values.shape)
        return float(result) if result.ndim == 0 else result

    return sample, (
        float(np.nanmin(values[..., 0])),
        float(np.nanmax(values[..., 0])),
    )


def centered_jacobian(sources, states, valid):
    """Centered source-order Jacobian, excluding ends of each valid group."""
    jacobian = np.full(len(sources), np.nan)
    indices = np.flatnonzero(valid)
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    for group in groups:
        if len(group) >= 9:
            interior = group[2:-2]
            jacobian[interior] = (
                states[interior + 1, 0] - states[interior - 1, 0]
            ) / (sources[interior + 1] - sources[interior - 1])
    return jacobian


def trace_surface_map(
    geometry,
    sample,
    maximum_age=4_000.0,
    time_step=1.0,
    source_count=401,
    minimum_valid_sources=21,
):
    """Advect one accumulating cohort and test loss of horizontal order.

    In addition to the lifetime minimum over all currently valid markers, the
    function reports a fixed-interior minimum.  That diagnostic uses the
    central 80% of the original deposition origins and stops as soon as any
    marker in that fixed set exits.  It therefore cannot follow a moving exit
    edge into the source family.
    """
    left_side, right_side = side_locations(geometry)
    sources = np.linspace(0.96 * left_side, 0.04 * left_side, source_count)
    state = np.column_stack((sources, np.full_like(sources, -0.05)))
    accumulation = -sample("w", sources, state[:, 1])
    keep = accumulation > 1.0e-6
    sources = sources[keep]
    accumulation = accumulation[keep]
    state = state[keep]

    if len(sources) == 0:
        return {
            "minimum J": None,
            "age of minimum J (yr)": None,
            "valid source count at minimum": 0,
            "valid source span at minimum (m)": None,
            "first negative": None,
            "first resolved overturn": None,
            "minimum path u (m/yr)": None,
            "deposition source count": 0,
            "deposition source span (m)": None,
            "accumulation min (m/yr)": None,
            "accumulation max (m/yr)": None,
            "minimum fixed-interior J": None,
            "age of minimum fixed-interior J (yr)": None,
            "fixed-interior source count": 0,
            "fixed-interior source span (m)": None,
            "fixed-interior evaluation end age (yr)": None,
        }

    margin = int(np.ceil(0.10 * len(sources)))
    fixed_interior = np.zeros(len(sources), dtype=bool)
    fixed_interior[margin : len(sources) - margin] = True

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
        "valid source count at minimum": 0,
        "valid source span at minimum (m)": np.nan,
    }
    first_negative = None
    first_resolved = None
    minimum_path_u = np.inf
    interior_best = {
        "minimum fixed-interior J": np.inf,
        "age of minimum fixed-interior J (yr)": np.nan,
    }
    interior_open = True
    interior_end_age = 0.0
    for step_index in range(int(np.ceil(maximum_age / time_step)) + 1):
        age = step_index * time_step
        local_bed = bed_height(state[:, 0], geometry)
        sigma = (state[:, 1] - local_bed) / (-local_bed)
        valid = (
            np.isfinite(state[:, 0])
            & (state[:, 0] > 0.994 * left_side)
            & (state[:, 0] < 0.994 * right_side)
            & (sigma > 0.006)
            & (sigma < 0.99994)
        )
        jacobian = centered_jacobian(sources, state, valid)
        finite = np.isfinite(jacobian)
        valid_count = int(np.count_nonzero(valid))
        if interior_open and np.all(valid[fixed_interior]):
            interior_jacobian = np.where(fixed_interior, jacobian, np.nan)
            interior_finite = np.isfinite(interior_jacobian)
            if np.any(interior_finite):
                interior_minimum = float(np.nanmin(interior_jacobian))
                if (
                    interior_minimum
                    < interior_best["minimum fixed-interior J"]
                ):
                    interior_best = {
                        "minimum fixed-interior J": interior_minimum,
                        "age of minimum fixed-interior J (yr)": age,
                    }
            interior_end_age = age
        else:
            interior_open = False
        if np.any(finite) and valid_count >= minimum_valid_sources:
            minimum = float(np.nanmin(jacobian))
            span = float(np.ptp(sources[valid]))
            snapshot = {
                "age (yr)": age,
                "minimum J": minimum,
                "valid source count": valid_count,
                "valid source span (m)": span,
            }
            if minimum < best["minimum J"]:
                best = {
                    "minimum J": minimum,
                    "age of minimum J (yr)": age,
                    "valid source count at minimum": valid_count,
                    "valid source span at minimum (m)": span,
                }
            if first_negative is None and minimum < 0.0:
                first_negative = snapshot
            if first_resolved is None and minimum < -0.20:
                first_resolved = snapshot
        if step_index * time_step >= maximum_age or valid_count < 9:
            break
        k1 = velocity(state)
        minimum_path_u = min(minimum_path_u, float(np.nanmin(k1[valid, 0])))
        k2 = velocity(state + 0.5 * time_step * k1)
        k3 = velocity(state + 0.5 * time_step * k2)
        k4 = velocity(state + time_step * k3)
        state += time_step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    best.update(
        {
            "first negative": first_negative,
            "first resolved overturn": first_resolved,
            "minimum path u (m/yr)": minimum_path_u,
            "deposition source count": int(len(sources)),
            "deposition source span (m)": float(np.ptp(sources)),
            "accumulation min (m/yr)": float(np.min(accumulation)),
            "accumulation max (m/yr)": float(np.max(accumulation)),
            "minimum fixed-interior J": (
                interior_best["minimum fixed-interior J"]
                if np.isfinite(interior_best["minimum fixed-interior J"])
                else None
            ),
            "age of minimum fixed-interior J (yr)": (
                interior_best["age of minimum fixed-interior J (yr)"]
                if np.isfinite(
                    interior_best["age of minimum fixed-interior J (yr)"]
                )
                else None
            ),
            "fixed-interior source count": int(np.count_nonzero(fixed_interior)),
            "fixed-interior source span (m)": float(
                np.ptp(sources[fixed_interior])
            ),
            "fixed-interior evaluation end age (yr)": interior_end_age,
        }
    )
    return best


def run_case(
    geometry,
    resolution,
    maximum_age=4_000.0,
    time_step=1.0,
    source_count=401,
):
    velocity, diagnostics = solve_stokes(geometry, resolution)
    sample, speed_range = make_sampler(geometry, velocity)
    material_map = trace_surface_map(
        geometry,
        sample,
        maximum_age=maximum_age,
        time_step=time_step,
        source_count=source_count,
    )
    left_side, right_side = side_locations(geometry)
    return {
        "geometry": geometry["name"],
        "target spacing (m)": resolution,
        "domain length (m)": right_side - left_side,
        "left boundary thickness (m)": -float(bed_height(left_side, geometry)),
        "right boundary thickness (m)": -float(bed_height(right_side, geometry)),
        "maximum thickness (m)": (
            geometry["vertical radius"] - geometry["center height"]
        ),
        "minimum grid u (m/yr)": speed_range[0],
        "maximum grid u (m/yr)": speed_range[1],
    } | material_map | diagnostics


def flatten_result(result):
    """Expand event dictionaries into scalar columns for CSV output."""
    flat = result.copy()
    for source_name, column_name in (
        ("first negative", "first negative"),
        ("first resolved overturn", "first resolved"),
    ):
        event = flat.pop(source_name)
        flat[f"{column_name} age (yr)"] = None if event is None else event["age (yr)"]
        flat[f"{column_name} minimum J"] = (
            None if event is None else event["minimum J"]
        )
        flat[f"{column_name} valid source count"] = (
            None if event is None else event["valid source count"]
        )
        flat[f"{column_name} valid source span (m)"] = (
            None if event is None else event["valid source span (m)"]
        )
    return flat


def material_curve_snapshots(
    geometry,
    sample,
    ages=(0.0, 700.0, 1_400.0, 2_100.0, 2_800.0),
    time_step=1.0,
    source_count=401,
):
    """Return common-age material curves, removing markers at first exit."""
    left_side, right_side = side_locations(geometry)
    sources = np.linspace(0.96 * left_side, 0.04 * left_side, source_count)
    state = np.column_stack((sources, np.full_like(sources, -0.05)))
    accumulation = -sample("w", sources, state[:, 1])
    keep = accumulation > 1.0e-6
    sources = sources[keep]
    state = state[keep]
    active = np.ones(len(sources), dtype=bool)
    requested_steps = {
        int(round(age / time_step)): float(age) for age in ages
    }
    maximum_step = max(requested_steps)
    snapshots = []

    def velocity(values):
        return np.column_stack(
            (
                sample("u", values[:, 0], values[:, 1]),
                sample("w", values[:, 0], values[:, 1]),
            )
        )

    for step_index in range(maximum_step + 1):
        age = step_index * time_step
        local_bed = bed_height(state[:, 0], geometry)
        sigma = (state[:, 1] - local_bed) / (-local_bed)
        inside = (
            np.isfinite(state[:, 0])
            & (state[:, 0] > 0.994 * left_side)
            & (state[:, 0] < 0.994 * right_side)
            & (sigma > 0.006)
            & (sigma < 0.99994)
        )
        active &= inside
        if step_index in requested_steps:
            jacobian = centered_jacobian(sources, state, active)
            snapshots.append({
                "age": age,
                "sources": sources,
                "states": state.copy(),
                "valid": active.copy(),
                "jacobian": jacobian,
            })
        if step_index == maximum_step:
            break
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            continue
        current = state[indices]
        k1 = velocity(current)
        k2 = velocity(current + 0.5 * time_step * k1)
        k3 = velocity(current + 0.5 * time_step * k2)
        k4 = velocity(current + time_step * k3)
        finite_step = np.all(
            np.isfinite(np.column_stack((k1, k2, k3, k4))), axis=1
        )
        state[indices[finite_step]] += time_step * (
            k1[finite_step]
            + 2.0 * k2[finite_step]
            + 2.0 * k3[finite_step]
            + k4[finite_step]
        ) / 6.0
        active[indices[~finite_step]] = False
    return snapshots


def save_map_figure(path, resolution=7.5):
    """Overlay common-age material curves for two overturning troughs."""
    selected = (GEOMETRIES[1], GEOMETRIES[4])
    ages = (0.0, 700.0, 1_400.0, 2_100.0, 2_800.0)
    figure = plt.figure(figsize=(12.0, 5.7))
    grid = figure.add_gridspec(
        1,
        2,
        left=0.07,
        right=0.985,
        bottom=0.27,
        top=0.92,
        wspace=0.12,
    )
    axes = np.asarray(
        [
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
        ]
    )
    all_snapshots = []
    for column, geometry in enumerate(selected):
        velocity, _ = solve_stokes(geometry, resolution)
        sample, _ = make_sampler(geometry, velocity)
        snapshots = material_curve_snapshots(geometry, sample, ages=ages)
        all_snapshots.append(snapshots)
    source_min = min(
        np.min(snapshot["sources"])
        for snapshots in all_snapshots
        for snapshot in snapshots
    )
    source_max = max(
        np.max(snapshot["sources"])
        for snapshots in all_snapshots
        for snapshot in snapshots
    )
    source_norm = plt.Normalize(source_min, source_max)
    cmap = plt.get_cmap("viridis")

    label_fractions = (0.55, 0.30, 0.18, 0.47, 0.55)
    label_offsets = ((0, -18), (-8, 10), (-12, 1), (10, 1), (10, -9))
    panel_titles = (
        "Narrow symmetric trough",
        "Steep-entry, gentle-exit trough",
    )
    for column, (geometry, snapshots) in enumerate(zip(selected, all_snapshots)):
        axis = axes[column]
        left_side, right_side = side_locations(geometry)
        x = np.linspace(left_side, right_side, 801)
        bed = bed_height(x, geometry)
        axis.fill_between(x, bed, np.min(bed) - 10.0, color="0.88")
        axis.plot(x, bed, color="0.25", linewidth=0.9)
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        for age_index, snapshot in enumerate(snapshots):
            states = snapshot["states"]
            valid = snapshot["valid"]
            valid_pairs = valid[:-1] & valid[1:]
            segments = np.stack((states[:-1], states[1:]), axis=1)[valid_pairs]
            source_midpoints = 0.5 * (
                snapshot["sources"][:-1] + snapshot["sources"][1:]
            )[valid_pairs]
            if len(segments):
                axis.add_collection(
                    LineCollection(
                        segments,
                        array=source_midpoints,
                        cmap=cmap,
                        norm=source_norm,
                        linewidth=2.2,
                        zorder=3 + age_index,
                    )
                )
            overturned = valid & np.isfinite(snapshot["jacobian"])
            overturned &= snapshot["jacobian"] < 0.0
            overturned_indices = np.flatnonzero(overturned)[::10]
            axis.scatter(
                states[overturned_indices, 0],
                states[overturned_indices, 1],
                s=10,
                facecolors="none",
                edgecolors="#e41a1c",
                linewidths=0.6,
                zorder=9,
            )
            finite_j = snapshot["jacobian"][np.isfinite(snapshot["jacobian"])]
            j_text = ""
            if len(finite_j):
                displayed_j = float(np.min(finite_j))
                if abs(displayed_j) < 0.005:
                    sign = "-" if displayed_j < 0.0 else "+"
                    j_text = rf"; $J_{{\min}}\!\approx\!0^{{{sign}}}$"
                else:
                    j_text = rf"; $J_{{\min}}\!=\!{displayed_j:.2f}$"
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices):
                label_index = valid_indices[
                    min(
                        int(label_fractions[age_index] * len(valid_indices)),
                        len(valid_indices) - 1,
                    )
                ]
                age_label = f"{snapshot['age'] / 1000:.1f} kyr{j_text}"
                axis.annotate(
                    age_label,
                    xy=states[label_index],
                    xytext=label_offsets[age_index],
                    textcoords="offset points",
                    ha=(
                        "center"
                        if age_index == 0
                        else "right"
                        if age_index == len(ages) - 1
                        else "left"
                    ),
                    va="center",
                    fontsize=8.0,
                    color="0.12",
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                    },
                    zorder=10,
                )
        axis.set(
            xlim=(left_side, right_side),
            ylim=(min(-160.0, np.min(bed) - 5.0), 4.0),
            xlabel="Horizontal coordinate (m)",
        )
        axis.set_title(
            f"{'AB'[column]}.  {panel_titles[column]}",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        axis.grid(color="0.92", linewidth=0.5, zorder=0)
    axes[0].set_ylabel("Elevation (m)")
    axes[1].tick_params(labelleft=False)
    scalar_mappable = plt.cm.ScalarMappable(norm=source_norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar_axis = figure.add_axes((0.20, 0.075, 0.60, 0.035))
    colorbar = figure.colorbar(
        scalar_mappable,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label(r"Deposition coordinate $x_0$ (m)")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#e41a1c",
                label=(
                    r"open red circles: centered $J<0$; "
                    "markers removed at first exit"
                ),
            )
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.145),
        fontsize=8.5,
    )
    figure.savefig(path, dpi=220)
    if str(path).lower().endswith(".png"):
        figure.savefig(str(path)[:-4] + ".pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, action="append")
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--maximum-age", type=float, default=4_000.0)
    parser.add_argument("--time-step", type=float, default=1.0)
    parser.add_argument("--source-count", type=int, default=401)
    parser.add_argument(
        "--output-prefix",
        help="Optional path prefix for a CSV and compact PNG/PDF summary",
    )
    parser.add_argument(
        "--map-figure",
        help="Optional PNG path for narrow-symmetric and asymmetric map panels",
    )
    args = parser.parse_args()
    indices = args.index if args.index is not None else range(len(GEOMETRIES))
    results = []
    for index in indices:
        geometry = GEOMETRIES[index]
        print(f"Solving {index}: {geometry['name']}", flush=True)
        result = run_case(
            geometry,
            args.resolution,
            maximum_age=args.maximum_age,
            time_step=args.time_step,
            source_count=args.source_count,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    if args.output_prefix:
        table = pd.DataFrame([flatten_result(result) for result in results])
        output_name = Path(args.output_prefix).name
        repository = Path(__file__).resolve().parents[1]
        csv_path = repository / "data" / f"{output_name}.csv"
        figure_prefix = repository / "figures" / output_name
        table.to_csv(csv_path, index=False, na_rep="NA")
        figure_height = max(3.2, 0.42 * len(table))
        figure, axis = plt.subplots(figsize=(7.2, figure_height), constrained_layout=True)
        plot_table = table.dropna(subset=["minimum J"]).sort_values("minimum J")
        colors = np.where(plot_table["minimum J"] < 0.0, "0.2", "0.7")
        axis.barh(plot_table["geometry"], plot_table["minimum J"], color=colors)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set(
            xlabel=r"Minimum surface-map order $\partial x/\partial x_0$",
            ylabel="",
            title="Finite elliptical-trough family",
        )
        figure.savefig(f"{figure_prefix}.png", dpi=220)
        figure.savefig(f"{figure_prefix}.pdf")
    if args.map_figure:
        save_map_figure(args.map_figure)


if __name__ == "__main__":
    main()
