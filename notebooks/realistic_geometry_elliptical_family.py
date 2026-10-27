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
from matplotlib.patches import Rectangle
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
    """Advect one accumulating cohort and test loss of horizontal order."""
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
        }

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
    for step_index in range(int(np.ceil(maximum_age / time_step)) + 1):
        age = step_index * time_step
        local_bed = bed_height(state[:, 0], geometry)
        sigma = (state[:, 1] - local_bed) / (-local_bed)
        valid = (
            np.isfinite(state[:, 0])
            & (state[:, 0] > 0.994 * left_side)
            & (state[:, 0] < 0.994 * right_side)
            & (sigma > 0.006)
            & (sigma < 0.9998)
        )
        jacobian = centered_jacobian(sources, state, valid)
        finite = np.isfinite(jacobian)
        valid_count = int(np.count_nonzero(valid))
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


def first_resolved_snapshot(
    geometry,
    sample,
    threshold=-0.20,
    maximum_age=4_000.0,
    time_step=1.0,
    source_count=401,
):
    """Return particle positions when a well-resolved map first crosses a threshold."""
    left_side, right_side = side_locations(geometry)
    sources = np.linspace(0.96 * left_side, 0.04 * left_side, source_count)
    state = np.column_stack((sources, np.full_like(sources, -0.05)))
    accumulation = -sample("w", sources, state[:, 1])
    keep = accumulation > 1.0e-6
    sources = sources[keep]
    state = state[keep]

    def velocity(values):
        return np.column_stack(
            (
                sample("u", values[:, 0], values[:, 1]),
                sample("w", values[:, 0], values[:, 1]),
            )
        )

    for step_index in range(int(np.ceil(maximum_age / time_step)) + 1):
        age = step_index * time_step
        local_bed = bed_height(state[:, 0], geometry)
        sigma = (state[:, 1] - local_bed) / (-local_bed)
        valid = (
            np.isfinite(state[:, 0])
            & (state[:, 0] > 0.994 * left_side)
            & (state[:, 0] < 0.994 * right_side)
            & (sigma > 0.006)
            & (sigma < 0.9998)
        )
        jacobian = centered_jacobian(sources, state, valid)
        if (
            np.count_nonzero(valid) >= 21
            and np.any(np.isfinite(jacobian))
            and np.nanmin(jacobian) < threshold
        ):
            return {
                "age": age,
                "minimum J": float(np.nanmin(jacobian)),
                "sources": sources,
                "states": state.copy(),
                "valid": valid,
                "jacobian": jacobian,
            }
        if step_index * time_step >= maximum_age or np.count_nonzero(valid) < 9:
            break
        k1 = velocity(state)
        k2 = velocity(state + 0.5 * time_step * k1)
        k3 = velocity(state + 0.5 * time_step * k2)
        k4 = velocity(state + time_step * k3)
        state += time_step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    raise RuntimeError(f"{geometry['name']} did not cross J={threshold:g}")


def save_map_figure(path, resolution=7.5):
    """Plot two geometrically distinct maps at their first resolved overturn."""
    selected = (GEOMETRIES[1], GEOMETRIES[4])
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(10.5, 10.8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.05, 0.85, 0.75)},
    )
    for column, geometry in enumerate(selected):
        full_axis = axes[0, column]
        zoom_axis = axes[1, column]
        order_axis = axes[2, column]
        velocity, _ = solve_stokes(geometry, resolution)
        sample, _ = make_sampler(geometry, velocity)
        snapshot = first_resolved_snapshot(geometry, sample)
        left_side, right_side = side_locations(geometry)
        x = np.linspace(left_side, right_side, 801)
        bed = bed_height(x, geometry)
        full_axis.fill_between(x, bed, np.min(bed) - 12.0, color="0.85")
        full_axis.plot(x, bed, color="0.25", linewidth=0.9)
        full_axis.axhline(0.0, color="0.25", linewidth=0.9)

        states = snapshot["states"]
        valid_pairs = snapshot["valid"][:-1] & snapshot["valid"][1:]
        segments = np.stack((states[:-1], states[1:]), axis=1)[valid_pairs]
        pair_jacobian = 0.5 * (
            snapshot["jacobian"][:-1] + snapshot["jacobian"][1:]
        )[valid_pairs]
        colors = np.where(pair_jacobian < 0.0, "#b2182b", "#2166ac")
        full_axis.add_collection(
            LineCollection(segments, colors=colors, linewidth=2.0)
        )
        zoom_axis.add_collection(
            LineCollection(segments, colors=colors, linewidth=2.4)
        )
        full_axis.set(
            xlim=(left_side, right_side),
            ylim=(np.min(bed) - 5.0, 5.0),
            ylabel="Elevation (m)",
            title=(
                f"{geometry['name']}\n"
                f"{snapshot['age'] / 1000:.3f} kyr, "
                f"$J_{{min}}={snapshot['minimum J']:.3f}$"
            ),
        )
        plotted = states[snapshot["valid"]]
        x_padding = max(0.08, 0.12 * np.ptp(plotted[:, 0]))
        z_padding = max(0.4, 0.08 * np.ptp(plotted[:, 1]))
        zoom_limits = (
            np.min(plotted[:, 0]) - x_padding,
            np.max(plotted[:, 0]) + x_padding,
            np.min(plotted[:, 1]) - z_padding,
            np.max(plotted[:, 1]) + z_padding,
        )
        full_axis.add_patch(
            Rectangle(
                (zoom_limits[0], zoom_limits[2]),
                zoom_limits[1] - zoom_limits[0],
                zoom_limits[3] - zoom_limits[2],
                fill=False,
                edgecolor="0.15",
                linewidth=1.0,
                linestyle=":",
            )
        )
        zoom_axis.set(
            xlim=zoom_limits[:2],
            ylim=zoom_limits[2:],
            xlabel="Horizontal coordinate (m)",
            ylabel="Elevation (m)",
            title=(
                "Tight material-map view "
                f"({np.ptp(snapshot['sources'][snapshot['valid']]):.1f} m source span)"
            ),
        )
        source_pairs = np.stack(
            (
                np.column_stack((snapshot["sources"][:-1], states[:-1, 0])),
                np.column_stack((snapshot["sources"][1:], states[1:, 0])),
            ),
            axis=1,
        )[valid_pairs]
        order_axis.add_collection(
            LineCollection(source_pairs, colors=colors, linewidth=2.4)
        )
        valid_sources = snapshot["sources"][snapshot["valid"]]
        valid_x = states[snapshot["valid"], 0]
        source_padding = 0.03 * np.ptp(valid_sources)
        current_padding = 0.08 * np.ptp(valid_x)
        order_axis.set(
            xlim=(
                np.min(valid_sources) - source_padding,
                np.max(valid_sources) + source_padding,
            ),
            ylim=(np.min(valid_x) - current_padding, np.max(valid_x) + current_padding),
            xlabel=r"Deposition origin $x_0$ (m)",
            ylabel=r"Current $x$ (m)",
            title=r"Order diagnostic: red has $\partial x/\partial x_0<0$",
        )
    legend_handles = [
        Line2D([0], [0], color="#2166ac", linewidth=2.4, label="order retained"),
        Line2D([0], [0], color="#b2182b", linewidth=2.4, label="overturned limb"),
    ]
    axes[0, 0].legend(handles=legend_handles, frameon=False, loc="upper left")
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
