"""Steady isochrones over one bed spike in a finite, nonperiodic reach.

The calculation solves two-dimensional, incompressible, linear-viscous Stokes
flow.  Coordinates follow a mean surface inclined by ``SURFACE_SLOPE``; the
model surface is therefore horizontal and gravity has along- and cross-slope
components.  The bed obeys no penetration and the passive linear law

    tangential traction + BASAL_DRAG * tangential velocity = 0.

At the two finite ends, the applied traction is the exact traction of a flat,
uniform slab with the same thickness, gravity, viscosity, and sliding law.
The spike vanishes well before either end.  This makes the ends transparent
to the reference through-flow without prescribing velocity.  The upper
surface is traction free.  Its steady accumulation/ablation rate is diagnosed
from the kinematic condition; only inflowing (accumulating) parts of the
surface are used as material-age sources.

All lengths are metres, time is years, velocity is m/yr, stress is Pa, and
viscosity is Pa yr.
"""

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
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq


LENGTH = 1200.0
THICKNESS = 160.0
DENSITY = 917.0
GRAVITY = 9.81
VISCOSITY = 2.5e8
SURFACE_SLOPE = 0.05
BASAL_DRAG = 1.0e5
NITSCHE_PENALTY = 3.0e6

SPIKES = [
    {"name": "low broad", "height": 80.0, "half width": 160.0},
    {"name": "high broad", "height": 120.0, "half width": 120.0},
    {"name": "high steep", "height": 120.0, "half width": 70.0},
    {"name": "high very steep", "height": 120.0, "half width": 45.0},
    {"name": "near pinch broad", "height": 145.0, "half width": 100.0},
    {"name": "near pinch steep", "height": 145.0, "half width": 60.0},
]

SURVEY_MESH = (160, 24)
CONVERGENCE_MESHES = [(160, 24), (240, 36), (320, 48)]
SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def bed_height(x, spike):
    """Compact raised-cosine spike on an otherwise flat bed."""
    x = np.asarray(x)
    length = spike.get("domain length", LENGTH)
    distance = x - 0.5 * length
    half_width = spike["half width"]
    inside = np.abs(distance) < half_width
    bed = np.zeros_like(distance, dtype=float)
    bed[inside] = 0.5 * spike["height"] * (
        1.0 + np.cos(np.pi * distance[inside] / half_width)
    )
    return bed


def maximum_bed_slope(spike):
    return 0.5 * np.pi * spike["height"] / spike["half width"]


def focused_surface_sources(spike, onset=None, minimum_offset=0.25):
    """Resolve the accumulation onset and the downstream source interval."""
    length = spike.get("domain length", LENGTH)
    if onset is None:
        onset = 0.5 * length
    offsets = np.unique(
        np.r_[
            np.geomspace(minimum_offset, 20.0, 41),
            np.linspace(20.0, 0.45 * length, 281),
        ]
    )
    return onset + offsets


def represented_bed(x, spike, nx):
    """Piecewise-linear bed that is actually represented on the mesh."""
    length = spike.get("domain length", LENGTH)
    mesh_x = np.linspace(0.0, length, nx + 1)
    mesh_bed = bed_height(mesh_x, spike)
    return np.interp(np.asarray(x), mesh_x, mesh_bed, left=np.nan, right=np.nan)


def make_mesh(spike, nx, nz):
    """Map a rectangle to the spike while preserving boundary markers."""
    length = spike.get("domain length", LENGTH)
    mesh = fd.RectangleMesh(nx, nz, length, 1.0)
    reference = mesh.coordinates.dat.data_ro.copy()
    bed = bed_height(reference[:, 0], spike)
    mesh.coordinates.dat.data[:, 1] = bed + reference[:, 1] * (THICKNESS - bed)
    mesh.clear_spatial_index()
    return mesh


def flat_slab_fields(mesh):
    """Exact flat-slab velocity, pressure, and side tractions."""
    x = fd.SpatialCoordinate(mesh)
    along_force = DENSITY * GRAVITY * np.sin(SURFACE_SLOPE)
    normal_force = DENSITY * GRAVITY * np.cos(SURFACE_SLOPE)
    sliding_speed = along_force * THICKNESS / BASAL_DRAG
    exact_velocity = fd.as_vector(
        (
            sliding_speed
            + along_force
            * (THICKNESS * x[1] - 0.5 * x[1] ** 2)
            / VISCOSITY,
            0.0,
        )
    )
    exact_pressure = normal_force * (THICKNESS - x[1])
    shear_stress = along_force * (THICKNESS - x[1])
    left_traction = fd.as_vector((exact_pressure, -shear_stress))
    right_traction = fd.as_vector((-exact_pressure, shear_stress))
    return exact_velocity, exact_pressure, left_traction, right_traction


def solve_stokes(spike, nx, nz):
    """Solve one finite-reach Stokes problem."""
    mesh = make_mesh(spike, nx, nz)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)
    u, p = fd.split(solution)
    v, q = fd.TestFunctions(mixed_space)
    body_force = fd.Constant(
        (
            DENSITY * GRAVITY * np.sin(SURFACE_SLOPE),
            -DENSITY * GRAVITY * np.cos(SURFACE_SLOPE),
        )
    )
    stress = 2.0 * VISCOSITY * fd.sym(fd.grad(u)) - p * fd.Identity(2)
    test_stress = 2.0 * VISCOSITY * fd.sym(fd.grad(v)) - q * fd.Identity(2)
    residual = (
        2.0 * VISCOSITY * fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v)))
        - p * fd.div(v)
        - q * fd.div(u)
        - fd.dot(body_force, v)
    ) * fd.dx

    normal = fd.FacetNormal(mesh)
    tangent = fd.as_vector((-normal[1], normal[0]))
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
    ) * fd.ds(3)

    exact_velocity, exact_pressure, left_traction, right_traction = flat_slab_fields(mesh)
    residual -= fd.dot(left_traction, v) * fd.ds(1)
    residual -= fd.dot(right_traction, v) * fd.ds(2)
    solution.sub(0).interpolate(exact_velocity)
    solution.sub(1).interpolate(exact_pressure)
    fd.solve(residual == 0, solution, solver_parameters=SOLVER_PARAMETERS)
    velocity, pressure = solution.subfunctions
    gradient_space = fd.TensorFunctionSpace(mesh, "DG", 1)
    velocity_gradient = fd.Function(gradient_space).interpolate(fd.grad(velocity))

    bed_length = fd.assemble(fd.Constant(1.0) * fd.ds(3, domain=mesh))
    basal_speed = fd.dot(velocity, tangent)
    gravity_power = fd.assemble(fd.dot(body_force, velocity) * fd.dx)
    side_power = fd.assemble(
        fd.dot(left_traction, velocity) * fd.ds(1)
        + fd.dot(right_traction, velocity) * fd.ds(2)
    )
    basal_dissipation = fd.assemble(BASAL_DRAG * basal_speed**2 * fd.ds(3))
    viscous_dissipation = fd.assemble(
        2.0
        * VISCOSITY
        * fd.inner(fd.sym(fd.grad(velocity)), fd.sym(fd.grad(velocity)))
        * fd.dx
    )
    diagnostics = {
        "cells": mesh.num_cells(),
        "gravity power": gravity_power,
        "side power": side_power,
        "basal dissipation / total input": basal_dissipation
        / (gravity_power + side_power),
        "relative energy residual": (
            gravity_power + side_power - basal_dissipation - viscous_dissipation
        )
        / (gravity_power + side_power),
        "rms basal normal speed (m/yr)": np.sqrt(
            fd.assemble(fd.dot(velocity, normal) ** 2 * fd.ds(3)) / bed_length
        ),
        "minimum nodal u (m/yr)": np.min(velocity.dat.data_ro[:, 0]),
        "maximum nodal u (m/yr)": np.max(velocity.dat.data_ro[:, 0]),
    }
    return mesh, velocity, velocity_gradient, diagnostics


def flat_slab_validation(nx=80, nz=16):
    """Check the finite-end tractions and sliding implementation at zero relief."""
    spike = {"name": "flat", "height": 0.0, "half width": 100.0}
    mesh, velocity, _, diagnostics = solve_stokes(spike, nx, nz)
    exact_velocity, _, _, _ = flat_slab_fields(mesh)
    error = fd.sqrt(fd.assemble(fd.inner(velocity - exact_velocity, velocity - exact_velocity) * fd.dx))
    scale = fd.sqrt(fd.assemble(fd.inner(exact_velocity, exact_velocity) * fd.dx))
    diagnostics["relative flat velocity L2 error"] = float(error / scale)
    return diagnostics


def make_sampler(spike, mesh_nx, velocity, velocity_gradient, nx=601, nz=181):
    """Interpolate the finite-element velocity in terrain-following coordinates."""
    length = spike.get("domain length", LENGTH)
    margin = 1.0e-5 * length
    x = np.linspace(margin, length - margin, nx)
    sigma = np.linspace(0.003, 0.99995, nz)
    bed = represented_bed(x, spike, mesh_nx)
    X = np.tile(x, (nz, 1))
    Z = bed[None, :] + sigma[:, None] * (THICKNESS - bed)[None, :]
    points = np.column_stack((X.ravel(), Z.ravel()))
    values = np.asarray(velocity.at(points, tolerance=1.0e-7)).reshape(Z.shape + (2,))
    fields = {
        name: RegularGridInterpolator(
            (sigma, x),
            values[..., component],
            bounds_error=False,
            fill_value=np.nan,
        )
        for component, name in enumerate(("u", "w"))
    }
    gradient_values = np.asarray(
        velocity_gradient.at(points, tolerance=1.0e-7)
    ).reshape(Z.shape + (2, 2))
    for index, name in np.ndenumerate(
        np.array([["ux", "uz"], ["wx", "wz"]])
    ):
        fields[name] = RegularGridInterpolator(
            (sigma, x),
            gradient_values[..., index[0], index[1]],
            bounds_error=False,
            fill_value=np.nan,
        )

    def sample(name, x_value, z_value):
        x_values, z_values = np.broadcast_arrays(x_value, z_value)
        local_bed = represented_bed(x_values, spike, mesh_nx)
        local_sigma = (z_values - local_bed) / (THICKNESS - local_bed)
        points = np.column_stack((local_sigma.ravel(), x_values.ravel()))
        result = fields[name](points).reshape(x_values.shape)
        return float(result) if result.ndim == 0 else result

    return sample, (np.nanmin(values[..., 0]), np.nanmax(values[..., 0]))


def trace_surface_map(
    spike,
    mesh_nx,
    sample,
    maximum_age=100_000.0,
    time_step=2.0,
    source_count=601,
    source_positions=None,
    accumulation_tolerance=None,
):
    """Advect simultaneous surface deposits and find their minimum order Jacobian."""
    length = spike.get("domain length", LENGTH)
    if source_positions is None:
        all_sources = np.linspace(0.03 * length, 0.97 * length, source_count)
    else:
        all_sources = np.sort(np.asarray(source_positions))
    start_z = np.full_like(all_sources, THICKNESS - 0.05)
    surface_w = sample("w", all_sources, start_z)
    accumulation = -surface_w
    if accumulation_tolerance is None:
        accumulation_tolerance = max(
            1.0e-9, 1.0e-7 * np.nanmax(np.abs(accumulation))
        )
    source_mask = accumulation > accumulation_tolerance
    sources = all_sources[source_mask]
    state = np.column_stack(
        (
            sources,
            start_z[source_mask],
            np.ones_like(sources),
            np.zeros_like(sources),
        )
    )

    if len(sources) < 5:
        return {
            "minimum J": np.nan,
            "minimum neighboring J": np.nan,
            "age of minimum J (yr)": np.nan,
            "minimum path u (m/yr)": np.nan,
            "surface accumulation min (m/yr)": np.nan,
            "surface accumulation max (m/yr)": np.nan,
            "source count": len(sources),
            "sources": sources,
            "states": state[:, :2],
            "jacobian": np.full(len(sources), np.nan),
            "valid": np.zeros(len(sources), dtype=bool),
            "individual minimum J": np.full(len(sources), np.nan),
            "individual age of minimum J (yr)": np.full(len(sources), np.nan),
            "individual last valid age (yr)": np.full(len(sources), np.nan),
            "active at age ceiling": 0,
            "age ceiling (yr)": maximum_age,
        }

    def right_hand_side(values):
        positions = values[:, :2]
        tangent = values[:, 2:]
        velocity = np.column_stack(
            (
                sample("u", positions[:, 0], positions[:, 1]),
                sample("w", positions[:, 0], positions[:, 1]),
            )
        )
        gradient = np.empty((len(values), 2, 2))
        gradient[:, 0, 0] = sample("ux", positions[:, 0], positions[:, 1])
        gradient[:, 0, 1] = sample("uz", positions[:, 0], positions[:, 1])
        gradient[:, 1, 0] = sample("wx", positions[:, 0], positions[:, 1])
        gradient[:, 1, 1] = sample("wz", positions[:, 0], positions[:, 1])
        tangent_rate = np.einsum("nij,nj->ni", gradient, tangent)
        return np.column_stack((velocity, tangent_rate))

    best = {
        "minimum J": np.inf,
        "age of minimum J (yr)": np.nan,
        "sources": None,
        "states": None,
        "jacobian": None,
        "valid": None,
    }
    individual_minimum = np.full(len(sources), np.inf)
    individual_age = np.full(len(sources), np.nan)
    individual_last_valid_age = np.full(len(sources), np.nan)
    minimum_path_speed = np.inf
    active_at_age_ceiling = 0
    steps = int(np.ceil(maximum_age / time_step))
    for step_index in range(steps + 1):
        age = step_index * time_step
        local_bed = represented_bed(state[:, 0], spike, mesh_nx)
        local_sigma = (state[:, 1] - local_bed) / (THICKNESS - local_bed)
        valid = (
            np.isfinite(state[:, 0])
            & (state[:, 0] > 0.003 * length)
            & (state[:, 0] < 0.997 * length)
            & (local_sigma > 0.005)
            & (local_sigma < 0.9999)
        )
        jacobian = np.full(len(sources), np.nan)
        jacobian[valid] = state[valid, 2]
        individual_last_valid_age[valid] = age
        improved = valid & (jacobian < individual_minimum)
        individual_minimum[improved] = jacobian[improved]
        individual_age[improved] = age
        if np.any(np.isfinite(jacobian)):
            minimum = np.nanmin(jacobian)
            if minimum < best["minimum J"]:
                best.update(
                    {
                        "minimum J": minimum,
                        "age of minimum J (yr)": age,
                        "sources": sources.copy(),
                        "states": state[:, :2].copy(),
                        "jacobian": jacobian.copy(),
                        "valid": valid.copy(),
                    }
                )
        if step_index == steps:
            active_at_age_ceiling = int(np.sum(valid))
            break
        if not np.any(valid):
            break
        k1 = right_hand_side(state)
        if np.any(valid):
            minimum_path_speed = min(minimum_path_speed, np.nanmin(k1[valid, 0]))
        k2 = right_hand_side(state + 0.5 * time_step * k1)
        k3 = right_hand_side(state + 0.5 * time_step * k2)
        k4 = right_hand_side(state + time_step * k3)
        state += time_step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    best.update(
        {
            "minimum path u (m/yr)": minimum_path_speed,
            "surface accumulation min (m/yr)": np.min(accumulation[source_mask]),
            "surface accumulation max (m/yr)": np.max(accumulation[source_mask]),
            "source count": len(sources),
            "individual minimum J": individual_minimum,
            "individual age of minimum J (yr)": individual_age,
            "individual last valid age (yr)": individual_last_valid_age,
            "active at age ceiling": active_at_age_ceiling,
            "age ceiling (yr)": maximum_age,
        }
    )
    neighboring_jacobian = np.full(len(sources), np.nan)
    valid_indices = np.where(best["valid"])[0]
    groups = np.split(valid_indices, np.where(np.diff(valid_indices) > 1)[0] + 1)
    for group in groups:
        if len(group) >= 5:
            neighboring_jacobian[group] = np.gradient(
                best["states"][group, 0], sources[group]
            )
    best["minimum neighboring J"] = (
        np.nanmin(neighboring_jacobian)
        if np.any(np.isfinite(neighboring_jacobian))
        else np.nan
    )
    best["neighboring jacobian"] = neighboring_jacobian
    return best


def surface_accumulation_onset(spike, sample):
    """Find the ablation-to-accumulation transition nearest the spike crest."""
    length = spike.get("domain length", LENGTH)
    center = 0.5 * length
    search_half_width = min(0.25 * length, 2.0 * spike["half width"])
    probes = np.linspace(
        center - search_half_width, center + search_half_width, 801
    )
    start_z = np.full_like(probes, THICKNESS - 0.05)
    accumulation = -sample("w", probes, start_z)
    crossings = np.where(
        (accumulation[:-1] <= 0.0) & (accumulation[1:] > 0.0)
    )[0]
    if len(crossings) == 0:
        raise RuntimeError("No ablation-to-accumulation transition near the spike.")
    index = crossings[np.argmin(np.abs(probes[crossings] - center))]

    def accumulation_at(x):
        return -sample("w", x, THICKNESS - 0.05)

    return brentq(accumulation_at, probes[index], probes[index + 1])


def run_case(spike, nx, nz, **trace_options):
    mesh, velocity, velocity_gradient, diagnostics = solve_stokes(spike, nx, nz)
    sample, sampled_speed_range = make_sampler(
        spike, nx, velocity, velocity_gradient
    )
    accumulation_onset = surface_accumulation_onset(spike, sample)
    if "source_positions" not in trace_options:
        trace_options["source_positions"] = focused_surface_sources(
            spike, accumulation_onset
        )
    material_map = trace_surface_map(spike, nx, sample, **trace_options)
    row = {
        "geometry": spike["name"],
        "mesh": f"{nx} x {nz}",
        "domain length (m)": spike.get("domain length", LENGTH),
        "accumulation onset x (m)": accumulation_onset,
        "smallest source offset from onset (m)": np.min(
            material_map["sources"] - accumulation_onset
        ),
        "height (m)": spike["height"],
        "half width (m)": spike["half width"],
        "minimum thickness (m)": THICKNESS - spike["height"],
        "maximum bed slope": maximum_bed_slope(spike),
        "maximum bed angle (degrees)": np.degrees(np.arctan(maximum_bed_slope(spike))),
        "minimum sampled u (m/yr)": sampled_speed_range[0],
        "maximum sampled u (m/yr)": sampled_speed_range[1],
        "minimum J": material_map["minimum J"],
        "minimum neighboring J": material_map["minimum neighboring J"],
        "age of minimum J (yr)": material_map["age of minimum J (yr)"],
        "minimum path u (m/yr)": material_map["minimum path u (m/yr)"],
        "surface accumulation min (m/yr)": material_map[
            "surface accumulation min (m/yr)"
        ],
        "surface accumulation max (m/yr)": material_map[
            "surface accumulation max (m/yr)"
        ],
        "surface source count": material_map["source count"],
        "active sources at age ceiling": material_map["active at age ceiling"],
        "latest last-valid age (yr)": np.nanmax(
            material_map["individual last valid age (yr)"]
        ),
    } | diagnostics
    return row, material_map


def plot_best_cases(saved_cases, output):
    selected = sorted(saved_cases, key=lambda name: saved_cases[name][1]["minimum J"])[:4]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    for axis, name in zip(axes.ravel(), selected):
        spike, material_map = saved_cases[name]
        length = spike.get("domain length", LENGTH)
        x = np.linspace(0.0, length, 1001)
        bed = bed_height(x, spike)
        axis.fill_between(x, bed, -12.0, color="0.82")
        axis.plot(x, bed, color="0.25", linewidth=0.8)
        axis.axhline(THICKNESS, color="0.25", linewidth=0.8)
        states = material_map["states"]
        if states is not None:
            valid = material_map["valid"] & np.isfinite(material_map["jacobian"])
            points = states[valid]
            sources = material_map["sources"][valid]
            if len(points) > 1:
                typical_spacing = np.nanmedian(np.diff(sources))
                neighbors = np.diff(sources) < 5.0 * typical_spacing
                segments = np.stack((points[:-1], points[1:]), axis=1)[neighbors]
                collection = LineCollection(segments, cmap="viridis", linewidth=1.8)
                collection.set_array(
                    (0.5 * (sources[:-1] + sources[1:]))[neighbors]
                )
                axis.add_collection(collection)
                negative = material_map["jacobian"][valid] < 0.0
                axis.scatter(
                    points[negative, 0],
                    points[negative, 1],
                    s=22,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=0.9,
                )
        axis.set(
            xlim=(0.0, length),
            ylim=(-10.0, THICKNESS + 8.0),
            xlabel="Along-flow coordinate (m)",
            ylabel="Elevation (m)",
            title=(
                f"{name}: $J_{{min}}={material_map['minimum J']:.3f}$, "
                f"{material_map['age of minimum J (yr)'] / 1000.0:.2f} kyr"
            ),
        )
    figure.savefig(output, dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    print("Flat-slab validation:", flat_slab_validation(), flush=True)
    rows = []
    saved_cases = {}
    for spike in SPIKES:
        print(f"Surveying {spike['name']}.", flush=True)
        row, material_map = run_case(spike, *SURVEY_MESH)
        print(row, flush=True)
        rows.append(row)
        saved_cases[spike["name"]] = (spike, material_map)

    survey = pd.DataFrame(rows)
    reference_name = survey.sort_values("minimum J").iloc[0]["geometry"]
    reference = next(spike for spike in SPIKES if spike["name"] == reference_name)
    convergence_rows = []
    for nx, nz in CONVERGENCE_MESHES:
        existing = survey[(survey["geometry"] == reference_name) & (survey["mesh"] == f"{nx} x {nz}")]
        if len(existing):
            convergence_rows.append(existing.iloc[0].to_dict())
        else:
            print(f"Convergence run for {reference_name} on {nx} x {nz}.", flush=True)
            row, _ = run_case(reference, nx, nz)
            convergence_rows.append(row)

    # Move the transparent sides while retaining approximately 7.5 m cells.
    domain_rows = []
    for length in (800.0, 1200.0, 1800.0):
        length_case = reference | {
            "name": f"{reference_name}, L={length:g} m",
            "domain length": length,
        }
        nx = int(round(length / 7.5))
        if length == LENGTH:
            existing = survey[survey["geometry"] == reference_name]
            domain_rows.append(existing.iloc[0].to_dict())
        else:
            print(f"Boundary-distance run for {reference_name} at L={length:g} m.", flush=True)
            row, _ = run_case(length_case, nx, SURVEY_MESH[1])
            domain_rows.append(row)

    # Integrator/source convergence uses the finest velocity field once.
    print(f"Map convergence for {reference_name}.", flush=True)
    _, fine_velocity, fine_gradient, _ = solve_stokes(
        reference, *CONVERGENCE_MESHES[-1]
    )
    fine_sample, _ = make_sampler(
        reference,
        CONVERGENCE_MESHES[-1][0],
        fine_velocity,
        fine_gradient,
    )
    map_rows = []
    fine_onset = surface_accumulation_onset(reference, fine_sample)
    focused_sources = focused_surface_sources(reference, fine_onset)
    boundary_sources = fine_onset + np.array(
        [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0]
    )
    for time_step in (4.0, 2.0, 1.0, 0.5):
        material_map = trace_surface_map(
            reference,
            CONVERGENCE_MESHES[-1][0],
            fine_sample,
            source_positions=focused_sources,
            time_step=time_step,
            maximum_age=20_000.0,
        )
        map_rows.append(
            {
                "geometry": reference_name,
                "mesh": f"{CONVERGENCE_MESHES[-1][0]} x {CONVERGENCE_MESHES[-1][1]}",
                "surface origins": len(focused_sources),
                "smallest source offset (m)": np.min(
                    focused_sources - fine_onset
                ),
                "time step (yr)": time_step,
                "minimum J": material_map["minimum J"],
                "minimum neighboring J": material_map["minimum neighboring J"],
                "age of minimum J (yr)": material_map["age of minimum J (yr)"],
            }
        )

    boundary_map = trace_surface_map(
        reference,
        CONVERGENCE_MESHES[-1][0],
        fine_sample,
        source_positions=boundary_sources,
        time_step=2.0,
        accumulation_tolerance=0.0,
    )
    boundary_rows = []
    for source, minimum, age, last_valid_age in zip(
        boundary_map["sources"],
        boundary_map["individual minimum J"],
        boundary_map["individual age of minimum J (yr)"],
        boundary_map["individual last valid age (yr)"],
    ):
        boundary_rows.append(
            {
                "geometry": reference_name,
                "mesh": f"{CONVERGENCE_MESHES[-1][0]} x {CONVERGENCE_MESHES[-1][1]}",
                "source offset from accumulation onset (m)": source - fine_onset,
                "time step (yr)": 2.0,
                "age ceiling (yr)": boundary_map["age ceiling (yr)"],
                "minimum J": minimum,
                "age of minimum J (yr)": age,
                "last valid age (yr)": last_valid_age,
                "completed residence before age ceiling": (
                    last_valid_age < boundary_map["age ceiling (yr)"]
                ),
            }
        )

    root = Path(__file__).resolve().parents[1]
    survey.to_csv(root / "data" / "finite_spike_survey.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(
        root / "data" / "finite_spike_mesh_convergence.csv", index=False
    )
    pd.DataFrame(map_rows).to_csv(
        root / "data" / "finite_spike_map_convergence.csv", index=False
    )
    pd.DataFrame(domain_rows).to_csv(
        root / "data" / "finite_spike_domain_convergence.csv", index=False
    )
    pd.DataFrame(boundary_rows).to_csv(
        root / "data" / "finite_spike_accumulation_boundary.csv", index=False
    )
    plot_best_cases(saved_cases, root / "figures" / "finite_spike_open_boundary.png")
    print("\nSurvey:\n", survey.to_string(index=False), flush=True)
    print("\nMesh convergence:\n", pd.DataFrame(convergence_rows).to_string(index=False), flush=True)
    print("\nMap convergence:\n", pd.DataFrame(map_rows).to_string(index=False), flush=True)
    print("\nDomain convergence:\n", pd.DataFrame(domain_rows).to_string(index=False), flush=True)
    print("\nAccumulation-boundary limit:\n", pd.DataFrame(boundary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
