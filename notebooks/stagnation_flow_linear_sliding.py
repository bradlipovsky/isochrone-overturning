"""Two-sided convergent glacier flow with passive linear basal sliding.

The glacier occupies an asymmetric, finite V-shaped catchment.  Its left
flank is short and steep and its right flank is longer and gentler.  Bed and
surface relief may differ through ``KAPPA = Delta s / Delta b``; the original
constant-thickness case has ``KAPPA=1``.  When the surface descends toward
the basin center, gravity drives two converging flows.  The surface mass
balance needed to hold the geometry fixed is diagnosed from the solved
velocity; central ablation removes the converging flux.

This script tests whether equal-age curves launched from the diagnosed
surface-accumulation zones develop a vertical tangent (dx/dx0 = 0) or an
overturned limb (dx/dx0 < 0).  Length is in m, time in yr, velocity in
m/yr, and stress in Pa.
"""

from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore", message="Unable to import recommended hash")
warnings.filterwarnings("ignore", message="The .*Function.at.* method")
logging.getLogger("tsfc").setLevel(logging.ERROR)

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator


DENSITY = 917.0
GRAVITY = 9.81
VISCOSITY = 2.5e8  # Pa yr, linear viscous ice
BASAL_DRAG = 1.0e5  # Pa yr m^-1; mobility C = 1 / BASAL_DRAG
NITSCHE_PENALTY = 3.0e4

LEFT_LENGTH = 1_000.0
RIGHT_LENGTH = 5_000.0
RELIEF = 150.0
THICKNESS = 200.0
# Ratio of surface relief to bed relief.  The represented boundaries are
#
#     b(x) = RELIEF f(x),
#     h(x) = THICKNESS + (1 - KAPPA) RELIEF (1/2 - f(x)),
#     s(x) = b(x) + h(x),
#
# where f=0 at the basin center, f=1 at both divides, and its exact domain
# mean is 1/2.  THICKNESS is therefore the mean vertical thickness for every
# KAPPA.  KAPPA=1 is the original constant-thickness geometry and KAPPA=0
# has a flat surface.
KAPPA = 1.0

OUTPUT_DIRECTORY = Path("notebooks/stagnation_flow_results")


def flank_shape(x):
    """Bed elevation above the central minimum."""
    x = np.asarray(x)
    left = RELIEF * np.sin(0.5 * np.pi * (-x) / LEFT_LENGTH) ** 2
    right = RELIEF * np.sin(0.5 * np.pi * x / RIGHT_LENGTH) ** 2
    return np.where(x <= 0.0, left, right)


def flank_slope(x):
    """Bed slope, the derivative of :func:`flank_shape`."""
    x = np.asarray(x)
    left = -(
        0.5
        * np.pi
        * RELIEF
        / LEFT_LENGTH
        * np.sin(np.pi * (-x) / LEFT_LENGTH)
    )
    right = (
        0.5
        * np.pi
        * RELIEF
        / RIGHT_LENGTH
        * np.sin(np.pi * x / RIGHT_LENGTH)
    )
    return np.where(x <= 0.0, left, right)


def surface_shape(x):
    """Surface elevation for the current bed--surface relief ratio."""
    return (
        THICKNESS
        + 0.5 * (1.0 - KAPPA) * RELIEF
        + KAPPA * flank_shape(x)
    )


def surface_slope(x):
    """Surface slope for the current bed--surface relief ratio."""
    return KAPPA * flank_slope(x)


def ice_thickness(x):
    """Local vertical ice thickness."""
    return surface_shape(x) - flank_shape(x)


def minimum_ice_thickness():
    """Exact minimum vertical thickness over the two-flank geometry."""
    return THICKNESS - 0.5 * abs(1.0 - KAPPA) * RELIEF


def maximum_ice_thickness():
    """Exact maximum vertical thickness over the two-flank geometry."""
    return THICKNESS + 0.5 * abs(1.0 - KAPPA) * RELIEF


def center_ice_thickness():
    """Vertical thickness at the basin center, where f=0."""
    return THICKNESS + 0.5 * (1.0 - KAPPA) * RELIEF


def divide_ice_thickness():
    """Vertical thickness at either divide, where f=1."""
    return THICKNESS - 0.5 * (1.0 - KAPPA) * RELIEF


def make_mesh(nx, nz):
    """Map a rectangle to an asymmetric basin with independent top relief."""
    minimum_thickness = minimum_ice_thickness()
    if minimum_thickness <= 0.0:
        raise ValueError(
            "nonpositive ice thickness: require "
            "THICKNESS - abs(1 - KAPPA) * RELIEF / 2 > 0; "
            f"got {minimum_thickness:g} m"
        )
    total_length = LEFT_LENGTH + RIGHT_LENGTH
    base_mesh = fd.RectangleMesh(nx, nz, total_length, THICKNESS)
    coordinate_space = fd.VectorFunctionSpace(base_mesh, "CG", 2)
    reference_x, reference_z = fd.SpatialCoordinate(base_mesh)
    # Put the catchment minimum exactly on a mesh column.  This matters when
    # LEFT_LENGTH / total_length is not an integer multiple of 1 / nx in the
    # parameter study.  The slight (< half-cell) redistribution of horizontal
    # spacing avoids interpolating the piecewise flank formula across its join.
    left_columns = max(
        1,
        min(nx - 1, int(round(nx * LEFT_LENGTH / total_length))),
    )
    reference_join = total_length * left_columns / nx
    x = fd.conditional(
        reference_x <= reference_join,
        -LEFT_LENGTH + reference_x * LEFT_LENGTH / reference_join,
        RIGHT_LENGTH
        * (reference_x - reference_join)
        / (total_length - reference_join),
    )
    left_relief = RELIEF * fd.sin(
        0.5 * np.pi * (-x) / LEFT_LENGTH
    ) ** 2
    right_relief = RELIEF * fd.sin(
        0.5 * np.pi * x / RIGHT_LENGTH
    ) ** 2
    bed = fd.conditional(x <= 0.0, left_relief, right_relief)
    surface = THICKNESS + 0.5 * (1.0 - KAPPA) * RELIEF + KAPPA * bed
    local_thickness = surface - bed
    coordinates = fd.Function(coordinate_space).interpolate(
        fd.as_vector((x, bed + reference_z * local_thickness / THICKNESS))
    )
    return fd.Mesh(coordinates)


def solve_stokes(nx, nz):
    """Solve incompressible linear Stokes flow with basal linear sliding."""
    mesh = make_mesh(nx, nz)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)
    u, p = fd.split(solution)
    v, q = fd.TestFunctions(mixed_space)
    body_force = fd.Constant((0.0, -DENSITY * GRAVITY))

    residual = (
        2.0
        * VISCOSITY
        * fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v)))
        - p * fd.div(v)
        - q * fd.div(u)
        - fd.dot(body_force, v)
    ) * fd.dx

    # Boundary 3 is the mapped bed.  Nitsche terms impose u.n = 0, and the
    # Robin term imposes (sigma n).t = -BASAL_DRAG u.t.
    normal = fd.FacetNormal(mesh)
    tangent = fd.as_vector((-normal[1], normal[0]))
    stress = 2.0 * VISCOSITY * fd.sym(fd.grad(u)) - p * fd.Identity(2)
    test_stress = (
        2.0 * VISCOSITY * fd.sym(fd.grad(v)) - q * fd.Identity(2)
    )
    normal_speed = fd.dot(u, normal)
    test_normal_speed = fd.dot(v, normal)
    nominal_cell_size = (LEFT_LENGTH + RIGHT_LENGTH) / nx
    residual += (
        BASAL_DRAG * fd.dot(u, tangent) * fd.dot(v, tangent)
        - fd.dot(stress * normal, normal) * test_normal_speed
        - fd.dot(test_stress * normal, normal) * normal_speed
        + NITSCHE_PENALTY
        * VISCOSITY
        / fd.Constant(nominal_cell_size)
        * normal_speed
        * test_normal_speed
    ) * fd.ds(3)

    # The vertical sides are ice divides: zero normal speed and natural
    # tangential traction.  The upper surface is traction free.
    divide_bcs = [
        fd.DirichletBC(mixed_space.sub(0).sub(0), fd.Constant(0.0), 1),
        fd.DirichletBC(mixed_space.sub(0).sub(0), fd.Constant(0.0), 2),
    ]
    fd.solve(
        residual == 0,
        solution,
        bcs=divide_bcs,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    velocity, pressure = solution.subfunctions

    bed_length = fd.assemble(fd.Constant(1.0) * fd.ds(3, domain=mesh))
    top_length = fd.assemble(fd.Constant(1.0) * fd.ds(4, domain=mesh))
    gravity_power = fd.assemble(fd.dot(body_force, velocity) * fd.dx)
    basal_dissipation = fd.assemble(
        BASAL_DRAG * fd.dot(velocity, tangent) ** 2 * fd.ds(3)
    )
    viscous_dissipation = fd.assemble(
        2.0
        * VISCOSITY
        * fd.inner(fd.sym(fd.grad(velocity)), fd.sym(fd.grad(velocity)))
        * fd.dx
    )
    diagnostics = {
        "nx": nx,
        "nz": nz,
        "cells": mesh.num_cells(),
        "gravity_power": gravity_power,
        "basal_dissipation": basal_dissipation,
        "viscous_dissipation": viscous_dissipation,
        "relative_energy_residual": (
            gravity_power - basal_dissipation - viscous_dissipation
        )
        / gravity_power,
        "rms_basal_normal_speed": np.sqrt(
            fd.assemble(fd.dot(velocity, normal) ** 2 * fd.ds(3)) / bed_length
        ),
        "rms_surface_normal_velocity": np.sqrt(
            fd.assemble(fd.dot(velocity, normal) ** 2 * fd.ds(4)) / top_length
        ),
        "minimum_nodal_u": np.min(velocity.dat.data_ro[:, 0]),
        "maximum_nodal_u": np.max(velocity.dat.data_ro[:, 0]),
        "minimum_nodal_w": np.min(velocity.dat.data_ro[:, 1]),
        "maximum_nodal_w": np.max(velocity.dat.data_ro[:, 1]),
    }
    return mesh, velocity, diagnostics


def make_sampler(
    velocity,
    nx_sample=1201,
    nz_sample=101,
    sigma_lower=0.002,
    sigma_upper=0.999,
):
    """Sample velocity on a regular grid fitted to the represented boundary."""
    mesh = velocity.function_space().mesh()
    coordinates = mesh.coordinates.dat.data_ro
    zero = fd.Constant((0.0, 0.0))
    bed_nodes = fd.DirichletBC(velocity.function_space(), zero, 3).nodes
    surface_nodes = fd.DirichletBC(velocity.function_space(), zero, 4).nodes

    def boundary_data(nodes):
        order = np.argsort(coordinates[nodes, 0])
        xy = coordinates[nodes][order]
        uv = velocity.dat.data_ro[nodes][order]
        keep = np.r_[True, np.diff(xy[:, 0]) > 1.0e-10]
        return xy[keep], uv[keep]

    bed_xy, _ = boundary_data(bed_nodes)
    surface_xy, surface_uv = boundary_data(surface_nodes)
    bed_interpolator = PchipInterpolator(bed_xy[:, 0], bed_xy[:, 1])
    bed_slope_interpolator = bed_interpolator.derivative()
    surface_interpolator = PchipInterpolator(
        surface_xy[:, 0], surface_xy[:, 1]
    )
    surface_slope_interpolator = surface_interpolator.derivative()
    surface_u_interpolator = PchipInterpolator(
        surface_xy[:, 0], surface_uv[:, 0]
    )
    surface_w_interpolator = PchipInterpolator(
        surface_xy[:, 0], surface_uv[:, 1]
    )

    horizontal_margin = 0.5
    x = np.linspace(
        -LEFT_LENGTH + horizontal_margin,
        RIGHT_LENGTH - horizontal_margin,
        nx_sample,
    )
    if not 0.0 < sigma_lower < sigma_upper < 1.0:
        raise ValueError("require 0 < sigma_lower < sigma_upper < 1")
    sigma = np.linspace(sigma_lower, sigma_upper, nz_sample)
    bed = bed_interpolator(x)
    surface = surface_interpolator(x)
    X = np.tile(x, (nz_sample, 1))
    Z = bed[None, :] + (surface - bed)[None, :] * sigma[:, None]
    points = np.column_stack((X.ravel(), Z.ravel()))
    values = np.asarray(
        velocity.at(points, tolerance=1.0e-7)
    ).reshape(Z.shape + (2,))
    interpolators = {
        name: RegularGridInterpolator(
            (sigma, x),
            values[..., component],
            bounds_error=False,
            # Permit a small RK stage to cross the sampled boundary.  Markers
            # are removed immediately after the completed step if they have
            # reached the surface, bed, or a divide.
            fill_value=None,
        )
        for component, name in enumerate(("u", "w"))
    }
    # Derivatives in the regular (sigma, x) coordinates.  These are converted
    # to physical (x, z) derivatives in sample_gradient below and drive the
    # variational material-map equation dF/dt = grad(u) F.  This avoids taking
    # finite differences between the last few surviving markers at an exit.
    values_dsigma = np.gradient(values, sigma, axis=0, edge_order=2)
    values_dx_sigma = np.gradient(values, x, axis=1, edge_order=2)
    derivative_interpolators = {}
    for component, name in enumerate(("u", "w")):
        derivative_interpolators[(name, "sigma")] = RegularGridInterpolator(
            (sigma, x),
            values_dsigma[..., component],
            bounds_error=False,
            fill_value=None,
        )
        derivative_interpolators[(name, "x_sigma")] = RegularGridInterpolator(
            (sigma, x),
            values_dx_sigma[..., component],
            bounds_error=False,
            fill_value=None,
        )

    def sample(points):
        points = np.atleast_2d(points)
        local_bed = bed_interpolator(points[:, 0])
        local_surface = surface_interpolator(points[:, 0])
        local_sigma = (points[:, 1] - local_bed) / (
            local_surface - local_bed
        )
        coordinates = np.column_stack((local_sigma, points[:, 0]))
        return np.column_stack(
            (interpolators["u"](coordinates), interpolators["w"](coordinates))
        )

    def sample_gradient(points):
        """Velocity gradient in physical coordinates at marker positions."""
        points = np.atleast_2d(points)
        local_bed = bed_interpolator(points[:, 0])
        local_surface = surface_interpolator(points[:, 0])
        local_thickness = local_surface - local_bed
        local_sigma = (points[:, 1] - local_bed) / local_thickness
        coordinates = np.column_stack((local_sigma, points[:, 0]))
        bed_slope = bed_slope_interpolator(points[:, 0])
        surface_slope = surface_slope_interpolator(points[:, 0])
        sigma_x_at_fixed_z = -(
            bed_slope + local_sigma * (surface_slope - bed_slope)
        ) / local_thickness
        sigma_z = 1.0 / local_thickness
        gradient = np.empty((len(points), 2, 2))
        for component, name in enumerate(("u", "w")):
            derivative_sigma = derivative_interpolators[
                (name, "sigma")
            ](coordinates)
            derivative_x_sigma = derivative_interpolators[
                (name, "x_sigma")
            ](coordinates)
            gradient[:, component, 0] = (
                derivative_x_sigma + derivative_sigma * sigma_x_at_fixed_z
            )
            gradient[:, component, 1] = derivative_sigma * sigma_z
        return gradient

    geometry = {
        "bed": bed_interpolator,
        "surface": surface_interpolator,
        "bed_slope": bed_slope_interpolator,
        "surface_slope": surface_slope_interpolator,
        "velocity_gradient": sample_gradient,
        "surface_velocity": lambda x_value: np.column_stack(
            (
                surface_u_interpolator(np.atleast_1d(x_value)),
                surface_w_interpolator(np.atleast_1d(x_value)),
            )
        ),
        "surface_x": surface_xy[:, 0],
    }
    return sample, geometry, x, sigma, X, Z, values


def connected_intervals(mask):
    """Return inclusive index intervals for contiguous true samples."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [(group[0], group[-1]) for group in groups]


def trace_source_family(
    sample, geometry, source_x, maximum_age=120_000.0, time_step=50.0
):
    """Trace one connected surface source and its tangent sensitivity."""
    source_x = np.asarray(source_x)
    source_bed = geometry["bed"](source_x)
    source_surface = geometry["surface"](source_x)
    source_points = np.column_stack(
        (source_x, source_bed + 0.995 * (source_surface - source_bed))
    )
    source_velocity = geometry["surface_velocity"](source_x)
    source_accumulation = (
        source_velocity[:, 0] * geometry["surface_slope"](source_x)
        - source_velocity[:, 1]
    )
    source_surface_slope = geometry["surface_slope"](source_x)
    source_tangent = np.column_stack(
        (
            np.ones(len(source_x)),
            source_surface_slope,
        )
    )

    # Fixed-step vector RK4 is much faster here than integrating hundreds of
    # marker coordinates as one adaptive system.  The 50 yr step is at least
    # twenty times shorter than the local H/U advection time.
    current = source_points.copy()
    current_tangent = source_tangent.copy()
    active = np.ones(len(source_x), dtype=bool)
    position_history = [current.copy()]
    tangent_history = [current_tangent.copy()]
    active_history = [active.copy()]
    time_history = [0.0]
    for step in range(int(np.ceil(maximum_age / time_step))):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        points = current[indices]
        tangents = current_tangent[indices]
        k1 = sample(points)
        f1 = np.einsum(
            "nij,nj->ni", geometry["velocity_gradient"](points), tangents
        )
        points_2 = points + 0.5 * time_step * k1
        tangents_2 = tangents + 0.5 * time_step * f1
        k2 = sample(points_2)
        f2 = np.einsum(
            "nij,nj->ni",
            geometry["velocity_gradient"](points_2),
            tangents_2,
        )
        points_3 = points + 0.5 * time_step * k2
        tangents_3 = tangents + 0.5 * time_step * f2
        k3 = sample(points_3)
        f3 = np.einsum(
            "nij,nj->ni",
            geometry["velocity_gradient"](points_3),
            tangents_3,
        )
        points_4 = points + time_step * k3
        tangents_4 = tangents + time_step * f3
        k4 = sample(points_4)
        f4 = np.einsum(
            "nij,nj->ni",
            geometry["velocity_gradient"](points_4),
            tangents_4,
        )
        candidate = current.copy()
        candidate[indices] = points + time_step * (
            k1 + 2 * k2 + 2 * k3 + k4
        ) / 6
        candidate_tangent = current_tangent.copy()
        candidate_tangent[indices] = tangents + time_step * (
            f1 + 2 * f2 + 2 * f3 + f4
        ) / 6
        current = candidate
        current_tangent = candidate_tangent
        age = (step + 1) * time_step
        local_bed = geometry["bed"](current[indices, 0])
        local_surface = geometry["surface"](current[indices, 0])
        local_sigma = (current[indices, 1] - local_bed) / (
            local_surface - local_bed
        )
        remains = (
            np.all(np.isfinite(current[indices]), axis=1)
            & (local_sigma > 0.0022)
            & (local_sigma < 0.998)
            & (current[indices, 0] > -LEFT_LENGTH + 0.5)
            & (current[indices, 0] < RIGHT_LENGTH - 0.5)
        )
        active[indices[~remains]] = False
        recorded = current.copy()
        recorded[~active] = np.nan
        recorded_tangent = current_tangent.copy()
        recorded_tangent[~active] = np.nan
        position_history.append(recorded)
        tangent_history.append(recorded_tangent)
        active_history.append(active.copy())
        time_history.append(age)

    raw_times = np.asarray(time_history)
    raw_positions = np.asarray(position_history)
    output_times = raw_times
    positions = raw_positions
    tangents = np.asarray(tangent_history)
    active_history = np.asarray(active_history)
    jacobian_x = tangents[:, :, 0]
    jacobian_z = tangents[:, :, 1]
    neighbor_jacobian_x = np.full(active_history.shape, np.nan)
    speed = np.full(active_history.shape, np.nan)
    for time_index, is_active in enumerate(active_history):
        indices = np.flatnonzero(is_active)
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        for group in groups:
            if len(group) >= 5:
                neighbor_jacobian_x[time_index, group] = np.gradient(
                    positions[time_index, group, 0], source_x[group]
                )
                # Keep a neighbor-difference check away from the moving
                # endpoints, but use the variational tangent above for the
                # actual overturn diagnostic.
                neighbor_jacobian_x[time_index, group[:2]] = np.nan
                neighbor_jacobian_x[time_index, group[-2:]] = np.nan
        if len(indices):
            speed[time_index, indices] = np.linalg.norm(
                sample(positions[time_index, indices]), axis=1
            )
    local_bed = geometry["bed"](positions[:, :, 0])
    local_surface = geometry["surface"](positions[:, :, 0])
    sigma = (positions[:, :, 1] - local_bed) / (
        local_surface - local_bed
    )
    return {
        "source_x": source_x,
        "source_accumulation": source_accumulation,
        "source_velocity": source_velocity,
        "reached_maximum_age": output_times[-1] >= maximum_age,
        "times": output_times,
        "positions": positions,
        "tangents": tangents,
        "Jx": jacobian_x,
        "Jz": jacobian_z,
        "Jx_neighbor": neighbor_jacobian_x,
        "speed": speed,
        "sigma": sigma,
        "active": active_history,
    }


def diagnose_resolution(nx, nz, trace=True):
    """Solve one mesh and optionally trace all connected accumulation zones."""
    print(f"Solving {nx} by {nz} mesh", flush=True)
    mesh, velocity, diagnostics = solve_stokes(nx, nz)
    sample, geometry, x, sigma, X, Z, values = make_sampler(velocity)
    surface_velocity = geometry["surface_velocity"](x)
    accumulation = (
        surface_velocity[:, 0] * geometry["surface_slope"](x)
        - surface_velocity[:, 1]
    )
    intervals = connected_intervals(accumulation > 1.0e-5)
    diagnostics.update(
        {
            "minimum_surface_mass_balance": np.min(accumulation),
            "maximum_surface_mass_balance": np.max(accumulation),
            "integrated_positive_mass_balance": np.trapz(
                np.maximum(accumulation, 0.0), x
            ),
            "integrated_negative_mass_balance": np.trapz(
                np.minimum(accumulation, 0.0), x
            ),
            "accumulation_intervals": len(intervals),
        }
    )

    families = []
    if trace:
        for family_id, (start, stop) in enumerate(intervals):
            if stop - start < 10:
                continue
            interval = np.arange(start, stop + 1)
            flux = np.maximum(accumulation[interval], 0.0)
            cumulative = np.cumsum(flux)
            cumulative /= cumulative[-1]
            lower = interval[np.searchsorted(cumulative, 0.05)]
            upper = interval[np.searchsorted(cumulative, 0.95)]
            source_x = np.linspace(x[lower], x[upper], 81)
            print(
                f"  tracing source {family_id}: {source_x[0]:.1f} to "
                f"{source_x[-1]:.1f} m",
                flush=True,
            )
            family = trace_source_family(sample, geometry, source_x)
            family["family_id"] = family_id
            family["name"] = (
                "left_steep" if np.mean(source_x) < 0.0 else "right_gentle"
            )
            families.append(family)
            minimum_flat = int(np.nanargmin(family["Jx"]))
            minimum_time_index, minimum_source_index = np.unravel_index(
                minimum_flat, family["Jx"].shape
            )
            prefix = family["name"]
            diagnostics.update(
                {
                    f"{prefix}_minimum_Jx": family["Jx"]
                    [minimum_time_index, minimum_source_index],
                    f"{prefix}_age_at_minimum_Jx": family["times"]
                    [minimum_time_index],
                    f"{prefix}_source_x_at_minimum_Jx": source_x[
                        minimum_source_index
                    ],
                    f"{prefix}_active_at_minimum_Jx": int(
                        family["active"][minimum_time_index].sum()
                    ),
                    f"{prefix}_final_age": family["times"][-1],
                }
            )

        if families:
            diagnostics.update(
                {
                    "minimum_Jx": min(
                        np.nanmin(family["Jx"]) for family in families
                    ),
                    "minimum_speed": min(
                        np.nanmin(family["speed"]) for family in families
                    ),
                    "maximum_age_before_exit": max(
                        family["times"][-1] for family in families
                    ),
                    "overturned_samples": sum(
                        np.count_nonzero(family["Jx"] < 0.0)
                        for family in families
                    ),
                }
            )
    return {
        "mesh": mesh,
        "velocity": velocity,
        "diagnostics": diagnostics,
        "sample": sample,
        "geometry": geometry,
        "x": x,
        "sigma": sigma,
        "X": X,
        "Z": Z,
        "values": values,
        "surface_mass_balance": accumulation,
        "families": families,
    }


def make_figure(result):
    """Plot geometry, velocity, mass balance, and oldest material curves."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    x = result["x"]
    values = result["values"]
    speed = np.linalg.norm(values, axis=2)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), constrained_layout=True)

    color = axes[0].contourf(
        result["X"] / 1000.0,
        result["Z"],
        speed,
        levels=20,
        cmap="viridis",
    )
    # A sparse direction field conveys convergence without obscuring the
    # speed field.  Normalize the arrows because speed is already encoded by
    # the background color.
    x_indices = np.linspace(70, len(x) - 71, 9, dtype=int)
    z_indices = np.array((12, 32, 52, 72, 92))
    arrow_values = values[np.ix_(z_indices, x_indices)]
    arrow_speed = np.linalg.norm(arrow_values, axis=2)
    arrow_direction = arrow_values / np.maximum(arrow_speed[..., None], 1e-12)
    axes[0].quiver(
        result["X"][np.ix_(z_indices, x_indices)] / 1000.0,
        result["Z"][np.ix_(z_indices, x_indices)],
        arrow_direction[..., 0],
        arrow_direction[..., 1],
        color="0.08",
        edgecolor="white",
        linewidth=0.35,
        angles="uv",
        scale=20.0,
        width=0.0032,
        headwidth=3.5,
        zorder=6,
    )
    axes[0].fill_between(
        x / 1000.0, 0.0, flank_shape(x), color="0.45", zorder=5
    )
    axes[0].plot(
        x / 1000.0,
        flank_shape(x),
        color="0.1",
        linewidth=1.0,
        zorder=7,
    )
    axes[0].plot(
        x / 1000.0,
        surface_shape(x),
        color="0.1",
        linewidth=1.2,
        zorder=7,
    )
    fig.colorbar(color, ax=axes[0], label="speed (m yr$^{-1}$)")
    axes[0].set(
        ylabel="elevation (m)",
        title="(a) Finite-sliding convergent flow",
    )

    axes[1].plot(x / 1000.0, result["surface_mass_balance"], color="black")
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].fill_between(
        x / 1000.0,
        0.0,
        result["surface_mass_balance"],
        where=result["surface_mass_balance"] > 0.0,
        color="#4477AA",
        alpha=0.35,
        label="accumulation/source",
    )
    axes[1].fill_between(
        x / 1000.0,
        0.0,
        result["surface_mass_balance"],
        where=result["surface_mass_balance"] < 0.0,
        color="#CC6677",
        alpha=0.35,
        label="ablation/sink",
    )
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set(
        ylabel="surface mass balance (m yr$^{-1}$)",
        title="(b) Diagnosed balance required for a fixed surface",
    )

    axes[2].fill_between(x / 1000.0, 0.0, flank_shape(x), color="0.6")
    axes[2].plot(x / 1000.0, surface_shape(x), color="0.1", linewidth=1.0)
    colors = ("#4477AA", "#EE7733")
    for family, family_color in zip(result["families"], colors):
        times = np.linspace(0.0, family["times"][-1], 7)
        for time in times:
            index = np.argmin(np.abs(family["times"] - time))
            position = family["positions"][index]
            axes[2].plot(
                position[:, 0] / 1000.0,
                position[:, 1],
                color=family_color,
                alpha=0.3 + 0.7 * time / max(times[-1], 1.0),
                linewidth=1.1,
            )
        oldest = family["positions"][-1]
        overturned = family["Jx"][-1] < 0.0
        axes[2].scatter(
            oldest[overturned, 0] / 1000.0,
            oldest[overturned, 1],
            facecolors="none",
            edgecolors="red",
            s=18,
            zorder=8,
        )
    if result["families"]:
        axes[2].plot([], [], color=colors[0], label="left-flank source")
        axes[2].plot([], [], color=colors[1], label="right-flank source")
        axes[2].legend(frameon=False, loc="lower right", ncol=2)
    axes[2].set(
        xlabel="horizontal coordinate (km)",
        ylabel="elevation (m)",
        title="(c) Equal-age curves with markers removed at surface exit",
    )
    for axis in axes:
        axis.set_xlim(-LEFT_LENGTH / 1000.0, RIGHT_LENGTH / 1000.0)
    figure_path = OUTPUT_DIRECTORY / "stagnation_flow_summary.png"
    fig.savefig(figure_path, dpi=220)
    fig.savefig(OUTPUT_DIRECTORY / "stagnation_flow_summary.pdf")
    plt.close(fig)
    return figure_path


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    meshes = [(120, 12), (180, 18), (240, 24)]
    rows = []
    finest = None
    for nx, nz in meshes:
        result = diagnose_resolution(nx, nz, trace=True)
        rows.append(result["diagnostics"])
        finest = result
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIRECTORY / "stagnation_flow_convergence.csv", index=False)
    material_map_rows = []
    configurations = (
        ("source_count", 41, 25.0),
        ("source_count", 81, 25.0),
        ("source_count", 161, 25.0),
        ("time_step", 161, 100.0),
        ("time_step", 161, 50.0),
        ("time_step", 161, 25.0),
    )
    for study, source_count, time_step in configurations:
        for original in finest["families"]:
            if source_count == 81 and time_step == 50.0:
                family = original
            else:
                repeated_sources = np.linspace(
                    original["source_x"][0],
                    original["source_x"][-1],
                    source_count,
                )
                family = trace_source_family(
                    finest["sample"],
                    finest["geometry"],
                    repeated_sources,
                    time_step=time_step,
                )
            if original["name"] == "left_steep":
                reference_age = 3000.0
                reference_source_x = -800.0
            else:
                reference_age = 3500.0
                reference_source_x = 4306.990438
            reference_time_index = np.argmin(
                np.abs(family["times"] - reference_age)
            )
            reference_source_index = np.argmin(
                np.abs(family["source_x"] - reference_source_x)
            )
            material_map_rows.append(
                {
                    "study": study,
                    "source": original["name"],
                    "source_count": source_count,
                    "time_step_yr": time_step,
                    "minimum_Jx": np.nanmin(family["Jx"]),
                    "age_at_minimum_Jx_yr": family["times"][
                        np.unravel_index(
                            int(np.nanargmin(family["Jx"])),
                            family["Jx"].shape,
                        )[0]
                    ],
                    "minimum_Jx_at_3500_yr": np.nanmin(
                        family["Jx"][
                            np.argmin(np.abs(family["times"] - 3500.0))
                        ]
                    ),
                    "reference_age_yr": reference_age,
                    "reference_source_x_m": reference_source_x,
                    "Jx_at_reference": family["Jx"][
                        reference_time_index, reference_source_index
                    ],
                    "final_age_yr": family["times"][-1],
                    "overturned_samples": np.count_nonzero(
                        family["Jx"] < 0.0
                    ),
                }
            )
    material_map_table = pd.DataFrame(material_map_rows)
    material_map_table.to_csv(
        OUTPUT_DIRECTORY / "stagnation_flow_material_map_convergence.csv",
        index=False,
    )
    figure_path = make_figure(finest)
    print("\nMesh convergence and material-map diagnostic")
    print(table.to_string(index=False))
    print("\nIndependent source-count and time-step tests on the finest mesh")
    print(material_map_table.to_string(index=False))
    print(f"\nSaved {figure_path}")


if __name__ == "__main__":
    main()
