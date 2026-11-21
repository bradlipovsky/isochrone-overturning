"""Steady-characteristic overturn diagnostic for one sinusoidal bed period.

The glacier occupies one exactly periodic wavelength, ``0 <= x < lambda``,
with a flat surface at ``z = 0`` and bed

    b(x) = -H + A cos(2 pi x / lambda).

Thus ``H`` is mean ice thickness and ``A`` is bed-elevation amplitude (the
peak-to-trough relief is ``2 A``).  A constant downslope component of gravity
drives incompressible linear-viscous Stokes flow.  The impermeable bed obeys
the same passive linear sliding law used elsewhere in this repository,
``tau_t + beta u_t = 0``.

For each depositing surface source, a direct steady characteristic carries
the equal-age tangent

    dX/dt = v(X),       dT/dt = grad(v)(X) T,       T(0) = (1, 0).

The order diagnostic is ``J = T_x``.  Horizontal positions are left
unwrapped while all velocity queries are wrapped into the one-period Stokes
cell.  Each path ends independently on its first bed or surface exit; the
integration age is a characteristic coordinate, not a time-dependent glacier
simulation.

This file is deliberately importable: the parameter-map driver can reuse the
geometry, Stokes solve, sampler, source selection, and characteristic tracer.
Running it directly solves either one case or a small four-case smoke grid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter
import warnings

warnings.filterwarnings("ignore", message="Unable to import recommended hash")
warnings.filterwarnings("ignore", message="The .*Function.at.* method")
logging.getLogger("tsfc").setLevel(logging.ERROR)

import firedrake as fd
import numpy as np
import pandas as pd


DENSITY = 917.0  # kg m^-3
GRAVITY = 9.81  # m s^-2
VISCOSITY = 2.5e8  # Pa yr
SURFACE_SLOPE = 0.05  # rad
BASAL_DRAG = 1.0e5  # Pa yr m^-1; C = 1 / beta = 10^-5 m yr^-1 Pa^-1
# Match the convergent- and elliptical-flow calculations on PR40.  This is a
# numerical normal-velocity penalty, not the basal sliding coefficient.
NITSCHE_PENALTY = 3.0e4

BED_SIGMA = 0.005
SURFACE_SIGMA = 1.0
ANALYSIS_SURFACE_SIGMA = 0.95
# Characteristics are born exactly on the accumulation surface.  A fixed
# dimensional offset is not asymptotically safe: as A -> 0 the
# surface-connected streamtube becomes arbitrarily shallow, and even a 5 cm
# offset can place a nominal source on a periodic interior streamline.
START_DEPTH = 0.0

DEFAULT_TIME_STEP = 10.0  # yr
DEFAULT_MAXIMUM_AGE = 120_000.0  # yr, numerical safety ceiling only
DEFAULT_SOURCE_COUNT = 241
DEFAULT_RETAINED_FLUX_FRACTION = 0.80
DEFAULT_SAMPLER_NX = 481
DEFAULT_SAMPLER_NZ = 161

# Written into every row so restartable sweeps can never mix results produced
# by materially different characteristic/source conventions.
DIAGNOSTIC_METHOD_VERSION = (
    "sinusoidal-steady-v2-exact-surface-ablation-only-buffer"
)

SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


@dataclass(frozen=True)
class SinusoidalGeometry:
    """Dimensional geometry for one periodic sinusoidal bed wavelength.

    ``phase_origin`` is a coordinate gauge for translation-invariance audits;
    it is fixed to zero in the three-parameter production sweep.
    """

    wavelength: float
    amplitude: float
    mean_thickness: float
    phase_origin: float = 0.0

    def __post_init__(self):
        values = (
            self.wavelength,
            self.amplitude,
            self.mean_thickness,
            self.phase_origin,
        )
        if not all(np.isfinite(values)):
            raise ValueError("geometry values must be finite")
        if self.wavelength <= 0.0:
            raise ValueError("wavelength must be positive")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be nonnegative")
        if self.mean_thickness <= self.amplitude:
            raise ValueError(
                "mean thickness must exceed amplitude so thickness stays positive"
            )

    @property
    def minimum_thickness(self):
        return self.mean_thickness - self.amplitude

    @property
    def maximum_thickness(self):
        return self.mean_thickness + self.amplitude

    @property
    def peak_bed_slope(self):
        return 2.0 * np.pi * self.amplitude / self.wavelength

    @property
    def amplitude_ratio(self):
        return self.amplitude / self.mean_thickness

    @property
    def wavelength_ratio(self):
        return self.wavelength / self.mean_thickness


def wrap_x(x, geometry):
    """Wrap horizontal coordinates into the one-wavelength model cell."""
    return np.mod(np.asarray(x, dtype=float), geometry.wavelength)


def thickness(x, geometry):
    """Analytical ice thickness, whose spatial mean is exactly ``H``.

    Geometry objects with a ``thickness`` method may supply another smooth
    periodic profile.  The fallback preserves the original one-harmonic bed.
    """
    if hasattr(geometry, "thickness"):
        return np.asarray(geometry.thickness(x), dtype=float)
    phase = (
        2.0
        * np.pi
        * wrap_x(np.asarray(x) - geometry.phase_origin, geometry)
        / geometry.wavelength
    )
    return geometry.mean_thickness - geometry.amplitude * np.cos(phase)


def thickness_slope(x, geometry):
    """Analytical derivative of ice thickness with respect to x."""
    if hasattr(geometry, "thickness_slope"):
        return np.asarray(geometry.thickness_slope(x), dtype=float)
    phase = (
        2.0
        * np.pi
        * wrap_x(np.asarray(x) - geometry.phase_origin, geometry)
        / geometry.wavelength
    )
    return (
        2.0
        * np.pi
        * geometry.amplitude
        / geometry.wavelength
        * np.sin(phase)
    )


def bed_height(x, geometry):
    """Analytical sinusoidal bed elevation below the flat surface."""
    if hasattr(geometry, "bed_height"):
        return np.asarray(geometry.bed_height(x), dtype=float)
    return -thickness(x, geometry)


def represented_thickness(x, geometry, mesh_nx):
    """Periodic piecewise-linear thickness represented by the Stokes mesh."""
    node_x = np.linspace(0.0, geometry.wavelength, mesh_nx, endpoint=False)
    node_h = thickness(node_x, geometry)
    return np.interp(
        wrap_x(x, geometry),
        np.r_[node_x, geometry.wavelength],
        np.r_[node_h, node_h[0]],
    )


def represented_thickness_slope(x, geometry, mesh_nx):
    """Derivative of the mesh's periodic piecewise-linear thickness."""
    dx = geometry.wavelength / mesh_nx
    wrapped = wrap_x(x, geometry)
    left = np.floor(wrapped / dx).astype(int) % mesh_nx
    right = (left + 1) % mesh_nx
    node_x = np.arange(mesh_nx) * dx
    node_h = thickness(node_x, geometry)
    return (node_h[right] - node_h[left]) / dx


def choose_mesh_dimensions(geometry, target_spacing=10.0):
    """Choose a modest resolution while resolving every period uniformly."""
    if target_spacing <= 0.0:
        raise ValueError("target spacing must be positive")
    nx = max(48, int(np.ceil(geometry.wavelength / target_spacing)))
    nx = 4 * int(np.ceil(nx / 4.0))
    nz = max(12, int(np.ceil(geometry.maximum_thickness / target_spacing)))
    return nx, nz


def endpoint_clustered_coordinate(coordinate, power=1.0):
    """Map ``[0, 1]`` to itself while clustering both endpoints.

    ``power=1`` is exactly the identity.  Larger powers symmetrically refine
    the basal forcing layer and the shallow surface-connected streamtube,
    whose dimensional scales both become ``O(lambda / (2 pi))`` in the
    short-wave limit.
    """
    power = float(power)
    if not np.isfinite(power) or power < 1.0:
        raise ValueError("vertical grading power must be finite and >= 1")
    coordinate = np.asarray(coordinate, dtype=float)
    if power == 1.0:
        return coordinate.copy()
    return np.where(
        coordinate <= 0.5,
        0.5 * (2.0 * coordinate) ** power,
        1.0 - 0.5 * (2.0 * (1.0 - coordinate)) ** power,
    )


def make_mesh(geometry, nx, nz, vertical_grading_power=1.0):
    """Map a graded periodic rectangle into one sinusoidal-bed period."""
    if nx < 8 or nz < 4:
        raise ValueError("mesh needs at least 8 horizontal and 4 vertical cells")
    mesh = fd.PeriodicRectangleMesh(
        int(nx), int(nz), geometry.wavelength, 1.0, direction="x"
    )
    reference = mesh.coordinates.dat.data_ro.copy()
    local_thickness = thickness(reference[:, 0], geometry)
    sigma = endpoint_clustered_coordinate(
        reference[:, 1], vertical_grading_power
    )
    mesh.coordinates.dat.data[:, 1] = -local_thickness * (
        1.0 - sigma
    )
    mesh.clear_spatial_index()
    return mesh


def solve_stokes(geometry, nx, nz, vertical_grading_power=1.0):
    """Solve periodic Stokes flow with no penetration and linear basal drag."""
    mesh = make_mesh(
        geometry, nx, nz, vertical_grading_power=vertical_grading_power
    )
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
        2.0
        * VISCOSITY
        * fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v)))
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

    bed_length = fd.assemble(fd.Constant(1.0) * fd.ds(1, domain=mesh))
    surface_length = fd.assemble(fd.Constant(1.0) * fd.ds(2, domain=mesh))
    basal_speed = fd.dot(velocity, tangent)
    physical_stress = (
        2.0 * VISCOSITY * fd.sym(fd.grad(velocity))
        - pressure * fd.Identity(2)
    )
    gravity_power = fd.assemble(fd.dot(body_force, velocity) * fd.dx)
    basal_dissipation = fd.assemble(
        BASAL_DRAG * basal_speed**2 * fd.ds(1)
    )
    viscous_dissipation = fd.assemble(
        2.0
        * VISCOSITY
        * fd.inner(
            fd.sym(fd.grad(velocity)), fd.sym(fd.grad(velocity))
        )
        * fd.dx
    )
    pressure_constraint_work = fd.assemble(
        -2.0 * pressure * fd.div(velocity) * fd.dx
    )
    nitsche_normal_work = fd.assemble(
        (
            -2.0
            * fd.dot(physical_stress * normal, normal)
            * fd.dot(velocity, normal)
            + NITSCHE_PENALTY
            * VISCOSITY
            / fd.CellDiameter(mesh)
            * fd.dot(velocity, normal) ** 2
        )
        * fd.ds(1)
    )
    surface_vertical_flux = fd.assemble(velocity[1] * fd.ds(2))
    velocity_data = velocity.dat.data_ro
    diagnostics = {
        "mesh cells": int(mesh.num_cells()),
        "gravity power": float(gravity_power),
        "basal dissipation / gravity power": float(
            basal_dissipation / gravity_power
        ),
        # This deliberately omits the consistent Nitsche normal-boundary
        # terms.  It is a useful physical power-partition diagnostic, but it
        # is not the algebraic residual of the discrete solve.
        "physical strong-boundary energy imbalance": float(
            (gravity_power - basal_dissipation - viscous_dissipation)
            / gravity_power
        ),
        "relative discrete weak-form residual": float(
            (
                gravity_power
                - basal_dissipation
                - viscous_dissipation
                - pressure_constraint_work
                - nitsche_normal_work
            )
            / gravity_power
        ),
        "rms basal normal speed (m/yr)": float(
            np.sqrt(
                fd.assemble(fd.dot(velocity, normal) ** 2 * fd.ds(1))
                / bed_length
            )
        ),
        "surface vertical flux (m2/yr)": float(surface_vertical_flux),
        "mean surface vertical flux (m/yr)": float(
            surface_vertical_flux / surface_length
        ),
        "minimum nodal u (m/yr)": float(np.min(velocity_data[:, 0])),
        "maximum nodal u (m/yr)": float(np.max(velocity_data[:, 0])),
    }
    return mesh, velocity, diagnostics


def make_variational_sampler(
    geometry,
    mesh_nx,
    velocity,
    horizontal_samples=DEFAULT_SAMPLER_NX,
    vertical_samples=DEFAULT_SAMPLER_NZ,
    vertical_grading_power=1.0,
):
    """Sample velocity periodically and return its bilinear physical gradient.

    The terrain-following coordinate is ``sigma = 1 + z / h(x)``.  Values
    and derivatives come from one identical bilinear interpolant, avoiding a
    separate finite-difference gradient with a different numerical stencil.
    """
    if horizontal_samples < 8 or vertical_samples < 4:
        raise ValueError("sampler needs at least 8 x 4 points")
    dx = geometry.wavelength / horizontal_samples
    x_grid = (np.arange(horizontal_samples) + 0.5) * dx
    # Include the physical surface itself so both the source accumulation and
    # its initial velocity are sampled without a dimensional-depth surrogate.
    reference_sigma = np.linspace(0.0, 1.0, vertical_samples)
    sigma_grid = BED_SIGMA + (SURFACE_SIGMA - BED_SIGMA) * (
        endpoint_clustered_coordinate(
            reference_sigma, vertical_grading_power
        )
    )
    local_thickness = represented_thickness(x_grid, geometry, mesh_nx)
    X = np.tile(x_grid, (vertical_samples, 1))
    Z = -(1.0 - sigma_grid[:, None]) * local_thickness[None, :]
    points = np.column_stack((X.ravel(), Z.ravel()))
    values = np.asarray(
        velocity.at(points, tolerance=1.0e-7), dtype=float
    ).reshape(vertical_samples, horizontal_samples, 2)

    def interpolate(points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        wrapped = wrap_x(points[:, 0], geometry)
        local_h = represented_thickness(wrapped, geometry, mesh_nx)
        sigma = 1.0 + points[:, 1] / local_h

        horizontal_coordinate = wrapped / dx - 0.5
        ix_unwrapped = np.floor(horizontal_coordinate).astype(int)
        fx = horizontal_coordinate - ix_unwrapped
        ix0 = ix_unwrapped % horizontal_samples
        ix1 = (ix0 + 1) % horizontal_samples

        iz = np.searchsorted(sigma_grid, sigma, side="right") - 1
        iz = np.clip(iz, 0, vertical_samples - 2)
        ds = sigma_grid[iz + 1] - sigma_grid[iz]
        fs = (sigma - sigma_grid[iz]) / ds

        v00 = values[iz, ix0]
        v01 = values[iz, ix1]
        v10 = values[iz + 1, ix0]
        v11 = values[iz + 1, ix1]
        velocity_value = (
            ((1.0 - fs) * (1.0 - fx))[:, None] * v00
            + ((1.0 - fs) * fx)[:, None] * v01
            + (fs * (1.0 - fx))[:, None] * v10
            + (fs * fx)[:, None] * v11
        )
        derivative_sigma = (
            (1.0 - fx)[:, None] * (v10 - v00)
            + fx[:, None] * (v11 - v01)
        ) / ds[:, None]
        derivative_x_sigma = (
            (1.0 - fs)[:, None] * (v01 - v00)
            + fs[:, None] * (v11 - v10)
        ) / dx

        h_x = represented_thickness_slope(wrapped, geometry, mesh_nx)
        sigma_x = (1.0 - sigma) * h_x / local_h
        gradient = np.empty((len(points), 2, 2))
        gradient[:, :, 0] = (
            derivative_x_sigma + derivative_sigma * sigma_x[:, None]
        )
        gradient[:, :, 1] = derivative_sigma / local_h[:, None]
        return velocity_value, gradient

    def sample(name, x, z):
        x, z = np.broadcast_arrays(x, z)
        query = np.column_stack((x.ravel(), z.ravel()))
        component = 0 if name == "u" else 1
        result = interpolate(query)[0][:, component].reshape(x.shape)
        return float(result) if result.ndim == 0 else result

    def sample_vector(points):
        return interpolate(points)[0]

    def sample_gradient(points):
        return interpolate(points)[1]

    sampler_diagnostics = {
        "minimum sampled u (m/yr)": float(np.nanmin(values[..., 0])),
        "maximum sampled u (m/yr)": float(np.nanmax(values[..., 0])),
        "minimum sampled w (m/yr)": float(np.nanmin(values[..., 1])),
        "maximum sampled w (m/yr)": float(np.nanmax(values[..., 1])),
    }
    return sample, sample_vector, sample_gradient, sampler_diagnostics


def _cyclic_true_runs(mask):
    """Return connected true-index runs on a periodic one-dimensional grid."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []
    if np.all(mask):
        return [np.arange(len(mask))]
    anchor = int(np.flatnonzero(~mask)[0])
    order = (anchor + 1 + np.arange(len(mask))) % len(mask)
    true_positions = np.flatnonzero(mask[order])
    groups = np.split(
        true_positions, np.where(np.diff(true_positions) > 1)[0] + 1
    )
    return [order[group] for group in groups if len(group)]


def retained_surface_sources(
    geometry,
    sample,
    source_count=DEFAULT_SOURCE_COUNT,
    retained_flux_fraction=DEFAULT_RETAINED_FLUX_FRACTION,
    start_depth=START_DEPTH,
):
    """Find every depositing surface interval and trim its flux-weighted ends.

    Searching the full periodic surface avoids assuming which half of the bed
    supplies ice.  Within each connected accumulation interval, the retained
    sources span the central requested fraction of positive accumulation flux.
    This excludes the arbitrarily slow separatrix endpoints while retaining
    every distinct accumulation basin.
    """
    if not 0.0 < retained_flux_fraction <= 1.0:
        raise ValueError("retained flux fraction must be in (0, 1]")
    candidates = np.linspace(
        0.0, geometry.wavelength, source_count, endpoint=False
    )
    accumulation = -sample(
        "w", candidates, np.full(source_count, -start_depth)
    )
    tolerance = max(1.0e-10, 1.0e-8 * np.nanmax(np.abs(accumulation)))
    depositing_mask = np.isfinite(accumulation) & (accumulation > tolerance)
    runs = _cyclic_true_runs(depositing_mask)
    retained_indices = []
    tail = 0.5 * (1.0 - retained_flux_fraction)
    for run in runs:
        weights = accumulation[run]
        flux_coordinate = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
        selected = run[
            (flux_coordinate >= tail) & (flux_coordinate <= 1.0 - tail)
        ]
        if not len(selected):
            selected = np.array([run[int(np.argmax(weights))]])
        retained_indices.extend(selected.tolist())
    retained_indices = np.array(sorted(set(retained_indices)), dtype=int)
    depositing_indices = np.flatnonzero(depositing_mask)
    dx = geometry.wavelength / source_count
    net_flux = float(dx * np.sum(accumulation))
    gross_flux = float(dx * np.sum(np.abs(accumulation)))
    return {
        "retained sources": candidates[retained_indices],
        "depositing sources": candidates[depositing_indices],
        "candidate sources": candidates,
        "accumulation": accumulation,
        "accumulation interval count": len(runs),
        "accumulation tolerance": tolerance,
        "sampled net surface mass balance (m2/yr)": net_flux,
        "sampled gross surface mass exchange (m2/yr)": gross_flux,
        "sampled relative surface mass balance closure": (
            abs(net_flux) / gross_flux if gross_flux > 0.0 else 0.0
        ),
    }


def local_sigma(points, geometry, mesh_nx):
    """Terrain-following height of physical points, allowing unwrapped x."""
    local_h = represented_thickness(points[:, 0], geometry, mesh_nx)
    return 1.0 + points[:, 1] / local_h


def rk4_step(sample_vector, sample_gradient, points, tangents, step):
    """Advance positions and equal-age tangents by one vectorized RK4 step."""
    k1 = sample_vector(points)
    f1 = np.einsum("nij,nj->ni", sample_gradient(points), tangents)
    p2 = points + 0.5 * step * k1
    t2 = tangents + 0.5 * step * f1
    k2 = sample_vector(p2)
    f2 = np.einsum("nij,nj->ni", sample_gradient(p2), t2)
    p3 = points + 0.5 * step * k2
    t3 = tangents + 0.5 * step * f2
    k3 = sample_vector(p3)
    f3 = np.einsum("nij,nj->ni", sample_gradient(p3), t3)
    p4 = points + step * k3
    t4 = tangents + step * f3
    k4 = sample_vector(p4)
    f4 = np.einsum("nij,nj->ni", sample_gradient(p4), t4)
    return (
        points + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
        tangents + step * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0,
    )


def _vertical_exit_fraction(old_sigma, new_sigma):
    """Locate the first bed or surface boundary crossed during a step."""
    count = len(old_sigma)
    fraction = np.ones(count)
    kind = np.full(count, "numerical failure", dtype=object)
    candidates = (
        ("bed", BED_SIGMA, new_sigma <= BED_SIGMA),
        ("surface", SURFACE_SIGMA, new_sigma >= SURFACE_SIGMA),
    )
    for name, boundary, crossed in candidates:
        denominator = new_sigma - old_sigma
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate = (boundary - old_sigma) / denominator
        eligible = (
            crossed
            & np.isfinite(candidate)
            & (candidate >= 0.0)
            & (candidate <= fraction)
        )
        fraction[eligible] = candidate[eligible]
        kind[eligible] = name
    return fraction, kind


def trace_sources_to_exit(
    geometry,
    mesh_nx,
    sample_vector,
    sample_gradient,
    sources,
    time_step=DEFAULT_TIME_STEP,
    maximum_age=DEFAULT_MAXIMUM_AGE,
    start_depth=START_DEPTH,
    surface_accumulation_tolerance=0.0,
):
    """Carry periodic surface-source tangents independently to physical exit.

    With the production default ``start_depth=0``, the first Runge--Kutta
    step carries every depositing source inward from sigma=1.  A later outward
    crossing of sigma=1 is then its physical surface exit.  This construction
    remains well posed as the bed amplitude and accumulation both tend to
    zero.
    """
    if time_step <= 0.0 or maximum_age <= 0.0:
        raise ValueError("time step and maximum age must be positive")
    sources = np.asarray(sources, dtype=float)
    count = len(sources)
    points = np.column_stack((sources, np.full(count, -start_depth)))
    tangents = np.column_stack((np.ones(count), np.zeros(count)))
    active = np.ones(count, dtype=bool)
    age = np.zeros(count)
    raw_minimum_j = np.ones(count)
    raw_minimum_age = np.zeros(count)
    # Every retained source is depositing, so the source value J(0)=1 is an
    # eligible resolved sample even when the entire streamtube stays within
    # the upper five per cent of the ice column.
    buffered_minimum_j = np.ones(count)
    buffered_minimum_age = np.zeros(count)
    first_zero_age = np.full(count, np.nan)
    first_zero_surface_clearance = np.full(count, np.nan)
    maximum_negative_surface_clearance = np.full(count, np.nan)
    exit_age = np.full(count, np.nan)
    exit_kind = np.full(count, "active", dtype=object)
    source_velocity = sample_vector(points)
    source_u = source_velocity[:, 0]
    source_accumulation = -source_velocity[:, 1]
    minimum_u = source_u.copy()
    previous_u = source_u.copy()
    reversal_count = np.zeros(count, dtype=int)
    minimum_speed_ratio = np.ones(count)
    minimum_mechanism_c = np.zeros(count)
    raw_speed_ratio_at_minimum_j = np.ones(count)
    raw_mechanism_c_at_minimum_j = np.zeros(count)
    raw_reversal_before_minimum_j = np.zeros(count, dtype=bool)
    buffered_speed_ratio_at_minimum_j = np.ones(count)
    buffered_mechanism_c_at_minimum_j = np.zeros(count)
    buffered_reversal_before_minimum_j = np.zeros(count, dtype=bool)
    maximum_streamfunction_identity_residual = np.zeros(count)
    maximum_absolute_periods = np.zeros(count)
    maximum_step_periods = np.zeros(count)

    maximum_steps = int(np.ceil(maximum_age / time_step))
    for step_index in range(maximum_steps):
        step = min(time_step, maximum_age - step_index * time_step)
        if step <= 0.0:
            break
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        old_points = points[indices].copy()
        old_tangents = tangents[indices].copy()
        old_age = age[indices].copy()
        old_sigma = local_sigma(old_points, geometry, mesh_nx)
        candidate_points, candidate_tangents = rk4_step(
            sample_vector,
            sample_gradient,
            old_points,
            old_tangents,
            step,
        )
        candidate_sigma = local_sigma(
            candidate_points, geometry, mesh_nx
        )
        finite = np.all(np.isfinite(candidate_points), axis=1) & np.all(
            np.isfinite(candidate_tangents), axis=1
        )
        remains = (
            finite
            & (candidate_sigma > BED_SIGMA)
            & (candidate_sigma < SURFACE_SIGMA)
        )

        accepted_points = candidate_points.copy()
        accepted_tangents = candidate_tangents.copy()
        accepted_age = old_age + step
        step_exit_kind = np.full(len(indices), "", dtype=object)
        crossing = finite & ~remains
        if np.any(crossing):
            fraction, local_kind = _vertical_exit_fraction(
                old_sigma[crossing], candidate_sigma[crossing]
            )
            accepted_points[crossing] = old_points[crossing] + fraction[
                :, None
            ] * (candidate_points[crossing] - old_points[crossing])
            accepted_tangents[crossing] = old_tangents[crossing] + fraction[
                :, None
            ] * (candidate_tangents[crossing] - old_tangents[crossing])
            accepted_age[crossing] = old_age[crossing] + fraction * step
            step_exit_kind[crossing] = local_kind
        failed = ~finite
        if np.any(failed):
            accepted_points[failed] = old_points[failed]
            accepted_tangents[failed] = old_tangents[failed]
            accepted_age[failed] = old_age[failed]
            step_exit_kind[failed] = "numerical failure"

        old_j = old_tangents[:, 0]
        accepted_j = accepted_tangents[:, 0]
        accepted_sigma = local_sigma(accepted_points, geometry, mesh_nx)
        accepted_velocity = sample_vector(accepted_points)
        accepted_u = accepted_velocity[:, 0]
        speed_scale = np.maximum(
            np.maximum(np.abs(previous_u[indices]), np.abs(accepted_u)),
            1.0e-12,
        )
        reversal = (
            np.signbit(previous_u[indices]) != np.signbit(accepted_u)
        ) & (
            np.minimum(np.abs(previous_u[indices]), np.abs(accepted_u))
            > 1.0e-6 * speed_scale
        )
        reversals_through_sample = reversal_count[indices] + reversal.astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            speed_ratio = accepted_u / source_u[indices]
            mechanism_c = accepted_j / speed_ratio - 1.0
        mechanism_valid = (
            (reversals_through_sample == 0)
            & np.isfinite(speed_ratio)
            & np.isfinite(mechanism_c)
            & (speed_ratio >= 1.0e-3)
        )
        minimum_speed_ratio[indices] = np.fmin(
            minimum_speed_ratio[indices],
            np.where(mechanism_valid, speed_ratio, np.inf),
        )
        minimum_mechanism_c[indices] = np.fmin(
            minimum_mechanism_c[indices],
            np.where(mechanism_valid, mechanism_c, np.inf),
        )
        identity_residual = (
            -accepted_velocity[:, 1] * accepted_j
            + accepted_u * accepted_tangents[:, 1]
            - source_accumulation[indices]
        )
        identity_scale = np.maximum(
            np.abs(source_accumulation[indices]),
            max(float(surface_accumulation_tolerance), 1.0e-14),
        )
        maximum_streamfunction_identity_residual[indices] = np.fmax(
            maximum_streamfunction_identity_residual[indices],
            np.abs(identity_residual) / identity_scale,
        )
        crossed_zero = (
            np.isnan(first_zero_age[indices])
            & (old_j > 0.0)
            & (accepted_j <= 0.0)
        )
        if np.any(crossed_zero):
            fraction = old_j[crossed_zero] / (
                old_j[crossed_zero] - accepted_j[crossed_zero]
            )
            selected = indices[crossed_zero]
            first_zero_age[selected] = old_age[
                crossed_zero
            ] + fraction * (accepted_age[crossed_zero] - old_age[crossed_zero])
            zero_points = old_points[crossed_zero] + fraction[:, None] * (
                accepted_points[crossed_zero] - old_points[crossed_zero]
            )
            first_zero_surface_clearance[selected] = 1.0 - local_sigma(
                zero_points, geometry, mesh_nx
            )
        improves = accepted_j < raw_minimum_j[indices]
        if np.any(improves):
            selected = indices[improves]
            raw_minimum_j[selected] = accepted_j[improves]
            raw_minimum_age[selected] = accepted_age[improves]
            raw_speed_ratio_at_minimum_j[selected] = np.where(
                mechanism_valid[improves], speed_ratio[improves], np.nan
            )
            raw_mechanism_c_at_minimum_j[selected] = np.where(
                mechanism_valid[improves], mechanism_c[improves], np.nan
            )
            raw_reversal_before_minimum_j[selected] = (
                reversals_through_sample[improves] > 0
            )

        # The production diagnostic ignores the uppermost five per cent of
        # local thickness only beneath an ablating part of the surface.  The
        # accumulating side is retained all the way to its source, while the
        # final, arbitrarily thin return beneath surface ablation cannot set
        # the case minimum.  This also leaves shallow short-wave streamtubes
        # well defined instead of masking them merely because they never
        # descend below sigma=0.95.
        local_buffered_j = np.full(len(indices), np.inf)
        local_buffered_age = np.full(len(indices), np.nan)
        local_buffered_speed_ratio = np.full(len(indices), np.nan)
        local_buffered_mechanism_c = np.full(len(indices), np.nan)
        local_buffered_reversal = reversals_through_sample > 0
        accepted_surface_points = np.column_stack(
            (
                accepted_points[:, 0],
                np.full(len(accepted_points), -start_depth),
            )
        )
        accepted_surface_accumulation = -sample_vector(
            accepted_surface_points
        )[:, 1]
        resolved_sample = (accepted_sigma <= ANALYSIS_SURFACE_SIGMA) | (
            accepted_surface_accumulation
            >= -surface_accumulation_tolerance
        )
        local_buffered_j[resolved_sample] = accepted_j[resolved_sample]
        local_buffered_age[resolved_sample] = accepted_age[
            resolved_sample
        ]
        local_buffered_speed_ratio[resolved_sample] = np.where(
            mechanism_valid[resolved_sample], speed_ratio[resolved_sample], np.nan
        )
        local_buffered_mechanism_c[resolved_sample] = np.where(
            mechanism_valid[resolved_sample], mechanism_c[resolved_sample], np.nan
        )
        crosses_buffer = (
            (old_sigma - ANALYSIS_SURFACE_SIGMA)
            * (accepted_sigma - ANALYSIS_SURFACE_SIGMA)
            <= 0.0
        ) & (old_sigma != accepted_sigma)
        if np.any(crosses_buffer):
            fraction = (
                ANALYSIS_SURFACE_SIGMA - old_sigma[crosses_buffer]
            ) / (accepted_sigma[crosses_buffer] - old_sigma[crosses_buffer])
            boundary_j = old_j[crosses_buffer] + fraction * (
                accepted_j[crosses_buffer] - old_j[crosses_buffer]
            )
            boundary_age = old_age[crosses_buffer] + fraction * (
                accepted_age[crosses_buffer] - old_age[crosses_buffer]
            )
            boundary_points = old_points[crosses_buffer] + fraction[:, None] * (
                accepted_points[crosses_buffer] - old_points[crosses_buffer]
            )
            boundary_u = sample_vector(boundary_points)[:, 0]
            boundary_source_u = source_u[indices[crosses_buffer]]
            with np.errstate(divide="ignore", invalid="ignore"):
                boundary_speed_ratio = boundary_u / boundary_source_u
                boundary_mechanism_c = boundary_j / boundary_speed_ratio - 1.0
            boundary_reversal = reversals_through_sample[crosses_buffer] > 0
            boundary_valid = (
                ~boundary_reversal
                & np.isfinite(boundary_speed_ratio)
                & np.isfinite(boundary_mechanism_c)
                & (boundary_speed_ratio >= 1.0e-3)
            )
            target = np.flatnonzero(crosses_buffer)
            better_boundary = boundary_j < local_buffered_j[target]
            local_buffered_j[target[better_boundary]] = boundary_j[
                better_boundary
            ]
            local_buffered_age[target[better_boundary]] = boundary_age[
                better_boundary
            ]
            local_buffered_speed_ratio[target[better_boundary]] = np.where(
                boundary_valid[better_boundary],
                boundary_speed_ratio[better_boundary],
                np.nan,
            )
            local_buffered_mechanism_c[target[better_boundary]] = np.where(
                boundary_valid[better_boundary],
                boundary_mechanism_c[better_boundary],
                np.nan,
            )
            local_buffered_reversal[target[better_boundary]] = boundary_reversal[
                better_boundary
            ]
        improves_buffered = local_buffered_j < buffered_minimum_j[indices]
        if np.any(improves_buffered):
            selected = indices[improves_buffered]
            buffered_minimum_j[selected] = local_buffered_j[
                improves_buffered
            ]
            buffered_minimum_age[selected] = local_buffered_age[
                improves_buffered
            ]
            buffered_speed_ratio_at_minimum_j[selected] = (
                local_buffered_speed_ratio[improves_buffered]
            )
            buffered_mechanism_c_at_minimum_j[selected] = (
                local_buffered_mechanism_c[improves_buffered]
            )
            buffered_reversal_before_minimum_j[selected] = (
                local_buffered_reversal[improves_buffered]
            )

        negative = accepted_j < 0.0
        if np.any(negative):
            selected = indices[negative]
            clearance = 1.0 - accepted_sigma[negative]
            maximum_negative_surface_clearance[selected] = np.fmax(
                maximum_negative_surface_clearance[selected], clearance
            )

        maximum_step_periods[indices] = np.maximum(
            maximum_step_periods[indices],
            np.abs(accepted_points[:, 0] - old_points[:, 0])
            / geometry.wavelength,
        )
        minimum_u[indices] = np.minimum(minimum_u[indices], accepted_u)
        reversal_count[indices] += reversal.astype(int)
        previous_u[indices] = accepted_u
        maximum_absolute_periods[indices] = np.maximum(
            maximum_absolute_periods[indices],
            np.abs(accepted_points[:, 0] - sources[indices])
            / geometry.wavelength,
        )
        points[indices] = accepted_points
        tangents[indices] = accepted_tangents
        age[indices] = accepted_age

        exiting = ~remains
        if np.any(exiting):
            exited_indices = indices[exiting]
            active[exited_indices] = False
            exit_age[exited_indices] = accepted_age[exiting]
            exit_kind[exited_indices] = step_exit_kind[exiting]

    if np.any(active):
        exit_age[active] = age[active]
        exit_kind[active] = "safety cap"
    buffered_minimum_j[~np.isfinite(buffered_minimum_j)] = np.nan
    return pd.DataFrame(
        {
            "source x0 (m)": sources,
            "source phase": sources / geometry.wavelength,
            "minimum J outside ablating-surface buffer": buffered_minimum_j,
            "age of resolved minimum J (yr)": buffered_minimum_age,
            "raw minimum J to exit": raw_minimum_j,
            "age of raw minimum J (yr)": raw_minimum_age,
            "first J=0 age (yr)": first_zero_age,
            "surface-clearance fraction at first J=0": (
                first_zero_surface_clearance
            ),
            "maximum surface-clearance fraction while J<0": (
                maximum_negative_surface_clearance
            ),
            "minimum path u (m/yr)": minimum_u,
            "minimum path u / source u before reversal": minimum_speed_ratio,
            "minimum mechanism C before reversal": minimum_mechanism_c,
            "u / source u at minimum J": buffered_speed_ratio_at_minimum_j,
            "mechanism C at minimum J": buffered_mechanism_c_at_minimum_j,
            "reversal before minimum J": buffered_reversal_before_minimum_j,
            "u / source u at raw minimum J": raw_speed_ratio_at_minimum_j,
            "mechanism C at raw minimum J": raw_mechanism_c_at_minimum_j,
            "reversal before raw minimum J": raw_reversal_before_minimum_j,
            "maximum relative streamfunction identity residual": (
                maximum_streamfunction_identity_residual
            ),
            "horizontal velocity reversal count": reversal_count,
            "maximum absolute displacement (wavelengths)": (
                maximum_absolute_periods
            ),
            "maximum horizontal step (wavelengths)": maximum_step_periods,
            "exit age (yr)": exit_age,
            "exit kind": exit_kind,
            "exit phase": wrap_x(points[:, 0], geometry)
            / geometry.wavelength,
        }
    )


def classify(minimum_j):
    """Apply the same conservative numerical resolution band as prior maps."""
    if not np.isfinite(minimum_j):
        return "not evaluated"
    if minimum_j < -0.05:
        return "resolved overturn"
    if minimum_j > 0.05:
        return "resolved ordered"
    return "unresolved: |minimum J|<=0.05"


def solve_parameter_case(
    geometry,
    *,
    mesh_nx=None,
    mesh_nz=None,
    vertical_grading_power=1.0,
    target_spacing=10.0,
    sampler_nx=DEFAULT_SAMPLER_NX,
    sampler_nz=DEFAULT_SAMPLER_NZ,
    sampler_vertical_grading_power=1.0,
    source_count=DEFAULT_SOURCE_COUNT,
    retained_flux_fraction=DEFAULT_RETAINED_FLUX_FRACTION,
    time_step=DEFAULT_TIME_STEP,
    maximum_age=DEFAULT_MAXIMUM_AGE,
    minimum_surface_signal_to_basal_leakage=None,
):
    """Solve one geometry and summarize its direct steady characteristics."""
    if mesh_nx is None or mesh_nz is None:
        default_nx, default_nz = choose_mesh_dimensions(
            geometry, target_spacing
        )
        mesh_nx = default_nx if mesh_nx is None else mesh_nx
        mesh_nz = default_nz if mesh_nz is None else mesh_nz

    base = {
        "diagnostic method version": DIAGNOSTIC_METHOD_VERSION,
        "wavelength (m)": geometry.wavelength,
        "bed amplitude (m)": geometry.amplitude,
        "bed phase origin (m; coordinate gauge)": geometry.phase_origin,
        "bed peak-to-trough relief (m)": float(
            getattr(geometry, "relief", 2.0 * geometry.amplitude)
        ),
        "mean ice thickness (m)": geometry.mean_thickness,
        "minimum ice thickness (m)": geometry.minimum_thickness,
        "maximum ice thickness (m)": geometry.maximum_thickness,
        "amplitude / mean thickness": geometry.amplitude_ratio,
        "wavelength / mean thickness": geometry.wavelength_ratio,
        "peak absolute bed slope": geometry.peak_bed_slope,
        "mesh nx": int(mesh_nx),
        "mesh nz": int(mesh_nz),
        "mesh vertical grading power": float(vertical_grading_power),
        "sampler nx": int(sampler_nx),
        "sampler nz": int(sampler_nz),
        "sampler vertical grading power": float(
            sampler_vertical_grading_power
        ),
        "candidate source count": int(source_count),
        "retained accumulation-flux fraction": retained_flux_fraction,
        "characteristic source depth (m)": START_DEPTH,
        "ablating-surface buffer thickness fraction": (
            1.0 - ANALYSIS_SURFACE_SIGMA
        ),
        "characteristic time step (yr)": time_step,
        "safety maximum age (yr)": maximum_age,
        "minimum credible surface signal / basal leakage": (
            minimum_surface_signal_to_basal_leakage
        ),
        "basal drag (Pa yr/m)": BASAL_DRAG,
        "surface slope (rad)": SURFACE_SLOPE,
    }
    # With a flat bed and flat surface, incompressibility and impermeability
    # give zero surface mass balance, so there is no finite depositing-source
    # trajectory to trace.  Nevertheless, the continuous A -> 0 geometry is
    # the exactly ordered slab map J=1; retain that formal asymptotic limit in
    # parameter sweeps rather than displaying it as a numerical failure.
    if geometry.amplitude == 0.0:
        return base | {
            "status": "analytic flat-bed limit (zero surface mass balance)",
            "surface vertical flux (m2/yr)": 0.0,
            "mean surface vertical flux (m/yr)": 0.0,
            "maximum absolute surface mass balance (m/yr)": 0.0,
            "accumulation interval count": 0,
            "depositing source count": 0,
            "retained source count": 0,
            "minimum J": 1.0,
            "first percentile of per-source minimum J": 1.0,
            "raw minimum J to exit": 1.0,
            "raw first percentile of per-source minimum J": 1.0,
            "classification": "resolved ordered",
            "raw first J=0 age (yr)": np.nan,
            "surface-clearance fraction at earliest case J=0": np.nan,
            "maximum surface-clearance fraction while J<0": np.nan,
            "maximum surface-clearance fraction at first J=0": np.nan,
            "median surface-clearance fraction at first J=0": np.nan,
            "surface exit count": 0,
            "bed exit count": 0,
            "safety-cap count": 0,
            "numerical-failure count": 0,
            "maximum horizontal step (wavelengths)": 0.0,
            "minimum mechanism C before reversal": 0.0,
            "u / source u at minimum J": 1.0,
            "mechanism C at minimum J": 0.0,
            "reversal before minimum J": False,
            "maximum relative streamfunction identity residual": 0.0,
        }
    try:
        started = perf_counter()
        _, velocity, stokes = solve_stokes(
            geometry,
            int(mesh_nx),
            int(mesh_nz),
            vertical_grading_power=vertical_grading_power,
        )
        stokes_seconds = perf_counter() - started
        started = perf_counter()
        sample, sample_vector, sample_gradient, sampler = (
            make_variational_sampler(
                geometry,
                int(mesh_nx),
                velocity,
                horizontal_samples=int(sampler_nx),
                vertical_samples=int(sampler_nz),
                vertical_grading_power=sampler_vertical_grading_power,
            )
        )
        sampler_seconds = perf_counter() - started
        source_data = retained_surface_sources(
            geometry,
            sample,
            source_count=int(source_count),
            retained_flux_fraction=retained_flux_fraction,
            start_depth=START_DEPTH,
        )
        sources = source_data["retained sources"]
        depositing = source_data["depositing sources"]
        accumulation = source_data["accumulation"]
        surface_signal = float(np.nanmax(np.abs(accumulation)))
        basal_leakage = float(stokes["rms basal normal speed (m/yr)"])
        if basal_leakage > 0.0:
            signal_ratio = surface_signal / basal_leakage
        elif surface_signal > 0.0:
            signal_ratio = np.inf
        else:
            signal_ratio = 0.0
        source_summary = {
            "accumulation interval count": source_data[
                "accumulation interval count"
            ],
            "depositing source count": len(depositing),
            "retained source count": len(sources),
            "sampled net surface mass balance (m2/yr)": source_data[
                "sampled net surface mass balance (m2/yr)"
            ],
            "sampled gross surface mass exchange (m2/yr)": source_data[
                "sampled gross surface mass exchange (m2/yr)"
            ],
            "sampled relative surface mass balance closure": source_data[
                "sampled relative surface mass balance closure"
            ],
            "maximum absolute surface mass balance (m/yr)": surface_signal,
            "surface signal / basal normal leakage": signal_ratio,
        }
        if (
            minimum_surface_signal_to_basal_leakage is not None
            and (
                signal_ratio < minimum_surface_signal_to_basal_leakage
                or source_data[
                    "sampled relative surface mass balance closure"
                ]
                > 1.0e-3
            )
        ):
            return base | stokes | sampler | source_summary | {
                "status": "screened below numerical surface signal",
                "Stokes solve wall time (s)": stokes_seconds,
                "sampler construction wall time (s)": sampler_seconds,
                "minimum J": np.nan,
                "raw minimum J to exit": np.nan,
                "classification": (
                    "not evaluated: surface signal below numerical leakage"
                ),
            }
        if not len(sources):
            return base | stokes | sampler | source_summary | {
                "status": "no retained surface-deposition source",
                "Stokes solve wall time (s)": stokes_seconds,
                "sampler construction wall time (s)": sampler_seconds,
                "minimum J": np.nan,
                "raw minimum J to exit": np.nan,
                "classification": "not evaluated",
            }

        started = perf_counter()
        trajectories = trace_sources_to_exit(
            geometry,
            int(mesh_nx),
            sample_vector,
            sample_gradient,
            sources,
            time_step=time_step,
            maximum_age=maximum_age,
            start_depth=START_DEPTH,
            surface_accumulation_tolerance=source_data[
                "accumulation tolerance"
            ],
        )
        trace_seconds = perf_counter() - started
        failures = trajectories["exit kind"] == "numerical failure"
        bed_exits = trajectories["exit kind"] == "bed"
        safety_caps = trajectories["exit kind"] == "safety cap"
        if np.any(failures):
            status = "numerical characteristic failure"
        elif np.any(bed_exits):
            status = "numerical bed exit"
        elif np.any(safety_caps):
            status = "characteristic safety-cap censoring"
        else:
            status = "success"
        buffered_column = "minimum J outside ablating-surface buffer"
        buffered = trajectories.loc[
            trajectories[buffered_column].notna()
        ]
        if len(buffered):
            minimum_index = buffered[buffered_column].idxmin()
            minimum_row = trajectories.loc[minimum_index]
            minimum_j = float(minimum_row[buffered_column])
            minimum_phase = float(minimum_row["source phase"])
            minimum_age = float(
                minimum_row["age of resolved minimum J (yr)"]
            )
            first_percentile = float(
                np.quantile(buffered[buffered_column], 0.01)
            )
        else:
            minimum_j = np.nan
            minimum_phase = np.nan
            minimum_age = np.nan
            first_percentile = np.nan
            if status == "success":
                status = "no resolved characteristic sample"
        raw_index = trajectories["raw minimum J to exit"].idxmin()
        raw_row = trajectories.loc[raw_index]
        zero_crossings = trajectories.loc[
            trajectories["first J=0 age (yr)"].notna()
        ]
        if len(zero_crossings):
            earliest_zero = zero_crossings.loc[
                zero_crossings["first J=0 age (yr)"].idxmin()
            ]
            case_first_zero_age = float(earliest_zero["first J=0 age (yr)"])
            case_first_zero_clearance = float(
                earliest_zero["surface-clearance fraction at first J=0"]
            )
            maximum_first_zero_clearance = float(
                zero_crossings[
                    "surface-clearance fraction at first J=0"
                ].max()
            )
            median_first_zero_clearance = float(
                zero_crossings[
                    "surface-clearance fraction at first J=0"
                ].median()
            )
        else:
            case_first_zero_age = np.nan
            case_first_zero_clearance = np.nan
            maximum_first_zero_clearance = np.nan
            median_first_zero_clearance = np.nan
        negative_clearances = trajectories[
            "maximum surface-clearance fraction while J<0"
        ].dropna()
        maximum_negative_clearance = (
            float(negative_clearances.max())
            if len(negative_clearances)
            else np.nan
        )
        positive_accumulation = accumulation[accumulation > 0.0]
        exit_counts = trajectories["exit kind"].value_counts()
        return base | stokes | sampler | source_summary | {
            "status": status,
            "Stokes solve wall time (s)": stokes_seconds,
            "sampler construction wall time (s)": sampler_seconds,
            "characteristic wall time (s)": trace_seconds,
            "accumulation interval count": source_data[
                "accumulation interval count"
            ],
            "depositing source count": len(depositing),
            "retained source count": len(sources),
            "source count with resolved samples": len(buffered),
            "sampled net surface mass balance (m2/yr)": source_data[
                "sampled net surface mass balance (m2/yr)"
            ],
            "sampled gross surface mass exchange (m2/yr)": source_data[
                "sampled gross surface mass exchange (m2/yr)"
            ],
            "sampled relative surface mass balance closure": source_data[
                "sampled relative surface mass balance closure"
            ],
            "minimum positive accumulation (m/yr)": float(
                np.min(positive_accumulation)
            ),
            "maximum accumulation (m/yr)": float(
                np.max(positive_accumulation)
            ),
            "maximum absolute surface mass balance (m/yr)": float(
                np.nanmax(np.abs(accumulation))
            ),
            "minimum J": minimum_j,
            "first percentile of per-source minimum J": first_percentile,
            "raw minimum J to exit": float(
                raw_row["raw minimum J to exit"]
            ),
            "raw first percentile of per-source minimum J": float(
                np.quantile(trajectories["raw minimum J to exit"], 0.01)
            ),
            "classification": (
                classify(minimum_j)
                if status == "success"
                else "not evaluated"
            ),
            "source phase at minimum J": minimum_phase,
            "age at minimum J (yr)": minimum_age,
            "source phase at raw minimum J": float(raw_row["source phase"]),
            "age at raw minimum J (yr)": float(
                raw_row["age of raw minimum J (yr)"]
            ),
            "raw first J=0 age (yr)": case_first_zero_age,
            "surface-clearance fraction at earliest case J=0": (
                case_first_zero_clearance
            ),
            "maximum surface-clearance fraction while J<0": (
                maximum_negative_clearance
            ),
            "maximum surface-clearance fraction at first J=0": (
                maximum_first_zero_clearance
            ),
            "median surface-clearance fraction at first J=0": (
                median_first_zero_clearance
            ),
            "minimum path u (m/yr)": float(
                trajectories["minimum path u (m/yr)"].min()
            ),
            "minimum path u / source u before reversal": float(
                trajectories[
                    "minimum path u / source u before reversal"
                ].min()
            ),
            "minimum mechanism C before reversal": float(
                trajectories["minimum mechanism C before reversal"].min()
            ),
            "u / source u at minimum J": float(
                minimum_row["u / source u at minimum J"]
            ),
            "mechanism C at minimum J": float(
                minimum_row["mechanism C at minimum J"]
            ),
            "reversal before minimum J": bool(
                minimum_row["reversal before minimum J"]
            ),
            "u / source u at raw minimum J": float(
                raw_row["u / source u at raw minimum J"]
            ),
            "mechanism C at raw minimum J": float(
                raw_row["mechanism C at raw minimum J"]
            ),
            "reversal before raw minimum J": bool(
                raw_row["reversal before raw minimum J"]
            ),
            "maximum relative streamfunction identity residual": float(
                trajectories[
                    "maximum relative streamfunction identity residual"
                ].max()
            ),
            "maximum velocity reversal count": int(
                trajectories["horizontal velocity reversal count"].max()
            ),
            "minimum exit age (yr)": float(
                trajectories["exit age (yr)"].min()
            ),
            "maximum exit age (yr)": float(
                trajectories["exit age (yr)"].max()
            ),
            "maximum path displacement (wavelengths)": float(
                trajectories[
                    "maximum absolute displacement (wavelengths)"
                ].max()
            ),
            "maximum horizontal step (wavelengths)": float(
                trajectories[
                    "maximum horizontal step (wavelengths)"
                ].max()
            ),
            "surface exit count": int(exit_counts.get("surface", 0)),
            "bed exit count": int(np.count_nonzero(bed_exits)),
            "safety-cap count": int(np.count_nonzero(safety_caps)),
            "numerical-failure count": int(np.count_nonzero(failures)),
        }
    except Exception as error:
        return base | {
            "status": f"failed: {type(error).__name__}: {error}",
            "minimum J": np.nan,
            "classification": "not evaluated",
        }


SMOKE_GEOMETRIES = (
    SinusoidalGeometry(400.0, 30.0, 120.0),
    SinusoidalGeometry(400.0, 70.0, 120.0),
    SinusoidalGeometry(800.0, 40.0, 200.0),
    SinusoidalGeometry(800.0, 120.0, 200.0),
)


def run_smoke_grid():
    """Run four inexpensive cases spanning all three geometric inputs."""
    rows = []
    for geometry in SMOKE_GEOMETRIES:
        print(
            "Solving smoke case "
            f"lambda={geometry.wavelength:g} m, A={geometry.amplitude:g} m, "
            f"H={geometry.mean_thickness:g} m.",
            flush=True,
        )
        rows.append(
            solve_parameter_case(
                geometry,
                mesh_nx=48,
                mesh_nz=12,
                sampler_nx=241,
                sampler_nz=101,
                source_count=161,
                time_step=10.0,
                maximum_age=120_000.0,
            )
        )
    return pd.DataFrame(rows)


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-grid", action="store_true")
    parser.add_argument("--wavelength", type=float, default=600.0)
    parser.add_argument("--amplitude", type=float, default=60.0)
    parser.add_argument("--mean-thickness", type=float, default=160.0)
    parser.add_argument("--mesh-nx", type=int)
    parser.add_argument("--mesh-nz", type=int)
    parser.add_argument("--vertical-grading-power", type=float, default=1.0)
    parser.add_argument("--target-spacing", type=float, default=10.0)
    parser.add_argument("--sampler-nx", type=int, default=DEFAULT_SAMPLER_NX)
    parser.add_argument("--sampler-nz", type=int, default=DEFAULT_SAMPLER_NZ)
    parser.add_argument(
        "--sampler-vertical-grading-power", type=float, default=1.0
    )
    parser.add_argument("--source-count", type=int, default=DEFAULT_SOURCE_COUNT)
    parser.add_argument(
        "--retained-flux-fraction",
        type=float,
        default=DEFAULT_RETAINED_FLUX_FRACTION,
    )
    parser.add_argument("--time-step", type=float, default=DEFAULT_TIME_STEP)
    parser.add_argument(
        "--maximum-age", type=float, default=DEFAULT_MAXIMUM_AGE
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_sinusoidal_smoke.csv",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.smoke_grid:
        table = run_smoke_grid()
    else:
        geometry = SinusoidalGeometry(
            args.wavelength, args.amplitude, args.mean_thickness
        )
        table = pd.DataFrame(
            [
                solve_parameter_case(
                    geometry,
                    mesh_nx=args.mesh_nx,
                    mesh_nz=args.mesh_nz,
                    vertical_grading_power=args.vertical_grading_power,
                    target_spacing=args.target_spacing,
                    sampler_nx=args.sampler_nx,
                    sampler_nz=args.sampler_nz,
                    sampler_vertical_grading_power=(
                        args.sampler_vertical_grading_power
                    ),
                    source_count=args.source_count,
                    retained_flux_fraction=args.retained_flux_fraction,
                    time_step=args.time_step,
                    maximum_age=args.maximum_age,
                )
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
