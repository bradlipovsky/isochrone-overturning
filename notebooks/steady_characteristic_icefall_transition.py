"""Steady overturn diagnostic for a smooth icefall-to-tongue transition.

The calculation isolates a fourth candidate fold mechanism: a short, steep,
thin icefall feeding a long, gentle, thick tongue.  One exactly periodic cell
is used so that the result cannot be caused by an open boundary.  The mean
surface slope is represented by the gravity vector, while a smooth periodic
surface perturbation creates the local slope contrast.  Bed and surface are
both smooth and the basal boundary retains the same passive linear sliding
law used throughout the PR40 calculations.

The diagnostic is direct and steady.  Surface-deposited characteristics and
their equal-age tangents are carried through one fixed Stokes velocity field
until they return to the surface.  No time-dependent isochrone calculation is
performed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import comb
from pathlib import Path
from time import perf_counter
import logging
import warnings

warnings.filterwarnings("ignore", message="Unable to import recommended hash")
warnings.filterwarnings("ignore", message="The .*Function.at.* method")
logging.getLogger("tsfc").setLevel(logging.ERROR)

import numpy as np
import pandas as pd

try:
    import firedrake as fd
except ModuleNotFoundError:  # Geometry/plotting utilities remain importable.
    fd = None

import steady_characteristic_sinusoidal_parameter_map as characteristic


DIAGNOSTIC_METHOD_VERSION = "icefall-transition-steady-v2-physical-tangent"
DEFAULT_MEAN_THICKNESS = 160.0
DEFAULT_SHARPNESS = 4
DEFAULT_TARGET_SPACING = 8.0
DEFAULT_TIME_STEP = 10.0
DEFAULT_MAXIMUM_AGE = 300_000.0
DEFAULT_SOURCE_COUNT = 241
DEFAULT_SAMPLER_NX = 481
DEFAULT_SAMPLER_NZ = 161


@dataclass(frozen=True)
class IcefallTransitionGeometry:
    """A periodic steep/thin reach followed by a gentle/thick reach.

    ``slope_ratio`` is the ratio of the maximum local downhill surface slope
    to its gentle-reach value.  ``thickness_ratio`` is the ratio of the thick
    gentle-reach thickness to the thin icefall thickness.  ``sharpness`` is
    the integer exponent in the smooth window

        W = [(1 + cos(2 pi x / L)) / 2] ** sharpness.

    The mean surface slope and mean thickness are preserved exactly.
    """

    wavelength: float
    mean_thickness: float
    slope_ratio: float
    thickness_ratio: float
    sharpness: int = DEFAULT_SHARPNESS
    phase_origin: float = 0.0

    def __post_init__(self):
        values = (
            self.wavelength,
            self.mean_thickness,
            self.slope_ratio,
            self.thickness_ratio,
            self.phase_origin,
        )
        if not all(np.isfinite(values)):
            raise ValueError("geometry values must be finite")
        if self.wavelength <= 0.0 or self.mean_thickness <= 0.0:
            raise ValueError("wavelength and mean thickness must be positive")
        if self.slope_ratio < 1.0 or self.thickness_ratio < 1.0:
            raise ValueError("ratios must be at least one")
        if int(self.sharpness) != self.sharpness or self.sharpness < 1:
            raise ValueError("sharpness must be a positive integer")
        if self.minimum_thickness <= 0.0:
            raise ValueError("thickness ratio pinches the domain")

    @property
    def wavelength_ratio(self):
        return self.wavelength / self.mean_thickness

    @property
    def window_mean(self):
        p = int(self.sharpness)
        return comb(2 * p, p) / 4.0**p

    @property
    def _window_coefficients(self):
        p = int(self.sharpness)
        return np.array(
            [2.0 ** (1 - 2 * p) * comb(2 * p, p - k) for k in range(1, p + 1)]
        )

    @property
    def gentle_slope(self):
        f = self.window_mean
        return characteristic.SURFACE_SLOPE / (
            1.0 + (self.slope_ratio - 1.0) * f
        )

    @property
    def steep_slope(self):
        return self.slope_ratio * self.gentle_slope

    @property
    def _thickness_contrast(self):
        f = self.window_mean
        ratio = self.thickness_ratio
        return (ratio - 1.0) / (f + ratio * (1.0 - f))

    @property
    def minimum_thickness(self):
        f = self.window_mean
        return self.mean_thickness * (
            1.0 - self._thickness_contrast * (1.0 - f)
        )

    @property
    def maximum_thickness(self):
        f = self.window_mean
        return self.mean_thickness * (1.0 + self._thickness_contrast * f)

    @property
    def amplitude(self):
        """Half of the thickness range (for generic diagnostic metadata)."""
        return 0.5 * (self.maximum_thickness - self.minimum_thickness)

    @property
    def amplitude_ratio(self):
        return self.amplitude / self.mean_thickness

    def _phase(self, x):
        return (
            2.0
            * np.pi
            * np.mod(np.asarray(x, dtype=float) - self.phase_origin, self.wavelength)
            / self.wavelength
        )

    def window(self, x):
        phase = self._phase(x)
        return (0.5 * (1.0 + np.cos(phase))) ** int(self.sharpness)

    def window_slope(self, x):
        phase = self._phase(x)
        coefficients = self._window_coefficients
        harmonics = np.arange(1, len(coefficients) + 1, dtype=float)
        return np.sum(
            -coefficients[:, None]
            * harmonics[:, None]
            * (2.0 * np.pi / self.wavelength)
            * np.sin(harmonics[:, None] * np.atleast_1d(phase)[None, :]),
            axis=0,
        ).reshape(np.shape(phase))

    def local_downhill_slope(self, x):
        return self.gentle_slope * (
            1.0 + (self.slope_ratio - 1.0) * self.window(x)
        )

    def surface_slope(self, x):
        """Slope of the periodic surface perturbation in mean-slope axes."""
        return characteristic.SURFACE_SLOPE - self.local_downhill_slope(x)

    def surface_height(self, x):
        phase = self._phase(x)
        coefficients = self._window_coefficients
        harmonics = np.arange(1, len(coefficients) + 1, dtype=float)
        slope_contrast = self.steep_slope - self.gentle_slope
        values = np.sum(
            -slope_contrast
            * coefficients[:, None]
            * self.wavelength
            / (2.0 * np.pi * harmonics[:, None])
            * np.sin(harmonics[:, None] * np.atleast_1d(phase)[None, :]),
            axis=0,
        ).reshape(np.shape(phase))
        return values

    def thickness(self, x):
        return self.mean_thickness * (
            1.0
            + self._thickness_contrast * (self.window_mean - self.window(x))
        )

    def thickness_slope(self, x):
        return (
            -self.mean_thickness
            * self._thickness_contrast
            * self.window_slope(x)
        )

    def bed_height(self, x):
        return self.surface_height(x) - self.thickness(x)

    def bed_slope(self, x):
        return self.surface_slope(x) - self.thickness_slope(x)

    @property
    def peak_bed_slope(self):
        sample_x = np.linspace(0.0, self.wavelength, 4096, endpoint=False)
        return float(np.max(np.abs(self.bed_slope(sample_x))))


def represented_periodic_value(function, x, geometry, mesh_nx):
    nodes = np.arange(mesh_nx, dtype=float) * geometry.wavelength / mesh_nx
    values = np.asarray(function(nodes), dtype=float)
    return np.interp(
        np.mod(np.asarray(x, dtype=float), geometry.wavelength),
        np.r_[nodes, geometry.wavelength],
        np.r_[values, values[0]],
    )


def represented_periodic_slope(function, x, geometry, mesh_nx):
    dx = geometry.wavelength / mesh_nx
    wrapped = np.mod(np.asarray(x, dtype=float), geometry.wavelength)
    left = np.floor(wrapped / dx).astype(int) % mesh_nx
    right = (left + 1) % mesh_nx
    nodes = np.arange(mesh_nx, dtype=float) * dx
    values = np.asarray(function(nodes), dtype=float)
    return (values[right] - values[left]) / dx


def choose_mesh_dimensions(geometry, target_spacing=DEFAULT_TARGET_SPACING):
    nx = max(
        64,
        8 * int(np.ceil(geometry.wavelength / target_spacing / 8.0)),
        16 * int(geometry.sharpness),
    )
    nz = max(16, int(np.ceil(geometry.maximum_thickness / target_spacing)))
    return nx, nz


def make_mesh(geometry, nx, nz, vertical_grading_power=1.0):
    if fd is None:
        raise ModuleNotFoundError("Firedrake is required to solve the flow")
    mesh = fd.PeriodicRectangleMesh(
        int(nx), int(nz), geometry.wavelength, 1.0, direction="x"
    )
    reference = mesh.coordinates.dat.data_ro.copy()
    x = reference[:, 0]
    sigma = characteristic.endpoint_clustered_coordinate(
        reference[:, 1], vertical_grading_power
    )
    surface = geometry.surface_height(x)
    local_thickness = geometry.thickness(x)
    mesh.coordinates.dat.data[:, 1] = surface - local_thickness * (1.0 - sigma)
    mesh.clear_spatial_index()
    return mesh


def solve_stokes(geometry, nx, nz, vertical_grading_power=1.0):
    if fd is None:
        raise ModuleNotFoundError("Firedrake is required to solve the flow")
    mesh = make_mesh(geometry, nx, nz, vertical_grading_power)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    solution = fd.Function(velocity_space * pressure_space)
    u, p = fd.split(solution)
    v, q = fd.TestFunctions(solution.function_space())
    body_force = fd.Constant(
        (
            characteristic.DENSITY
            * characteristic.GRAVITY
            * np.sin(characteristic.SURFACE_SLOPE),
            -characteristic.DENSITY
            * characteristic.GRAVITY
            * np.cos(characteristic.SURFACE_SLOPE),
        )
    )
    residual = (
        2.0
        * characteristic.VISCOSITY
        * fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v)))
        - p * fd.div(v)
        - q * fd.div(u)
        - fd.dot(body_force, v)
    ) * fd.dx

    normal = fd.FacetNormal(mesh)
    tangent = fd.as_vector((-normal[1], normal[0]))
    stress = (
        2.0 * characteristic.VISCOSITY * fd.sym(fd.grad(u))
        - p * fd.Identity(2)
    )
    test_stress = (
        2.0 * characteristic.VISCOSITY * fd.sym(fd.grad(v))
        - q * fd.Identity(2)
    )
    normal_speed = fd.dot(u, normal)
    test_normal_speed = fd.dot(v, normal)
    residual += (
        characteristic.BASAL_DRAG * fd.dot(u, tangent) * fd.dot(v, tangent)
        - fd.dot(stress * normal, normal) * test_normal_speed
        - fd.dot(test_stress * normal, normal) * normal_speed
        + characteristic.NITSCHE_PENALTY
        * characteristic.VISCOSITY
        / fd.CellDiameter(mesh)
        * normal_speed
        * test_normal_speed
    ) * fd.ds(1)

    fd.solve(
        residual == 0,
        solution,
        solver_parameters=characteristic.SOLVER_PARAMETERS,
    )
    velocity, pressure = solution.subfunctions

    bed_length = fd.assemble(fd.Constant(1.0) * fd.ds(1, domain=mesh))
    surface_length = fd.assemble(fd.Constant(1.0) * fd.ds(2, domain=mesh))
    physical_stress = (
        2.0 * characteristic.VISCOSITY * fd.sym(fd.grad(velocity))
        - pressure * fd.Identity(2)
    )
    gravity_power = fd.assemble(fd.dot(body_force, velocity) * fd.dx)
    basal_dissipation = fd.assemble(
        characteristic.BASAL_DRAG
        * fd.dot(velocity, tangent) ** 2
        * fd.ds(1)
    )
    viscous_dissipation = fd.assemble(
        2.0
        * characteristic.VISCOSITY
        * fd.inner(fd.sym(fd.grad(velocity)), fd.sym(fd.grad(velocity)))
        * fd.dx
    )
    pressure_work = fd.assemble(-2.0 * pressure * fd.div(velocity) * fd.dx)
    nitsche_work = fd.assemble(
        (
            -2.0
            * fd.dot(physical_stress * normal, normal)
            * fd.dot(velocity, normal)
            + characteristic.NITSCHE_PENALTY
            * characteristic.VISCOSITY
            / fd.CellDiameter(mesh)
            * fd.dot(velocity, normal) ** 2
        )
        * fd.ds(1)
    )
    surface_normal_flux = fd.assemble(
        fd.dot(velocity, normal) * fd.ds(2)
    )
    return velocity, {
        "mesh cells": int(mesh.num_cells()),
        "gravity power": float(gravity_power),
        "basal dissipation / gravity power": float(
            basal_dissipation / gravity_power
        ),
        "relative discrete weak-form residual": float(
            (
                gravity_power
                - basal_dissipation
                - viscous_dissipation
                - pressure_work
                - nitsche_work
            )
            / gravity_power
        ),
        "rms basal normal speed (m/yr)": float(
            np.sqrt(
                fd.assemble(fd.dot(velocity, normal) ** 2 * fd.ds(1))
                / bed_length
            )
        ),
        "mean surface normal flux (m/yr)": float(
            surface_normal_flux / surface_length
        ),
        "minimum nodal u (m/yr)": float(
            np.min(velocity.dat.data_ro[:, 0])
        ),
        "maximum nodal u (m/yr)": float(
            np.max(velocity.dat.data_ro[:, 0])
        ),
    }


def make_physical_sampler(
    geometry,
    mesh_nx,
    velocity,
    horizontal_samples=DEFAULT_SAMPLER_NX,
    vertical_samples=DEFAULT_SAMPLER_NZ,
    vertical_grading_power=1.0,
):
    """Return one bilinear interpolant of physical velocity and its gradient.

    Characteristics stay in physical coordinates.  This matters because the
    finite-element coordinate map is only piecewise smooth: differentiating a
    flattened vertical velocity would introduce artificial jumps at mesh
    columns even though the physical finite-element velocity is continuous.
    """

    dx = geometry.wavelength / horizontal_samples
    x_grid = (np.arange(horizontal_samples) + 0.5) * dx
    reference_sigma = np.linspace(0.0, 1.0, vertical_samples)
    sigma_grid = characteristic.BED_SIGMA + (
        characteristic.SURFACE_SIGMA - characteristic.BED_SIGMA
    ) * characteristic.endpoint_clustered_coordinate(
        reference_sigma, vertical_grading_power
    )
    local_h = represented_periodic_value(
        geometry.thickness, x_grid, geometry, mesh_nx
    )
    bed = represented_periodic_value(
        geometry.bed_height, x_grid, geometry, mesh_nx
    )
    physical_z = bed[None, :] + sigma_grid[:, None] * local_h[None, :]
    X = np.tile(x_grid, (vertical_samples, 1))
    points = np.column_stack((X.ravel(), physical_z.ravel()))
    values = np.asarray(
        velocity.at(points, tolerance=1.0e-7), dtype=float
    ).reshape(vertical_samples, horizontal_samples, 2)

    def interpolate(points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        wrapped = np.mod(points[:, 0], geometry.wavelength)
        h = represented_periodic_value(
            geometry.thickness, wrapped, geometry, mesh_nx
        )
        local_bed = represented_periodic_value(
            geometry.bed_height, wrapped, geometry, mesh_nx
        )
        sigma = (points[:, 1] - local_bed) / h
        horizontal_coordinate = wrapped / dx - 0.5
        ix_unwrapped = np.floor(horizontal_coordinate).astype(int)
        fx = horizontal_coordinate - ix_unwrapped
        ix0 = ix_unwrapped % horizontal_samples
        ix1 = (ix0 + 1) % horizontal_samples
        iz = np.searchsorted(sigma_grid, sigma, side="right") - 1
        iz = np.clip(iz, 0, vertical_samples - 2)
        dsigma = sigma_grid[iz + 1] - sigma_grid[iz]
        fsigma = (sigma - sigma_grid[iz]) / dsigma
        v00 = values[iz, ix0]
        v01 = values[iz, ix1]
        v10 = values[iz + 1, ix0]
        v11 = values[iz + 1, ix1]
        velocity_value = (
            ((1.0 - fsigma) * (1.0 - fx))[:, None] * v00
            + ((1.0 - fsigma) * fx)[:, None] * v01
            + (fsigma * (1.0 - fx))[:, None] * v10
            + (fsigma * fx)[:, None] * v11
        )
        derivative_sigma = (
            (1.0 - fx)[:, None] * (v10 - v00)
            + fx[:, None] * (v11 - v01)
        ) / dsigma[:, None]
        derivative_x_sigma = (
            (1.0 - fsigma)[:, None] * (v01 - v00)
            + fsigma[:, None] * (v11 - v10)
        ) / dx
        h_x = represented_periodic_slope(
            geometry.thickness, wrapped, geometry, mesh_nx
        )
        bed_x = represented_periodic_slope(
            geometry.bed_height, wrapped, geometry, mesh_nx
        )
        sigma_x = -(bed_x + sigma * h_x) / h
        gradient = np.empty((len(points), 2, 2))
        gradient[:, :, 0] = (
            derivative_x_sigma + derivative_sigma * sigma_x[:, None]
        )
        gradient[:, :, 1] = derivative_sigma / h[:, None]
        return velocity_value, gradient

    def sample(name, x, z):
        x, z = np.broadcast_arrays(x, z)
        if name == "a":
            wrapped = np.mod(x.ravel(), geometry.wavelength)
            surface = represented_periodic_value(
                geometry.surface_height, wrapped, geometry, mesh_nx
            )
            query = np.column_stack((wrapped, surface))
            velocity_value = interpolate(query)[0]
            slope = represented_periodic_slope(
                geometry.surface_height, wrapped, geometry, mesh_nx
            )
            result = (
                velocity_value[:, 0] * slope - velocity_value[:, 1]
            ).reshape(x.shape)
        else:
            query = np.column_stack((x.ravel(), z.ravel()))
            component = 0 if name == "u" else 1
            result = interpolate(query)[0][:, component].reshape(x.shape)
        return float(result) if result.ndim == 0 else result

    return (
        sample,
        lambda points: interpolate(points)[0],
        lambda points: interpolate(points)[1],
        {
            "minimum sampled u (m/yr)": float(np.nanmin(values[..., 0])),
            "maximum sampled u (m/yr)": float(np.nanmax(values[..., 0])),
            "minimum sampled w (m/yr)": float(np.nanmin(values[..., 1])),
            "maximum sampled w (m/yr)": float(np.nanmax(values[..., 1])),
        },
    )


def physical_local_sigma(points, geometry, mesh_nx):
    points = np.asarray(points, dtype=float)
    local_bed = represented_periodic_value(
        geometry.bed_height, points[:, 0], geometry, mesh_nx
    )
    local_h = represented_periodic_value(
        geometry.thickness, points[:, 0], geometry, mesh_nx
    )
    return (points[:, 1] - local_bed) / local_h


def physical_surface_points(x, geometry, mesh_nx):
    x = np.asarray(x, dtype=float)
    return np.column_stack(
        (
            x,
            represented_periodic_value(
                geometry.surface_height, x, geometry, mesh_nx
            ),
        )
    )


def physical_surface_accumulation(x, geometry, mesh_nx, sample_vector):
    x = np.asarray(x, dtype=float)
    velocity = sample_vector(physical_surface_points(x, geometry, mesh_nx))
    slope = represented_periodic_slope(
        geometry.surface_height, x, geometry, mesh_nx
    )
    return velocity[:, 0] * slope - velocity[:, 1]


def retained_physical_surface_sources(
    geometry,
    sample,
    source_count=DEFAULT_SOURCE_COUNT,
    retained_flux_fraction=characteristic.DEFAULT_RETAINED_FLUX_FRACTION,
):
    """Find cyclic surface-deposition intervals using physical mass balance."""
    candidates = np.linspace(
        0.0, geometry.wavelength, int(source_count), endpoint=False
    )
    accumulation = sample("a", candidates, np.zeros(len(candidates)))
    tolerance = max(1.0e-10, 1.0e-8 * np.nanmax(np.abs(accumulation)))
    mask = np.isfinite(accumulation) & (accumulation > tolerance)
    runs = characteristic._cyclic_true_runs(mask)
    retained = []
    tail = 0.5 * (1.0 - retained_flux_fraction)
    for run in runs:
        weights = accumulation[run]
        flux_coordinate = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
        selected = run[
            (flux_coordinate >= tail) & (flux_coordinate <= 1.0 - tail)
        ]
        if not len(selected):
            selected = np.array([run[int(np.argmax(weights))]])
        retained.extend(selected.tolist())
    retained = np.array(sorted(set(retained)), dtype=int)
    depositing = np.flatnonzero(mask)
    dx = geometry.wavelength / source_count
    net_flux = float(dx * np.sum(accumulation))
    gross_flux = float(dx * np.sum(np.abs(accumulation)))
    return {
        "retained sources": candidates[retained],
        "depositing sources": candidates[depositing],
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


def trace_physical_sources_to_exit(
    geometry,
    mesh_nx,
    sample_vector,
    sample_gradient,
    sources,
    time_step=DEFAULT_TIME_STEP,
    maximum_age=DEFAULT_MAXIMUM_AGE,
    surface_accumulation_tolerance=0.0,
):
    """Trace physical characteristics and physical surface tangents to exit."""
    sources = np.asarray(sources, dtype=float)
    count = len(sources)
    points = physical_surface_points(sources, geometry, mesh_nx)
    source_slope = represented_periodic_slope(
        geometry.surface_height, sources, geometry, mesh_nx
    )
    tangents = np.column_stack((np.ones(count), source_slope))
    active = np.ones(count, dtype=bool)
    age = np.zeros(count)
    raw_minimum_j = np.ones(count)
    raw_minimum_age = np.zeros(count)
    buffered_minimum_j = np.ones(count)
    buffered_minimum_age = np.zeros(count)
    first_zero_age = np.full(count, np.nan)
    first_zero_clearance = np.full(count, np.nan)
    maximum_negative_clearance = np.full(count, np.nan)
    exit_age = np.full(count, np.nan)
    exit_kind = np.full(count, "active", dtype=object)
    source_velocity = sample_vector(points)
    source_u = source_velocity[:, 0]
    source_accumulation = source_u * source_slope - source_velocity[:, 1]
    minimum_u = source_u.copy()
    previous_u = source_u.copy()
    reversal_count = np.zeros(count, dtype=int)
    minimum_speed_ratio = np.ones(count)
    minimum_mechanism_c = np.zeros(count)
    raw_speed_ratio = np.ones(count)
    raw_mechanism_c = np.zeros(count)
    raw_reversal = np.zeros(count, dtype=bool)
    buffered_speed_ratio = np.ones(count)
    buffered_mechanism_c = np.zeros(count)
    buffered_reversal = np.zeros(count, dtype=bool)
    maximum_identity_residual = np.zeros(count)
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
        old_sigma = physical_local_sigma(old_points, geometry, mesh_nx)
        candidate_points, candidate_tangents = characteristic.rk4_step(
            sample_vector,
            sample_gradient,
            old_points,
            old_tangents,
            step,
        )
        candidate_sigma = physical_local_sigma(
            candidate_points, geometry, mesh_nx
        )
        finite = np.all(np.isfinite(candidate_points), axis=1) & np.all(
            np.isfinite(candidate_tangents), axis=1
        )
        remains = (
            finite
            & (candidate_sigma > characteristic.BED_SIGMA)
            & (candidate_sigma < characteristic.SURFACE_SIGMA)
        )
        accepted_points = candidate_points.copy()
        accepted_tangents = candidate_tangents.copy()
        accepted_age = old_age + step
        step_exit_kind = np.full(len(indices), "", dtype=object)
        crossing = finite & ~remains
        if np.any(crossing):
            fraction, local_kind = characteristic._vertical_exit_fraction(
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
        accepted_sigma = physical_local_sigma(
            accepted_points, geometry, mesh_nx
        )
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
        reversals = reversal_count[indices] + reversal.astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            speed_ratio = accepted_u / source_u[indices]
            mechanism_c = accepted_j / speed_ratio - 1.0
        mechanism_valid = (
            (reversals == 0)
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
        maximum_identity_residual[indices] = np.fmax(
            maximum_identity_residual[indices],
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
            first_zero_age[selected] = old_age[crossed_zero] + fraction * (
                accepted_age[crossed_zero] - old_age[crossed_zero]
            )
            zero_points = old_points[crossed_zero] + fraction[:, None] * (
                accepted_points[crossed_zero] - old_points[crossed_zero]
            )
            first_zero_clearance[selected] = 1.0 - physical_local_sigma(
                zero_points, geometry, mesh_nx
            )

        improves = accepted_j < raw_minimum_j[indices]
        if np.any(improves):
            selected = indices[improves]
            raw_minimum_j[selected] = accepted_j[improves]
            raw_minimum_age[selected] = accepted_age[improves]
            raw_speed_ratio[selected] = np.where(
                mechanism_valid[improves], speed_ratio[improves], np.nan
            )
            raw_mechanism_c[selected] = np.where(
                mechanism_valid[improves], mechanism_c[improves], np.nan
            )
            raw_reversal[selected] = reversals[improves] > 0

        local_accumulation = physical_surface_accumulation(
            accepted_points[:, 0], geometry, mesh_nx, sample_vector
        )
        resolved = (accepted_sigma <= characteristic.ANALYSIS_SURFACE_SIGMA) | (
            local_accumulation >= -surface_accumulation_tolerance
        )
        local_buffered_j = np.full(len(indices), np.inf)
        local_buffered_age = np.full(len(indices), np.nan)
        local_buffered_speed_ratio = np.full(len(indices), np.nan)
        local_buffered_mechanism_c = np.full(len(indices), np.nan)
        local_buffered_reversal = reversals > 0
        local_buffered_j[resolved] = accepted_j[resolved]
        local_buffered_age[resolved] = accepted_age[resolved]
        local_buffered_speed_ratio[resolved] = np.where(
            mechanism_valid[resolved], speed_ratio[resolved], np.nan
        )
        local_buffered_mechanism_c[resolved] = np.where(
            mechanism_valid[resolved], mechanism_c[resolved], np.nan
        )
        crosses_buffer = (
            (old_sigma - characteristic.ANALYSIS_SURFACE_SIGMA)
            * (accepted_sigma - characteristic.ANALYSIS_SURFACE_SIGMA)
            <= 0.0
        ) & (old_sigma != accepted_sigma)
        if np.any(crosses_buffer):
            fraction = (
                characteristic.ANALYSIS_SURFACE_SIGMA
                - old_sigma[crosses_buffer]
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
                boundary_mechanism_c = (
                    boundary_j / boundary_speed_ratio - 1.0
                )
            boundary_reversal = reversals[crosses_buffer] > 0
            boundary_valid = (
                ~boundary_reversal
                & np.isfinite(boundary_speed_ratio)
                & np.isfinite(boundary_mechanism_c)
                & (boundary_speed_ratio >= 1.0e-3)
            )
            target = np.flatnonzero(crosses_buffer)
            better = boundary_j < local_buffered_j[target]
            local_buffered_j[target[better]] = boundary_j[better]
            local_buffered_age[target[better]] = boundary_age[better]
            local_buffered_speed_ratio[target[better]] = np.where(
                boundary_valid[better], boundary_speed_ratio[better], np.nan
            )
            local_buffered_mechanism_c[target[better]] = np.where(
                boundary_valid[better], boundary_mechanism_c[better], np.nan
            )
            local_buffered_reversal[target[better]] = boundary_reversal[better]
        improves_buffered = (
            local_buffered_j < buffered_minimum_j[indices]
        )
        if np.any(improves_buffered):
            selected = indices[improves_buffered]
            buffered_minimum_j[selected] = local_buffered_j[improves_buffered]
            buffered_minimum_age[selected] = local_buffered_age[
                improves_buffered
            ]
            buffered_speed_ratio[selected] = local_buffered_speed_ratio[
                improves_buffered
            ]
            buffered_mechanism_c[selected] = local_buffered_mechanism_c[
                improves_buffered
            ]
            buffered_reversal[selected] = local_buffered_reversal[
                improves_buffered
            ]

        negative = accepted_j < 0.0
        if np.any(negative):
            selected = indices[negative]
            maximum_negative_clearance[selected] = np.fmax(
                maximum_negative_clearance[selected],
                1.0 - accepted_sigma[negative],
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
            exited = indices[exiting]
            active[exited] = False
            exit_age[exited] = accepted_age[exiting]
            exit_kind[exited] = step_exit_kind[exiting]

    if np.any(active):
        exit_age[active] = age[active]
        exit_kind[active] = "safety cap"
    return pd.DataFrame(
        {
            "source x0 (m)": sources,
            "source phase": sources / geometry.wavelength,
            "minimum J outside ablating-surface buffer": buffered_minimum_j,
            "age of resolved minimum J (yr)": buffered_minimum_age,
            "raw minimum J to exit": raw_minimum_j,
            "age of raw minimum J (yr)": raw_minimum_age,
            "first J=0 age (yr)": first_zero_age,
            "surface-clearance fraction at first J=0": first_zero_clearance,
            "maximum surface-clearance fraction while J<0": (
                maximum_negative_clearance
            ),
            "minimum path u (m/yr)": minimum_u,
            "minimum path u / source u before reversal": minimum_speed_ratio,
            "minimum mechanism C before reversal": minimum_mechanism_c,
            "u / source u at minimum J": buffered_speed_ratio,
            "mechanism C at minimum J": buffered_mechanism_c,
            "reversal before minimum J": buffered_reversal,
            "u / source u at raw minimum J": raw_speed_ratio,
            "mechanism C at raw minimum J": raw_mechanism_c,
            "reversal before raw minimum J": raw_reversal,
            "maximum relative streamfunction identity residual": (
                maximum_identity_residual
            ),
            "horizontal velocity reversal count": reversal_count,
            "maximum absolute displacement (wavelengths)": (
                maximum_absolute_periods
            ),
            "maximum horizontal step (wavelengths)": maximum_step_periods,
            "exit age (yr)": exit_age,
            "exit kind": exit_kind,
            "exit phase": np.mod(points[:, 0], geometry.wavelength)
            / geometry.wavelength,
        }
    )


def solve_parameter_case(
    geometry,
    *,
    mesh_nx=None,
    mesh_nz=None,
    vertical_grading_power=1.0,
    target_spacing=DEFAULT_TARGET_SPACING,
    sampler_nx=DEFAULT_SAMPLER_NX,
    sampler_nz=DEFAULT_SAMPLER_NZ,
    sampler_vertical_grading_power=1.0,
    source_count=DEFAULT_SOURCE_COUNT,
    retained_flux_fraction=characteristic.DEFAULT_RETAINED_FLUX_FRACTION,
    time_step=DEFAULT_TIME_STEP,
    maximum_age=DEFAULT_MAXIMUM_AGE,
):
    if mesh_nx is None or mesh_nz is None:
        default_nx, default_nz = choose_mesh_dimensions(geometry, target_spacing)
        mesh_nx = default_nx if mesh_nx is None else mesh_nx
        mesh_nz = default_nz if mesh_nz is None else mesh_nz
    base = {
        "diagnostic method version": DIAGNOSTIC_METHOD_VERSION,
        "wavelength (m)": geometry.wavelength,
        "wavelength / mean thickness": geometry.wavelength_ratio,
        "mean ice thickness (m)": geometry.mean_thickness,
        "minimum ice thickness (m)": geometry.minimum_thickness,
        "maximum ice thickness (m)": geometry.maximum_thickness,
        "steep / gentle surface-slope ratio": geometry.slope_ratio,
        "thick / thin thickness ratio": geometry.thickness_ratio,
        "gentle local downhill slope": geometry.gentle_slope,
        "steep local downhill slope": geometry.steep_slope,
        "mean downhill slope": characteristic.SURFACE_SLOPE,
        "smooth-window sharpness": geometry.sharpness,
        "smooth-window mean fraction": geometry.window_mean,
        "peak absolute physical bed slope": geometry.peak_bed_slope,
        "mesh nx": int(mesh_nx),
        "mesh nz": int(mesh_nz),
        "sampler nx": int(sampler_nx),
        "sampler nz": int(sampler_nz),
        "candidate source count": int(source_count),
        "retained accumulation-flux fraction": retained_flux_fraction,
        "characteristic time step (yr)": time_step,
        "safety maximum age (yr)": maximum_age,
        "basal drag (Pa yr/m)": characteristic.BASAL_DRAG,
    }
    if geometry.slope_ratio == 1.0 and geometry.thickness_ratio == 1.0:
        return base | {
            "status": "analytic uniform-slab limit (zero surface mass balance)",
            "minimum J": 1.0,
            "raw minimum J to exit": 1.0,
            "classification": "resolved ordered",
            "retained source count": 0,
            "minimum mechanism C before reversal": 0.0,
            "u / source u at minimum J": 1.0,
            "mechanism C at minimum J": 0.0,
            "maximum velocity reversal count": 0,
        }
    try:
        started = perf_counter()
        velocity, stokes = solve_stokes(
            geometry, int(mesh_nx), int(mesh_nz), vertical_grading_power
        )
        stokes_seconds = perf_counter() - started
        started = perf_counter()
        sample, sample_vector, sample_gradient, sampler = make_physical_sampler(
            geometry,
            int(mesh_nx),
            velocity,
            horizontal_samples=int(sampler_nx),
            vertical_samples=int(sampler_nz),
            vertical_grading_power=sampler_vertical_grading_power,
        )
        sampler_seconds = perf_counter() - started
        source_data = retained_physical_surface_sources(
            geometry,
            sample,
            source_count=int(source_count),
            retained_flux_fraction=retained_flux_fraction,
        )
        sources = source_data["retained sources"]
        accumulation = source_data["accumulation"]
        source_summary = {
            "accumulation interval count": source_data[
                "accumulation interval count"
            ],
            "depositing source count": len(source_data["depositing sources"]),
            "retained source count": len(sources),
            "sampled relative surface mass balance closure": source_data[
                "sampled relative surface mass balance closure"
            ],
            "sampled gross surface mass exchange (m2/yr)": source_data[
                "sampled gross surface mass exchange (m2/yr)"
            ],
            "maximum absolute surface mass balance (m/yr)": float(
                np.nanmax(np.abs(accumulation))
            ),
        }
        if not len(sources):
            return base | stokes | sampler | source_summary | {
                "status": "no retained surface-deposition source",
                "minimum J": np.nan,
                "raw minimum J to exit": np.nan,
                "classification": "not evaluated",
            }
        started = perf_counter()
        paths = trace_physical_sources_to_exit(
            geometry,
            int(mesh_nx),
            sample_vector,
            sample_gradient,
            sources,
            time_step=time_step,
            maximum_age=maximum_age,
            surface_accumulation_tolerance=source_data[
                "accumulation tolerance"
            ],
        )
        trace_seconds = perf_counter() - started
        failures = paths["exit kind"] == "numerical failure"
        bed_exits = paths["exit kind"] == "bed"
        caps = paths["exit kind"] == "safety cap"
        if np.any(failures):
            status = "numerical characteristic failure"
        elif np.any(bed_exits):
            status = "numerical bed exit"
        elif np.any(caps):
            status = "characteristic safety-cap censoring"
        else:
            status = "success"
        resolved_column = "minimum J outside ablating-surface buffer"
        resolved = paths.loc[paths[resolved_column].notna()]
        if len(resolved):
            minimum_row = paths.loc[resolved[resolved_column].idxmin()]
            minimum_j = float(minimum_row[resolved_column])
        else:
            minimum_row = paths.iloc[paths["raw minimum J to exit"].argmin()]
            minimum_j = np.nan
        raw_row = paths.loc[paths["raw minimum J to exit"].idxmin()]
        exits = paths["exit kind"].value_counts()
        signal = float(np.nanmax(np.abs(accumulation)))
        leakage = float(stokes["rms basal normal speed (m/yr)"])
        return base | stokes | sampler | source_summary | {
            "status": status,
            "Stokes solve wall time (s)": stokes_seconds,
            "sampler construction wall time (s)": sampler_seconds,
            "characteristic wall time (s)": trace_seconds,
            "surface signal / basal normal leakage": (
                signal / leakage if leakage > 0.0 else np.inf
            ),
            "minimum J": minimum_j,
            "first percentile of per-source minimum J": float(
                np.nanquantile(resolved[resolved_column], 0.01)
            )
            if len(resolved)
            else np.nan,
            "raw minimum J to exit": float(raw_row["raw minimum J to exit"]),
            "classification": (
                characteristic.classify(minimum_j)
                if status == "success"
                else "not evaluated"
            ),
            "source phase at minimum J": float(minimum_row["source phase"]),
            "age at minimum J (yr)": float(
                minimum_row["age of resolved minimum J (yr)"]
            ),
            "minimum path u (m/yr)": float(paths["minimum path u (m/yr)"].min()),
            "minimum path u / source u before reversal": float(
                paths["minimum path u / source u before reversal"].min()
            ),
            "minimum mechanism C before reversal": float(
                paths["minimum mechanism C before reversal"].min()
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
            "maximum velocity reversal count": int(
                paths["horizontal velocity reversal count"].max()
            ),
            "maximum relative streamfunction identity residual": float(
                paths["maximum relative streamfunction identity residual"].max()
            ),
            "maximum horizontal step (wavelengths)": float(
                paths["maximum horizontal step (wavelengths)"].max()
            ),
            "maximum exit age (yr)": float(paths["exit age (yr)"].max()),
            "surface exit count": int(exits.get("surface", 0)),
            "bed exit count": int(exits.get("bed", 0)),
            "safety-cap count": int(exits.get("safety cap", 0)),
            "numerical-failure count": int(exits.get("numerical failure", 0)),
        }
    except Exception as error:
        return base | {
            "status": f"failed: {type(error).__name__}: {error}",
            "minimum J": np.nan,
            "raw minimum J to exit": np.nan,
            "classification": "not evaluated",
        }


def sweep_geometries():
    for wavelength_ratio in (2.0, 4.0, 8.0, 12.0):
        for slope_ratio in (1.0, 2.0, 4.0, 8.0, 12.0):
            for thickness_ratio in (1.0, 1.5, 2.0, 3.0):
                yield IcefallTransitionGeometry(
                    wavelength=wavelength_ratio * DEFAULT_MEAN_THICKNESS,
                    mean_thickness=DEFAULT_MEAN_THICKNESS,
                    slope_ratio=slope_ratio,
                    thickness_ratio=thickness_ratio,
                    sharpness=DEFAULT_SHARPNESS,
                )


def run_sweep(output, resume=True):
    existing = pd.DataFrame()
    if resume and output.exists():
        existing = pd.read_csv(output)
    rows = []
    if len(existing):
        rows.extend(existing.to_dict("records"))
    completed = {
        (
            float(row["wavelength / mean thickness"]),
            float(row["steep / gentle surface-slope ratio"]),
            float(row["thick / thin thickness ratio"]),
        )
        for row in rows
        if str(row.get("status", "")).startswith(
            ("success", "analytic uniform-slab")
        )
    }
    for geometry in sweep_geometries():
        key = (
            geometry.wavelength_ratio,
            geometry.slope_ratio,
            geometry.thickness_ratio,
        )
        if key in completed:
            continue
        print(
            f"Solving L/H={key[0]:g}, slope ratio={key[1]:g}, "
            f"thickness ratio={key[2]:g}",
            flush=True,
        )
        row = solve_parameter_case(geometry)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(
            f"  {row['status']}; minimum J={row.get('minimum J', np.nan):.6g}",
            flush=True,
        )
    return pd.DataFrame(rows)


def run_validation(output):
    """Refine the production minimum and add one deliberately extreme audit."""
    cases = (
        (
            "production minimum",
            IcefallTransitionGeometry(
                2.0 * DEFAULT_MEAN_THICKNESS,
                DEFAULT_MEAN_THICKNESS,
                12.0,
                3.0,
                DEFAULT_SHARPNESS,
            ),
            {},
        ),
        (
            "production minimum: mesh and trace refinement",
            IcefallTransitionGeometry(
                2.0 * DEFAULT_MEAN_THICKNESS,
                DEFAULT_MEAN_THICKNESS,
                12.0,
                3.0,
                DEFAULT_SHARPNESS,
            ),
            {
                "mesh_nx": 128,
                "mesh_nz": 48,
                "sampler_nx": 641,
                "sampler_nz": 241,
                "source_count": 481,
                "time_step": 5.0,
            },
        ),
        (
            "extreme sharp-transition stress test",
            IcefallTransitionGeometry(
                4.0 * DEFAULT_MEAN_THICKNESS,
                DEFAULT_MEAN_THICKNESS,
                32.0,
                5.0,
                8,
            ),
            {
                "mesh_nx": 128,
                "mesh_nz": 28,
                "sampler_nx": 481,
                "sampler_nz": 161,
                "source_count": 241,
                "time_step": 10.0,
            },
        ),
        (
            "weak-signal failed-cell refinement",
            IcefallTransitionGeometry(
                8.0 * DEFAULT_MEAN_THICKNESS,
                DEFAULT_MEAN_THICKNESS,
                2.0,
                1.0,
                DEFAULT_SHARPNESS,
            ),
            {
                "sampler_nx": 641,
                "sampler_nz": 241,
                "source_count": 481,
                "time_step": 5.0,
            },
        ),
    )
    rows = []
    for label, geometry, options in cases:
        print(f"Solving validation case: {label}", flush=True)
        row = solve_parameter_case(geometry, **options)
        row["validation case"] = label
        rows.append(row)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(
            f"  {row['status']}; minimum J={row.get('minimum J', np.nan):.6g}",
            flush=True,
        )
    return pd.DataFrame(rows)


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--wavelength-ratio", type=float, default=4.0)
    parser.add_argument("--slope-ratio", type=float, default=8.0)
    parser.add_argument("--thickness-ratio", type=float, default=2.0)
    parser.add_argument("--sharpness", type=int, default=DEFAULT_SHARPNESS)
    parser.add_argument("--mesh-nx", type=int)
    parser.add_argument("--mesh-nz", type=int)
    parser.add_argument("--sampler-nx", type=int, default=DEFAULT_SAMPLER_NX)
    parser.add_argument("--sampler-nz", type=int, default=DEFAULT_SAMPLER_NZ)
    parser.add_argument("--source-count", type=int, default=DEFAULT_SOURCE_COUNT)
    parser.add_argument("--time-step", type=float, default=DEFAULT_TIME_STEP)
    parser.add_argument("--maximum-age", type=float, default=DEFAULT_MAXIMUM_AGE)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_icefall_transition.csv",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.validation:
        table = run_validation(args.output)
    elif args.sweep:
        table = run_sweep(args.output, resume=not args.no_resume)
    else:
        geometry = IcefallTransitionGeometry(
            wavelength=args.wavelength_ratio * DEFAULT_MEAN_THICKNESS,
            mean_thickness=DEFAULT_MEAN_THICKNESS,
            slope_ratio=args.slope_ratio,
            thickness_ratio=args.thickness_ratio,
            sharpness=args.sharpness,
        )
        table = pd.DataFrame(
            [
                solve_parameter_case(
                    geometry,
                    mesh_nx=args.mesh_nx,
                    mesh_nz=args.mesh_nz,
                    sampler_nx=args.sampler_nx,
                    sampler_nz=args.sampler_nz,
                    source_count=args.source_count,
                    time_step=args.time_step,
                    maximum_age=args.maximum_age,
                )
            ]
        )
        table.to_csv(args.output, index=False)
    print(table.to_string(index=False), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
