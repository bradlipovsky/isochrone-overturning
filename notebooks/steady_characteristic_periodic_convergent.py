"""Direct steady-characteristic test for a smooth periodic convergent basin.

This is the periodic control for the finite asymmetric convergent basin used
elsewhere in the repository.  One cell has total length ``L`` and constant
ice thickness ``H``.  Before smoothing, its common bed/surface relief is

    r(x) = Db cos^2(pi x / (2 L_left)),             0 <= x <= L_left,
           Db sin^2(pi (x-L_left) / (2 L_right)),   L_left < x < L,

with ``L_right/L_left=q`` and ``L_left+L_right=L``.  The two high endpoints
are identified periodically.  A wrapped-Gaussian convolution smooths the
continuation; a periodic cubic spline evaluates the result, which is shifted
and rescaled to retain exactly the requested peak-to-trough relief ``Db``.
The surface is ``s=r+H``.

Gravity is vertical, the upper surface is traction free, and the bed obeys
the same passive linear sliding law as the manuscript calculations.  In the
terrain-following coordinate ``zeta=z-s(x)`` the characteristic velocity is

    (u, w_tilde) = (u, w-u s_x).

This makes the periodic surface ``zeta=0`` and bed ``zeta=-H`` while leaving
the horizontal order diagnostic ``J=dx/dx0`` unchanged.  The direct tangent
ODE and the exact steady identity diagnostics are reused from
``steady_characteristic_sinusoidal_parameter_map``; no deposited cohort and
no time-dependent glacier simulation are involved.

Accumulation zones are found cyclically and then split wherever sampled
surface ``u`` changes sign.  This uses the actual flow divide rather than a
geometric maximum or the arbitrary coordinate seam, and is explicitly tested
under a quarter-period translation of the same geometry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
import traceback

import firedrake as fd
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid

import steady_characteristic_sinusoidal_parameter_map as characteristic

_SHARED_TRACE_SOURCES_TO_EXIT = characteristic.trace_sources_to_exit


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY / "data" / "steady_characteristic_periodic_convergent.csv"
)
DEFAULT_METADATA = (
    REPOSITORY
    / "data"
    / "steady_characteristic_periodic_convergent_metadata.json"
)

TOTAL_LENGTH = 6_000.0
THICKNESS = 200.0
PRODUCTION_MESH = (240, 32)
PRODUCTION_SAMPLER = (1201, 161)
PRODUCTION_SOURCE_COUNT = 481
PRODUCTION_RETAINED_FLUX_FRACTION = 0.90
PRODUCTION_TIME_STEP = 25.0
PRODUCTION_MAXIMUM_AGE = 300_000.0
SMOOTHING_FRACTIONS = (0.05, 0.10, 0.20, 0.40)
TARGET_CASES = ((1.0, 4.0), (8.0, 2.75), (12.0, 2.0))

METHOD_VERSION = "periodic-convergent-wrapped-gaussian-v9-flow-divide-split"
MINIMUM_SOURCE_BRANCH_FLUX_FRACTION = 1.0e-3
NORMAL_PENALTY_MULTIPLIER = 1.0
_LAST_TRAJECTORIES = None
_LAST_SOURCE_DIAGNOSTICS = None


@dataclass
class PeriodicConvergentGeometry:
    """Smooth periodic constant-thickness convergent-basin geometry."""

    length_ratio: float
    relief_ratio: float
    smoothing_fraction: float
    driving_slope: float = 0.0
    wavelength: float = TOTAL_LENGTH
    mean_thickness: float = THICKNESS
    phase_origin: float = 0.0
    fourier_sample_count: int = 32768
    _relief_spline: CubicSpline = field(init=False, repr=False)
    _slope_spline: CubicSpline = field(init=False, repr=False)
    _peak_slope: float = field(init=False, repr=False)
    _smoothing_translation: float = field(init=False, repr=False)

    def __post_init__(self):
        values = (
            self.length_ratio,
            self.relief_ratio,
            self.smoothing_fraction,
            self.driving_slope,
            self.wavelength,
            self.mean_thickness,
        )
        if not all(np.isfinite(values)):
            raise ValueError("geometry parameters must be finite")
        if self.length_ratio <= 0.0:
            raise ValueError("right/left length ratio must be positive")
        if self.relief_ratio <= 0.0:
            raise ValueError("relief/thickness ratio must be positive")
        if self.smoothing_fraction <= 0.0:
            raise ValueError("smoothing fraction must be positive")
        if abs(self.driving_slope) >= 0.5 * np.pi:
            raise ValueError("driving slope must lie between -pi/2 and pi/2")
        if self.wavelength <= 0.0 or self.mean_thickness <= 0.0:
            raise ValueError("length and thickness must be positive")
        n = int(self.fourier_sample_count)
        if n < 1024 or n % 2:
            raise ValueError("fourier_sample_count must be an even integer >=1024")

        x = np.arange(n, dtype=float) * self.wavelength / n
        raw = self._raw_relief(x)
        modes = 2.0 * np.pi * np.fft.rfftfreq(
            n, d=self.wavelength / n
        )
        sigma = self.smoothing_standard_deviation
        filtered = np.fft.rfft(raw) * np.exp(-0.5 * (modes * sigma) ** 2)
        smooth = np.fft.irfft(filtered, n=n)
        # Asymmetric convolution shifts the maximum away from the raw seam.
        # Translate the periodic result so x=0 is the *actual smoothed
        # divide* before splitting the two source branches there.
        maximum_index = int(np.argmax(smooth))
        self._smoothing_translation = float(x[maximum_index])
        smooth = np.roll(smooth, -maximum_index)
        span = float(np.max(smooth) - np.min(smooth))
        if span <= 0.0:
            raise ValueError("smoothed relief has zero range")
        smooth = self.relief * (smooth - np.min(smooth)) / span

        x_closed = np.r_[x, self.wavelength]
        y_closed = np.r_[smooth, smooth[0]]
        relief_spline = CubicSpline(x_closed, y_closed, bc_type="periodic")
        slope_spline = relief_spline.derivative()
        dense_x = np.linspace(0.0, self.wavelength, 8 * n, endpoint=False)
        self._relief_spline = relief_spline
        self._slope_spline = slope_spline
        self._peak_slope = float(np.max(np.abs(slope_spline(dense_x))))

    @property
    def left_length(self):
        return self.wavelength / (1.0 + self.length_ratio)

    @property
    def right_length(self):
        return self.wavelength - self.left_length

    @property
    def short_length(self):
        return min(self.left_length, self.right_length)

    @property
    def smoothing_standard_deviation(self):
        return self.smoothing_fraction * self.short_length

    @property
    def smoothing_translation(self):
        return self._smoothing_translation

    @property
    def relief(self):
        return self.relief_ratio * self.mean_thickness

    # Compatibility properties used by the shared characteristic summary.
    @property
    def amplitude(self):
        return self.relief

    @property
    def amplitude_ratio(self):
        return self.relief_ratio

    @property
    def wavelength_ratio(self):
        return self.wavelength / self.mean_thickness

    @property
    def minimum_thickness(self):
        return self.mean_thickness

    @property
    def maximum_thickness(self):
        return self.mean_thickness

    @property
    def peak_bed_slope(self):
        return self._peak_slope

    def _raw_relief(self, x):
        x = np.mod(np.asarray(x, dtype=float), self.wavelength)
        left = self.relief * np.cos(
            0.5 * np.pi * x / self.left_length
        ) ** 2
        right = self.relief * np.sin(
            0.5 * np.pi * (x - self.left_length) / self.right_length
        ) ** 2
        return np.where(x <= self.left_length, left, right)

    def relief_height(self, x):
        return np.asarray(
            self._relief_spline(
                np.mod(np.asarray(x) - self.phase_origin, self.wavelength)
            ),
            dtype=float,
        )

    def relief_slope(self, x):
        return np.asarray(
            self._slope_spline(
                np.mod(np.asarray(x) - self.phase_origin, self.wavelength)
            ),
            dtype=float,
        )

    def surface_height(self, x):
        return self.relief_height(x) + self.mean_thickness

    def bed_height(self, x):
        return self.relief_height(x)

    # In the transformed coordinate the local thickness is exactly H.
    def thickness(self, x):
        return np.full_like(np.asarray(x, dtype=float), self.mean_thickness)

    def thickness_slope(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))


def make_mesh(geometry, nx, nz, vertical_grading_power=1.0):
    """Map a periodic rectangle to parallel smooth bed and surface."""
    if vertical_grading_power != 1.0:
        raise ValueError("this constant-thickness study uses uniform vertical cells")
    mesh = fd.PeriodicRectangleMesh(
        int(nx),
        int(nz),
        geometry.wavelength,
        geometry.mean_thickness,
        direction="x",
    )
    coordinates = mesh.coordinates.dat.data_ro.copy()
    mesh.coordinates.dat.data[:, 1] = (
        coordinates[:, 1] + geometry.relief_height(coordinates[:, 0])
    )
    mesh.clear_spatial_index()
    return mesh


def represented_relief(x, geometry, mesh_nx):
    """Piecewise-linear relief represented by the periodic coordinate mesh."""
    node_x = np.arange(int(mesh_nx), dtype=float) * (
        geometry.wavelength / int(mesh_nx)
    )
    node_r = geometry.relief_height(node_x)
    return np.interp(
        np.mod(np.asarray(x, dtype=float), geometry.wavelength),
        np.r_[node_x, geometry.wavelength],
        np.r_[node_r, node_r[0]],
    )


def represented_relief_slope(x, geometry, mesh_nx):
    """Derivative of the coordinate mesh's piecewise-linear relief."""
    mesh_nx = int(mesh_nx)
    dx = geometry.wavelength / mesh_nx
    wrapped = np.mod(np.asarray(x, dtype=float), geometry.wavelength)
    left = np.floor(wrapped / dx).astype(int) % mesh_nx
    right = (left + 1) % mesh_nx
    node_x = np.arange(mesh_nx, dtype=float) * dx
    node_r = geometry.relief_height(node_x)
    return (node_r[right] - node_r[left]) / dx


def solve_stokes(geometry, nx, nz, vertical_grading_power=1.0):
    """Solve periodic Stokes flow under vertical gravity and linear sliding."""
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
            * np.sin(geometry.driving_slope),
            -characteristic.DENSITY
            * characteristic.GRAVITY
            * np.cos(geometry.driving_slope),
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
        characteristic.BASAL_DRAG
        * fd.dot(u, tangent)
        * fd.dot(v, tangent)
        - fd.dot(stress * normal, normal) * test_normal_speed
        - fd.dot(test_stress * normal, normal) * normal_speed
        + NORMAL_PENALTY_MULTIPLIER
        * characteristic.NITSCHE_PENALTY
        * characteristic.VISCOSITY
        / fd.CellDiameter(mesh)
        * normal_speed
        * test_normal_speed
    ) * fd.ds(1)
    fd.solve(residual == 0, solution, solver_parameters=characteristic.SOLVER_PARAMETERS)
    velocity, pressure = solution.subfunctions

    bed_length = fd.assemble(fd.Constant(1.0) * fd.ds(1, domain=mesh))
    basal_speed = fd.dot(velocity, tangent)
    physical_stress = (
        2.0 * characteristic.VISCOSITY * fd.sym(fd.grad(velocity))
        - pressure * fd.Identity(2)
    )
    gravity_power = fd.assemble(fd.dot(body_force, velocity) * fd.dx)
    basal_dissipation = fd.assemble(
        characteristic.BASAL_DRAG * basal_speed**2 * fd.ds(1)
    )
    viscous_dissipation = fd.assemble(
        2.0
        * characteristic.VISCOSITY
        * fd.inner(fd.sym(fd.grad(velocity)), fd.sym(fd.grad(velocity)))
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
            + NORMAL_PENALTY_MULTIPLIER
            * characteristic.NITSCHE_PENALTY
            * characteristic.VISCOSITY
            / fd.CellDiameter(mesh)
            * fd.dot(velocity, normal) ** 2
        )
        * fd.ds(1)
    )
    # n.ds on the parallel surface equals (w-u*s_x) dx.
    transformed_surface_flux = fd.assemble(fd.dot(velocity, normal) * fd.ds(2))
    velocity_data = velocity.dat.data_ro
    diagnostics = {
        "mesh cells": int(mesh.num_cells()),
        "gravity power": float(gravity_power),
        "basal dissipation / gravity power": float(
            basal_dissipation / gravity_power
        ),
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
        "surface vertical flux (m2/yr)": float(transformed_surface_flux),
        "mean surface vertical flux (m/yr)": float(
            transformed_surface_flux / geometry.wavelength
        ),
        "minimum nodal u (m/yr)": float(np.min(velocity_data[:, 0])),
        "maximum nodal u (m/yr)": float(np.max(velocity_data[:, 0])),
    }
    return mesh, velocity, diagnostics


def make_physical_variational_sampler_audit(
    geometry,
    mesh_nx,
    velocity,
    horizontal_samples=1201,
    vertical_samples=161,
    vertical_grading_power=1.0,
):
    """Return one bilinear interpolant of physical velocity and its gradient.

    Values are interpolated in ``(x,sigma)`` and both physical derivatives
    are obtained from that same interpolant.  Tracing in physical coordinates
    avoids differentiating the piecewise-linear mesh map inside the ODE: the
    transformed vertical velocity is discontinuous at coordinate-mesh
    columns even though the physical finite-element velocity is continuous.
    """
    if vertical_grading_power != 1.0:
        raise ValueError("this sampler uses uniform terrain-following height")
    if horizontal_samples < 8 or vertical_samples < 4:
        raise ValueError("sampler needs at least 8 x 4 points")
    dx = geometry.wavelength / horizontal_samples
    x_grid = (np.arange(horizontal_samples) + 0.5) * dx
    sigma_grid = np.linspace(
        characteristic.BED_SIGMA,
        characteristic.SURFACE_SIGMA,
        vertical_samples,
    )
    X = np.tile(x_grid, (vertical_samples, 1))
    physical_z = represented_relief(X, geometry, mesh_nx) + (
        sigma_grid[:, None] * geometry.mean_thickness
    )
    points = np.column_stack((X.ravel(), physical_z.ravel()))
    physical = np.asarray(
        velocity.at(points, tolerance=1.0e-7), dtype=float
    ).reshape(vertical_samples, horizontal_samples, 2)
    values = physical

    def interpolate(points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        wrapped = np.mod(points[:, 0], geometry.wavelength)
        local_relief = represented_relief(wrapped, geometry, mesh_nx)
        sigma = (points[:, 1] - local_relief) / geometry.mean_thickness
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
        value = (
            ((1.0 - fs) * (1.0 - fx))[:, None] * v00
            + ((1.0 - fs) * fx)[:, None] * v01
            + (fs * (1.0 - fx))[:, None] * v10
            + (fs * fx)[:, None] * v11
        )
        derivative_sigma = (
            (1.0 - fx)[:, None] * (v10 - v00)
            + fx[:, None] * (v11 - v01)
        ) / ds[:, None]
        derivative_x = (
            (1.0 - fs)[:, None] * (v01 - v00)
            + fs[:, None] * (v11 - v10)
        ) / dx
        gradient = np.empty((len(points), 2, 2))
        local_slope = represented_relief_slope(
            wrapped, geometry, mesh_nx
        )
        gradient[:, :, 0] = (
            derivative_x
            - derivative_sigma
            * (local_slope / geometry.mean_thickness)[:, None]
        )
        gradient[:, :, 1] = derivative_sigma / geometry.mean_thickness
        return value, gradient

    def sample(name, x, z):
        x, z = np.broadcast_arrays(x, z)
        if name == "a":
            wrapped = np.mod(x.ravel(), geometry.wavelength)
            surface = (
                represented_relief(wrapped, geometry, mesh_nx)
                + geometry.mean_thickness
            )
            query = np.column_stack((wrapped, surface))
            velocity_value = interpolate(query)[0]
            slope_value = represented_relief_slope(
                wrapped, geometry, mesh_nx
            )
            result = (
                velocity_value[:, 0] * slope_value - velocity_value[:, 1]
            ).reshape(x.shape)
        else:
            query = np.column_stack((x.ravel(), z.ravel()))
            component = 0 if name == "u" else 1
            result = interpolate(query)[0][:, component].reshape(x.shape)
        return float(result) if result.ndim == 0 else result

    diagnostics = {
        "minimum sampled u (m/yr)": float(np.nanmin(values[..., 0])),
        "maximum sampled u (m/yr)": float(np.nanmax(values[..., 0])),
        "minimum sampled w (m/yr)": float(np.nanmin(values[..., 1])),
        "maximum sampled w (m/yr)": float(np.nanmax(values[..., 1])),
    }
    return (
        sample,
        lambda points: interpolate(points)[0],
        lambda points: interpolate(points)[1],
        diagnostics,
    )


def make_variational_sampler(
    geometry,
    mesh_nx,
    velocity,
    horizontal_samples=1201,
    vertical_samples=161,
    vertical_grading_power=1.0,
):
    """Build a divergence-free terrain-coordinate velocity from streamfunction.

    The finite-element horizontal velocity is sampled on a regular
    ``(x,sigma)`` grid and integrated upward from the impermeable bed,

        Psi(x,sigma) = H integral u(x,sigma') d sigma'.

    A periodic Fourier representation in x and cubic representation in sigma
    then define ``u=Psi_sigma/H`` and ``W=-Psi_x``.  Thus
    ``u_x+W_zeta=0`` and ``W=0`` at the numerical bed exactly.  This removes
    the tiny Nitsche normal leakage that otherwise sends an isolated, very
    old separatrix trajectory through the bed and corrupts its tangent.
    """
    if vertical_grading_power != 1.0:
        raise ValueError("streamfunction sampler uses uniform terrain height")
    if horizontal_samples < 8 or vertical_samples < 4:
        raise ValueError("sampler needs at least 8 x 4 points")
    horizontal_samples = int(horizontal_samples)
    vertical_samples = int(vertical_samples)
    dx = geometry.wavelength / horizontal_samples
    x_grid = (np.arange(horizontal_samples) + 0.5) * dx
    sigma_grid = np.linspace(
        characteristic.BED_SIGMA,
        characteristic.SURFACE_SIGMA,
        vertical_samples,
    )
    X = np.tile(x_grid, (vertical_samples, 1))
    physical_z = represented_relief(X, geometry, mesh_nx) + (
        sigma_grid[:, None] * geometry.mean_thickness
    )
    points = np.column_stack((X.ravel(), physical_z.ravel()))
    physical = np.asarray(
        velocity.at(points, tolerance=1.0e-7), dtype=float
    ).reshape(vertical_samples, horizontal_samples, 2)
    sampled_u = physical[..., 0]
    streamfunction = np.zeros_like(sampled_u)
    streamfunction[1:] = geometry.mean_thickness * cumulative_trapezoid(
        sampled_u, sigma_grid, axis=0
    )

    coefficients = np.fft.fft(streamfunction, axis=1) / horizontal_samples
    integer_modes = np.fft.fftfreq(horizontal_samples) * horizontal_samples
    maximum_mode = min(max(8, int(mesh_nx) // 2), (horizontal_samples - 1) // 2)
    keep = np.abs(integer_modes) <= maximum_mode
    coefficients = coefficients[:, keep]
    integer_modes = integer_modes[keep]
    wave_numbers = 2.0 * np.pi * integer_modes / geometry.wavelength
    coefficient_spline = CubicSpline(sigma_grid, coefficients, axis=0)

    def interpolate(points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        wrapped = np.mod(points[:, 0], geometry.wavelength)
        sigma = 1.0 + points[:, 1] / geometry.mean_thickness
        sigma = np.clip(
            sigma,
            characteristic.BED_SIGMA,
            characteristic.SURFACE_SIGMA,
        )
        # DFT samples live at (j+1/2) dx, hence the half-cell phase shift.
        phase = np.exp(
            1j
            * (wrapped - 0.5 * dx)[:, None]
            * wave_numbers[None, :]
        )
        c0 = coefficient_spline(sigma, 0)
        c1 = coefficient_spline(sigma, 1)
        c2 = coefficient_spline(sigma, 2)
        h = geometry.mean_thickness
        u = np.real(np.sum(c1 * phase, axis=1)) / h
        transformed_w = np.real(
            np.sum((-1j * wave_numbers)[None, :] * c0 * phase, axis=1)
        )
        gradient = np.empty((len(points), 2, 2))
        gradient[:, 0, 0] = np.real(
            np.sum((1j * wave_numbers)[None, :] * c1 * phase, axis=1)
        ) / h
        gradient[:, 0, 1] = np.real(np.sum(c2 * phase, axis=1)) / h**2
        gradient[:, 1, 0] = np.real(
            np.sum((wave_numbers**2)[None, :] * c0 * phase, axis=1)
        )
        gradient[:, 1, 1] = np.real(
            np.sum((-1j * wave_numbers)[None, :] * c1 * phase, axis=1)
        ) / h
        return np.column_stack((u, transformed_w)), gradient

    def sample(name, x, zeta):
        x, zeta = np.broadcast_arrays(x, zeta)
        query = np.column_stack((x.ravel(), zeta.ravel()))
        component = 0 if name == "u" else 1
        result = interpolate(query)[0][:, component].reshape(x.shape)
        return float(result) if result.ndim == 0 else result

    # Quantify the correction relative to direct physical vertical sampling.
    full_coefficients = np.zeros(
        (vertical_samples, horizontal_samples), dtype=complex
    )
    full_coefficients[:, keep] = coefficients
    full_wave_numbers = (
        2.0 * np.pi * np.fft.fftfreq(horizontal_samples, d=dx)
    )
    reconstructed_w = np.fft.ifft(
        (-1j * full_wave_numbers[None, :])
        * full_coefficients
        * horizontal_samples,
        axis=1,
    ).real
    slope = represented_relief_slope(x_grid, geometry, mesh_nx)[None, :]
    direct_transformed_w = physical[..., 1] - physical[..., 0] * slope
    top_mismatch = reconstructed_w[-1] - direct_transformed_w[-1]
    diagnostics = {
        "minimum sampled u (m/yr)": float(np.nanmin(sampled_u)),
        "maximum sampled u (m/yr)": float(np.nanmax(sampled_u)),
        "minimum sampled w (m/yr)": float(np.nanmin(reconstructed_w)),
        "maximum sampled w (m/yr)": float(np.nanmax(reconstructed_w)),
        "streamfunction retained Fourier mode": int(maximum_mode),
        "rms reconstructed/direct surface-normal mismatch (m/yr)": float(
            np.sqrt(np.mean(top_mismatch**2))
        ),
        "maximum reconstructed bed-normal speed (m/yr)": float(
            np.max(np.abs(reconstructed_w[0]))
        ),
    }
    return (
        sample,
        lambda points: interpolate(points)[0],
        lambda points: interpolate(points)[1],
        diagnostics,
    )


def retained_surface_sources(
    geometry,
    sample,
    source_count=481,
    retained_flux_fraction=0.90,
    start_depth=0.0,
):
    """Trim cyclic accumulation zones separately on each flow-divide side."""
    if not 0.0 < retained_flux_fraction <= 1.0:
        raise ValueError("retained flux fraction must be in (0,1]")
    candidates = np.linspace(
        0.0, geometry.wavelength, int(source_count), endpoint=False
    )
    accumulation = -sample(
        "w", candidates, np.full(len(candidates), -start_depth)
    )
    tolerance = max(1.0e-10, 1.0e-8 * np.nanmax(np.abs(accumulation)))
    mask = np.isfinite(accumulation) & (accumulation > tolerance)
    depositing_indices = np.flatnonzero(mask)
    # First find accumulation zones cyclically, independent of the arbitrary
    # coordinate seam.  Then split each zone wherever sampled surface u
    # changes sign: that is the *flow* divide, which need not coincide with
    # the geometric maximum after smoothing or with x=0.
    accumulation_runs = characteristic._cyclic_true_runs(mask)
    surface_u = sample(
        "u", candidates, np.zeros(len(candidates), dtype=float)
    )
    runs = []
    for run in accumulation_runs:
        local_u = surface_u[run]
        cuts = np.flatnonzero(
            np.signbit(local_u[1:]) != np.signbit(local_u[:-1])
        ) + 1
        runs.extend(group for group in np.split(run, cuts) if len(group))
    # Pointwise interpolation noise can create one- to five-sample positive
    # islands inside the broad ablation zone.  They are not accumulation
    # basins and previously launched pathological near-bed trajectories.  A
    # branch must carry at least 0.1% of total positive sampled flux; both
    # physical seam branches exceed this threshold by orders of magnitude.
    positive_flux = float(np.sum(np.maximum(accumulation, 0.0)))
    runs = [
        run
        for run in runs
        if np.sum(accumulation[run])
        >= MINIMUM_SOURCE_BRANCH_FLUX_FRACTION * positive_flux
    ]
    source_family_indices = (
        np.concatenate(runs).astype(int) if len(runs) else np.array([], dtype=int)
    )
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
    dx = geometry.wavelength / source_count
    net_flux = float(dx * np.sum(accumulation))
    gross_flux = float(dx * np.sum(np.abs(accumulation)))
    result = {
        "retained sources": candidates[retained],
        "depositing sources": candidates[source_family_indices],
        "candidate sources": candidates,
        "accumulation": accumulation,
        "accumulation interval count": len(runs),
        "cyclic accumulation-zone count": len(accumulation_runs),
        "sampled surface-flow divide count": int(
            sum(
                np.count_nonzero(
                    np.signbit(surface_u[run][1:])
                    != np.signbit(surface_u[run][:-1])
                )
                for run in accumulation_runs
            )
        ),
        "accumulation tolerance": tolerance,
        "sampled net surface mass balance (m2/yr)": net_flux,
        "sampled gross surface mass exchange (m2/yr)": gross_flux,
        "sampled relative surface mass balance closure": (
            abs(net_flux) / gross_flux if gross_flux > 0.0 else 0.0
        ),
    }
    global _LAST_SOURCE_DIAGNOSTICS
    _LAST_SOURCE_DIAGNOSTICS = {
        "cyclic accumulation-zone count": len(accumulation_runs),
        "sampled surface-flow divide count": result[
            "sampled surface-flow divide count"
        ],
        "retained flow-directed source-family count": len(runs),
    }
    return result


def local_sigma(points, geometry, mesh_nx):
    """Terrain-following coordinate of physical points, with unwrapped x."""
    points = np.asarray(points, dtype=float)
    return (
        points[:, 1]
        - represented_relief(points[:, 0], geometry, mesh_nx)
    ) / geometry.mean_thickness


def surface_points(x, geometry, mesh_nx):
    """Physical points on the piecewise-curved finite-element surface."""
    x = np.asarray(x, dtype=float)
    return np.column_stack(
        (
            x,
            represented_relief(x, geometry, mesh_nx)
            + geometry.mean_thickness,
        )
    )


def surface_accumulation(x, geometry, mesh_nx, sample_vector):
    """Required steady mass balance ``a=u s_x-w`` on the surface."""
    x = np.asarray(x, dtype=float)
    velocity = sample_vector(surface_points(x, geometry, mesh_nx))
    slope = represented_relief_slope(x, geometry, mesh_nx)
    return velocity[:, 0] * slope - velocity[:, 1]


def trace_sources_physical_audit(
    geometry,
    mesh_nx,
    sample_vector,
    sample_gradient,
    sources,
    time_step=25.0,
    maximum_age=300_000.0,
    start_depth=0.0,
    surface_accumulation_tolerance=0.0,
):
    """Trace physical characteristics and their equal-age tangents to exit.

    The initial material tangent is the actual surface tangent
    ``(1,s_x)``.  Consequently ``J=T_x`` is exactly the same horizontal map
    derivative as in the finite convergent calculation, while the steady
    streamfunction identity is ``-w J + u T_z = a(x0)``.
    """
    if start_depth != 0.0:
        raise ValueError("physical periodic sources are launched on the surface")
    if time_step <= 0.0 or maximum_age <= 0.0:
        raise ValueError("time step and maximum age must be positive")
    sources = np.asarray(sources, dtype=float)
    count = len(sources)
    points = surface_points(sources, geometry, mesh_nx)
    tangents = np.column_stack(
        (
            np.ones(count),
            represented_relief_slope(sources, geometry, mesh_nx),
        )
    )
    active = np.ones(count, dtype=bool)
    age = np.zeros(count)
    raw_minimum_j = np.ones(count)
    raw_minimum_age = np.zeros(count)
    buffered_minimum_j = np.ones(count)
    buffered_minimum_age = np.zeros(count)
    first_zero_age = np.full(count, np.nan)
    first_zero_surface_clearance = np.full(count, np.nan)
    maximum_negative_surface_clearance = np.full(count, np.nan)
    exit_age = np.full(count, np.nan)
    exit_kind = np.full(count, "active", dtype=object)
    source_velocity = sample_vector(points)
    source_u = source_velocity[:, 0]
    source_accumulation = (
        source_u
        * represented_relief_slope(sources, geometry, mesh_nx)
        - source_velocity[:, 1]
    )
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
        old_sigma = local_sigma(old_points, geometry, mesh_nx)
        candidate_points, candidate_tangents = characteristic.rk4_step(
            sample_vector,
            sample_gradient,
            old_points,
            old_tangents,
            step,
        )
        candidate_sigma = local_sigma(candidate_points, geometry, mesh_nx)
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
            first_zero_surface_clearance[selected] = 1.0 - local_sigma(
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

        local_surface_accumulation = surface_accumulation(
            accepted_points[:, 0], geometry, mesh_nx, sample_vector
        )
        resolved = (accepted_sigma <= characteristic.ANALYSIS_SURFACE_SIGMA) | (
            local_surface_accumulation >= -surface_accumulation_tolerance
        )
        improves_buffered = resolved & (
            accepted_j < buffered_minimum_j[indices]
        )
        if np.any(improves_buffered):
            selected = indices[improves_buffered]
            buffered_minimum_j[selected] = accepted_j[improves_buffered]
            buffered_minimum_age[selected] = accepted_age[improves_buffered]
            buffered_speed_ratio[selected] = np.where(
                mechanism_valid[improves_buffered],
                speed_ratio[improves_buffered],
                np.nan,
            )
            buffered_mechanism_c[selected] = np.where(
                mechanism_valid[improves_buffered],
                mechanism_c[improves_buffered],
                np.nan,
            )
            buffered_reversal[selected] = reversals[improves_buffered] > 0

        negative = accepted_j < 0.0
        if np.any(negative):
            selected = indices[negative]
            maximum_negative_surface_clearance[selected] = np.fmax(
                maximum_negative_surface_clearance[selected],
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
    trajectories = pd.DataFrame(
        {
            "source x0 (m)": sources,
            "source phase": sources / geometry.wavelength,
            "minimum J outside ablating-surface buffer": buffered_minimum_j,
            "age of resolved minimum J (yr)": buffered_minimum_age,
            "raw minimum J to exit": raw_minimum_j,
            "age of raw minimum J (yr)": raw_minimum_age,
            "first J=0 age (yr)": first_zero_age,
            "surface-clearance fraction at first J=0": first_zero_surface_clearance,
            "maximum surface-clearance fraction while J<0": (
                maximum_negative_surface_clearance
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
            "source accumulation (m/yr)": source_accumulation,
        }
    )
    global _LAST_TRAJECTORIES
    _LAST_TRAJECTORIES = trajectories
    return trajectories


def trace_sources_to_exit(*args, **kwargs):
    """Record the shared transformed-coordinate trajectory table for audits."""
    trajectories = _SHARED_TRACE_SOURCES_TO_EXIT(*args, **kwargs)
    geometry = args[0]
    sample_vector = args[2]
    sources = np.asarray(args[4], dtype=float)
    source_points = np.column_stack((sources, np.zeros(len(sources))))
    trajectories["source accumulation (m/yr)"] = -sample_vector(
        source_points
    )[:, 1]
    global _LAST_TRAJECTORIES
    _LAST_TRAJECTORIES = trajectories
    return trajectories


def solve_case(task):
    """Solve one task while temporarily installing the periodic geometry hooks."""
    global _LAST_TRAJECTORIES, _LAST_SOURCE_DIAGNOSTICS
    _LAST_TRAJECTORIES = None
    _LAST_SOURCE_DIAGNOSTICS = None
    (
        run_kind,
        q,
        relief_ratio,
        smoothing_fraction,
        driving_slope,
        phase_origin_fraction,
        mesh_nx,
        mesh_nz,
        sampler_nx,
        sampler_nz,
        source_count,
        time_step,
        maximum_age,
    ) = task
    geometry = PeriodicConvergentGeometry(
        q,
        relief_ratio,
        smoothing_fraction,
        driving_slope,
        phase_origin=phase_origin_fraction * TOTAL_LENGTH,
    )
    originals = (
        characteristic.solve_stokes,
        characteristic.make_variational_sampler,
        characteristic.retained_surface_sources,
        characteristic.trace_sources_to_exit,
    )
    characteristic.solve_stokes = solve_stokes
    characteristic.make_variational_sampler = make_variational_sampler
    characteristic.retained_surface_sources = retained_surface_sources
    characteristic.trace_sources_to_exit = trace_sources_to_exit
    started = perf_counter()
    try:
        row = characteristic.solve_parameter_case(
            geometry,
            mesh_nx=mesh_nx,
            mesh_nz=mesh_nz,
            sampler_nx=sampler_nx,
            sampler_nz=sampler_nz,
            source_count=source_count,
            retained_flux_fraction=PRODUCTION_RETAINED_FLUX_FRACTION,
            time_step=time_step,
            maximum_age=maximum_age,
            minimum_surface_signal_to_basal_leakage=None,
        )
    except Exception as error:
        traceback.print_exc()
        row = {
            "status": f"failed: {type(error).__name__}: {error}",
            "minimum J": np.nan,
            "raw minimum J to exit": np.nan,
        }
    finally:
        (
            characteristic.solve_stokes,
            characteristic.make_variational_sampler,
            characteristic.retained_surface_sources,
            characteristic.trace_sources_to_exit,
        ) = originals
    trajectories = _LAST_TRAJECTORIES
    if trajectories is not None and len(trajectories):
        identity_column = "maximum relative streamfunction identity residual"
        row["per-source identity residual 99th percentile"] = float(
            trajectories[identity_column].quantile(0.99)
        )
        phase = float(row.get("source phase at minimum J", np.nan))
        if np.isfinite(phase):
            selected = trajectories.iloc[
                np.argmin(np.abs(trajectories["source phase"] - phase))
            ]
            row["identity residual on minimum-J source"] = float(
                selected[identity_column]
            )
            if "source accumulation (m/yr)" in selected.index:
                row["minimum-J source accumulation (m/yr)"] = float(
                    selected["source accumulation (m/yr)"]
                )
        resolved_column = "minimum J outside ablating-surface buffer"
        source_flux = np.maximum(
            trajectories["source accumulation (m/yr)"].to_numpy(float), 0.0
        )
        total_flux = float(np.sum(source_flux))

        def add_negative_band(prefix, mask):
            mask = np.asarray(mask, dtype=bool)
            row[f"{prefix} retained-source count"] = int(np.count_nonzero(mask))
            row[f"{prefix} retained accumulation-flux fraction"] = (
                float(np.sum(source_flux[mask]) / total_flux)
                if total_flux > 0.0
                else np.nan
            )
            phases = np.sort(
                trajectories.loc[mask, "source phase"].to_numpy(float) % 1.0
            )
            if not len(phases):
                row[f"{prefix} source phase width (periods)"] = 0.0
                row[f"{prefix} source phase arc start"] = np.nan
                row[f"{prefix} source phase arc end"] = np.nan
            elif len(phases) == 1:
                row[f"{prefix} source phase width (periods)"] = 0.0
                row[f"{prefix} source phase arc start"] = float(phases[0])
                row[f"{prefix} source phase arc end"] = float(phases[0])
            else:
                gaps = np.diff(np.r_[phases, phases[0] + 1.0])
                largest = int(np.argmax(gaps))
                start = float(phases[(largest + 1) % len(phases)])
                end = float(phases[largest])
                row[f"{prefix} source phase width (periods)"] = float(
                    1.0 - gaps[largest]
                )
                row[f"{prefix} source phase arc start"] = start
                row[f"{prefix} source phase arc end"] = end

        add_negative_band(
            "negative-J",
            trajectories[resolved_column].to_numpy(float) < 0.0,
        )
        add_negative_band(
            "resolved-overturn",
            trajectories[resolved_column].to_numpy(float) < -0.05,
        )
    if _LAST_SOURCE_DIAGNOSTICS is not None:
        row.update(_LAST_SOURCE_DIAGNOSTICS)
    row.update(
        {
            "diagnostic method version": METHOD_VERSION,
            "run kind": run_kind,
            "right / left length ratio": q,
            "bed relief / ice thickness": relief_ratio,
            "left length (m)": geometry.left_length,
            "right length (m)": geometry.right_length,
            "total periodic length (m)": geometry.wavelength,
            "constant ice thickness (m)": geometry.mean_thickness,
            "wrapped-Gaussian sigma / short flank length": smoothing_fraction,
            "mean driving slope (rad)": driving_slope,
            "coordinate gauge phase / period": phase_origin_fraction,
            "wrapped-Gaussian sigma (m)": geometry.smoothing_standard_deviation,
            "translation placing smoothed divide at seam (m)": (
                geometry.smoothing_translation
            ),
            "renormalized peak-to-trough relief (m)": geometry.relief,
            "peak absolute smoothed slope": geometry.peak_bed_slope,
            "body force geometry": (
                "gravity with stated mean driving slope; parallel periodic "
                "bed and surface"
            ),
            "periodic horizontal boundary": True,
            "minimum retained source-branch flux fraction": (
                MINIMUM_SOURCE_BRANCH_FLUX_FRACTION
            ),
            "normal Nitsche penalty multiplier": NORMAL_PENALTY_MULTIPLIER,
            "surface slope (rad)": np.nan,
            "runtime including geometry (s)": perf_counter() - started,
        }
    )
    return row


def production_tasks(maximum_age):
    tasks = []
    for q, relief_ratio in TARGET_CASES:
        for smoothing in SMOOTHING_FRACTIONS:
            tasks.append(
                (
                    "production",
                    q,
                    relief_ratio,
                    smoothing,
                    0.0,
                    0.0,
                    *PRODUCTION_MESH,
                    *PRODUCTION_SAMPLER,
                    PRODUCTION_SOURCE_COUNT,
                    PRODUCTION_TIME_STEP,
                    maximum_age,
                )
            )
    # Coarse/fine mesh pairs at the representative sigma/L_short=0.10.
    for q, relief_ratio in TARGET_CASES:
        tasks.extend(
            [
                (
                    "mesh coarse",
                    q,
                    relief_ratio,
                    0.10,
                    0.0,
                    0.0,
                    180,
                    24,
                    901,
                    121,
                    PRODUCTION_SOURCE_COUNT,
                    PRODUCTION_TIME_STEP,
                    maximum_age,
                ),
                (
                    "mesh fine",
                    q,
                    relief_ratio,
                    0.10,
                    0.0,
                    0.0,
                    320,
                    40,
                    1601,
                    201,
                    PRODUCTION_SOURCE_COUNT,
                    PRODUCTION_TIME_STEP,
                    maximum_age,
                ),
            ]
        )
    # The narrowest smoothing width in the strongest-asymmetry target gets
    # one combined characteristic/sampler/source refinement as well.
    tasks.append(
        (
            "focused full refinement",
            12.0,
            2.0,
            0.05,
            0.0,
            0.0,
            360,
            48,
            1801,
            241,
            721,
            12.5,
            maximum_age,
        )
    )
    # Any negative isolated minimum seen on the production mesh must pass a
    # dedicated refinement sequence before it can be interpreted physically.
    for run_kind, q, relief_ratio, smoothing, nx, nz, snx, snz, sources, dt in (
        ("negative audit coarse", 8.0, 2.75, 0.20, 180, 24, 901, 121, 481, 25.0),
        ("negative audit fine", 8.0, 2.75, 0.20, 320, 40, 1601, 201, 481, 25.0),
        ("negative audit full", 8.0, 2.75, 0.20, 360, 48, 1801, 241, 721, 12.5),
        ("negative audit full", 8.0, 2.75, 0.40, 360, 48, 1801, 241, 721, 12.5),
        ("negative audit full", 12.0, 2.0, 0.10, 360, 48, 1801, 241, 721, 12.5),
    ):
        tasks.append(
            (
                run_kind,
                q,
                relief_ratio,
                smoothing,
                0.0,
                0.0,
                nx,
                nz,
                snx,
                snz,
                sources,
                dt,
                maximum_age,
            )
        )
    # A mean downslope body-force component continuously converts the
    # two-sided basin into one-directional periodic throughflow without
    # changing the smooth bed.  Use the stable q=8, eps=0.10 geometry.
    for slope in (
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.20,
        0.30,
    ):
        tasks.append(
            (
                "mean-slope continuation",
                8.0,
                2.75,
                0.10,
                slope,
                0.0,
                *PRODUCTION_MESH,
                *PRODUCTION_SAMPLER,
                PRODUCTION_SOURCE_COUNT,
                PRODUCTION_TIME_STEP,
                maximum_age,
            )
        )
    tasks.append(
        (
            "mean-slope fine",
            8.0,
            2.75,
            0.10,
            0.02,
            0.0,
            320,
            40,
            1601,
            201,
            PRODUCTION_SOURCE_COUNT,
            PRODUCTION_TIME_STEP,
            maximum_age,
        )
    )
    for run_kind, slope, nx, nz, snx, snz, sources, dt in (
        ("mean-slope audit coarse", 0.10, 180, 24, 901, 121, 481, 25.0),
        ("mean-slope audit fine", 0.10, 320, 40, 1601, 201, 481, 25.0),
        ("mean-slope audit full", 0.10, 360, 48, 1801, 241, 721, 12.5),
        ("mean-slope audit fine", 0.05, 320, 40, 1601, 201, 481, 25.0),
        ("mean-slope audit full", 0.05, 360, 48, 1801, 241, 721, 12.5),
        ("mean-slope audit ultra", 0.10, 480, 64, 2401, 321, 961, 6.25),
        ("mean-slope audit fine", 0.12, 320, 40, 1601, 201, 481, 25.0),
        ("mean-slope audit full", 0.12, 360, 48, 1801, 241, 721, 12.5),
        ("mean-slope audit fine", 0.15, 320, 40, 1601, 201, 481, 25.0),
        ("mean-slope audit fine", 0.20, 320, 40, 1601, 201, 481, 25.0),
    ):
        tasks.append(
            (
                run_kind,
                8.0,
                2.75,
                0.10,
                slope,
                0.0,
                nx,
                nz,
                snx,
                snz,
                sources,
                dt,
                maximum_age,
            )
        )
    for run_kind, q, relief, smoothing, nx, nz, snx, snz, sources, dt in (
        ("retargeted audit coarse", 8.0, 2.75, 0.40, 180, 24, 901, 121, 481, 25.0),
        ("retargeted audit fine", 8.0, 2.75, 0.40, 320, 40, 1601, 201, 481, 25.0),
        ("retargeted audit fine", 12.0, 2.0, 0.05, 320, 40, 1601, 201, 481, 25.0),
        ("retargeted audit coarse", 12.0, 2.0, 0.20, 180, 24, 901, 121, 481, 25.0),
        ("retargeted audit fine", 12.0, 2.0, 0.20, 320, 40, 1601, 201, 481, 25.0),
        ("retargeted audit full", 12.0, 2.0, 0.20, 360, 48, 1801, 241, 721, 12.5),
        ("retargeted audit coarse", 12.0, 2.0, 0.40, 180, 24, 901, 121, 481, 25.0),
        ("retargeted audit fine", 12.0, 2.0, 0.40, 320, 40, 1601, 201, 481, 25.0),
        ("retargeted audit full", 12.0, 2.0, 0.40, 360, 48, 1801, 241, 721, 12.5),
    ):
        tasks.append(
            (
                run_kind,
                q,
                relief,
                smoothing,
                0.0,
                0.0,
                nx,
                nz,
                snx,
                snz,
                sources,
                dt,
                maximum_age,
            )
        )
    tasks.append(
        (
            "coordinate gauge audit",
            8.0,
            2.75,
            0.10,
            0.0,
            0.25,
            *PRODUCTION_MESH,
            *PRODUCTION_SAMPLER,
            PRODUCTION_SOURCE_COUNT,
            PRODUCTION_TIME_STEP,
            maximum_age,
        )
    )
    for source_count in (481, 721, 961, 1441):
        tasks.append(
            (
                "q12 source-density audit",
                12.0,
                2.0,
                0.10,
                0.0,
                0.0,
                360,
                48,
                1801,
                241,
                source_count,
                12.5,
                maximum_age,
            )
        )
    tasks.extend(
        [
            (
                "q12 ultra mesh audit",
                12.0,
                2.0,
                0.10,
                0.0,
                0.0,
                480,
                64,
                2401,
                321,
                961,
                6.25,
                maximum_age,
            ),
            (
                "q12 coordinate gauge audit",
                12.0,
                2.0,
                0.10,
                0.0,
                0.25,
                240,
                32,
                1201,
                161,
                481,
                25.0,
                maximum_age,
            ),
            (
                "q12 coordinate gauge full",
                12.0,
                2.0,
                0.10,
                0.0,
                0.25,
                360,
                48,
                1801,
                241,
                721,
                12.5,
                maximum_age,
            ),
            (
                "mean-slope low audit fine",
                8.0,
                2.75,
                0.10,
                0.01,
                0.0,
                320,
                40,
                1601,
                201,
                481,
                25.0,
                maximum_age,
            ),
            (
                "mean-slope low audit full",
                8.0,
                2.75,
                0.10,
                0.01,
                0.0,
                360,
                48,
                1801,
                241,
                721,
                12.5,
                maximum_age,
            ),
        ]
    )
    for source_count in (481, 721, 961, 1441):
        tasks.append(
            (
                "alpha10 source-density audit",
                8.0,
                2.75,
                0.10,
                0.10,
                0.0,
                360,
                48,
                1801,
                241,
                source_count,
                12.5,
                maximum_age,
            )
        )
    tasks.append(
        (
            "alpha10 coordinate gauge full",
            8.0,
            2.75,
            0.10,
            0.10,
            0.25,
            360,
            48,
            1801,
            241,
            721,
            12.5,
            maximum_age,
        )
    )
    return tasks


def quick_tasks(maximum_age):
    return [
        (
            "quick",
            8.0,
            2.75,
            0.10,
            0.0,
            0.0,
            96,
            16,
            481,
            81,
            241,
            50.0,
            maximum_age,
        )
    ]


def task_key_from_task(task):
    return tuple(
        round(float(value), 12) if isinstance(value, (float, np.floating)) else value
        for value in task[:-1]
    )


def task_key_from_row(row):
    return (
        str(row["run kind"]),
        round(float(row["right / left length ratio"]), 12),
        round(float(row["bed relief / ice thickness"]), 12),
        round(
            float(row["wrapped-Gaussian sigma / short flank length"]), 12
        ),
        round(float(row["mean driving slope (rad)"]), 12),
        round(float(row["coordinate gauge phase / period"]), 12),
        int(row["mesh nx"]),
        int(row["mesh nz"]),
        int(row["sampler nx"]),
        int(row["sampler nz"]),
        int(row["candidate source count"]),
        round(float(row["characteristic time step (yr)"]), 12),
    )


def sort_table(rows):
    return pd.DataFrame(rows).sort_values(
        [
            "right / left length ratio",
            "bed relief / ice thickness",
            "wrapped-Gaussian sigma / short flank length",
            "mean driving slope (rad)",
            "coordinate gauge phase / period",
            "mesh nx",
            "run kind",
        ]
    ).reset_index(drop=True)


def write_metadata(table, output, metadata):
    successful = table.loc[table["status"].eq("success")].copy()
    record = {
        "method": METHOD_VERSION,
        "description": (
            "Smooth wrapped-Gaussian periodic continuation of the finite "
            "piecewise-sin^2 convergent basin; direct steady characteristics"
        ),
        "data": (
            str(output.relative_to(REPOSITORY))
            if output.is_relative_to(REPOSITORY)
            else str(output)
        ),
        "row_count": int(len(table)),
        "status_counts": {
            str(k): int(v)
            for k, v in table["status"].value_counts(dropna=False).items()
        },
        "production_smoothing_fractions": list(SMOOTHING_FRACTIONS),
        "target_cases": [
            {"q": q, "relief/thickness": relief} for q, relief in TARGET_CASES
        ],
        "minimum_successful_J": (
            float(successful["minimum J"].min()) if len(successful) else None
        ),
        "minimum_successful_raw_J": (
            float(successful["raw minimum J to exit"].min())
            if len(successful)
            else None
        ),
        "overturning_success_count": (
            int(np.count_nonzero(successful["minimum J"] < 0.0))
            if len(successful)
            else 0
        ),
        "finite_flux_primary_p01_negative_count": (
            int(
                np.count_nonzero(
                    successful["first percentile of per-source minimum J"]
                    < 0.0
                )
            )
            if len(successful)
            else 0
        ),
        "minimum_finite_flux_primary_p01_J": (
            float(
                successful[
                    "first percentile of per-source minimum J"
                ].min()
            )
            if len(successful)
            else None
        ),
        "notes": [
            "Surface and bed are parallel; thickness is constant.",
            "Horizontal boundaries are exactly periodic.",
            "Cyclic accumulation zones are split into flow-directed source families at sampled surface-u sign changes.",
            "Positive islands carrying <0.1% of total accumulation flux are rejected as interpolation noise.",
            (
                "Negative minima are narrow and source-sampling sensitive; "
                "the refined sign persists under a quarter-period gauge "
                "translation, while the magnitude is not converged. No "
                "sampled case has negative per-source p01 J."
            ),
            "No manuscript source is modified.",
        ],
    }
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(record, indent=2) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--maximum-age", type=float, default=PRODUCTION_MAXIMUM_AGE
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.jobs < 1 or args.jobs > 2:
        raise ValueError("--jobs must be 1 or 2 for this focused study")
    tasks = (
        quick_tasks(args.maximum_age)
        if args.quick
        else production_tasks(args.maximum_age)
    )
    existing_rows = []
    completed = set()
    if args.output.exists() and not args.force:
        existing = pd.read_csv(args.output)
        if "diagnostic method version" in existing:
            existing = existing.loc[
                existing["diagnostic method version"].eq(METHOD_VERSION)
            ]
        for _, row in existing.iterrows():
            if str(row.get("status", "")) == "success":
                try:
                    completed.add(task_key_from_row(row))
                    existing_rows.append(row.to_dict())
                except (KeyError, TypeError, ValueError):
                    pass
    pending = [task for task in tasks if task_key_from_task(task) not in completed]
    rows = existing_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Periodic convergent study: {len(tasks)} tasks, "
        f"{len(pending)} pending, jobs={args.jobs}.",
        flush=True,
    )

    def checkpoint(row):
        rows.append(row)
        table = sort_table(rows)
        table.to_csv(args.output, index=False)
        write_metadata(table, args.output, args.metadata)
        print(
            f"[{len(rows)}/{len(tasks)}] {row['run kind']}: "
            f"q={row['right / left length ratio']:g}, "
            f"R={row['bed relief / ice thickness']:g}, "
            f"eps={row['wrapped-Gaussian sigma / short flank length']:g}, "
            f"alpha={row['mean driving slope (rad)']:g}, "
            f"status={row['status']}, J={row.get('minimum J', np.nan):.6g}, "
            f"raw={row.get('raw minimum J to exit', np.nan):.6g}",
            flush=True,
        )

    if args.jobs == 1:
        for task in pending:
            checkpoint(solve_case(task))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
            futures = {pool.submit(solve_case, task): task for task in pending}
            for future in as_completed(futures):
                checkpoint(future.result())
    table = sort_table(rows)
    table.to_csv(args.output, index=False)
    write_metadata(table, args.output, args.metadata)
    print(f"Wrote {args.output}", flush=True)
    print(f"Wrote {args.metadata}", flush=True)


if __name__ == "__main__":
    main()
