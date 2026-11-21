"""Idealized two-dimensional Allan Hills inflow/convergence experiment.

This standalone script asks whether the large thickness increase and flow
conversion seen in an Allan Hills cross-section can overturn a material line
in a *steady* velocity field.  It is deliberately separate from the manuscript
and from the existing convergent-basin parameter studies.

The domain is 2 km long.  Its surface changes smoothly from a negative slope
on the thin left side to a positive slope on the thick right side.  The bed is
``b = s - h``, where ``h`` grows smoothly from 70 to 360 m.  The upper surface
is traction free.  The bed has no penetration and the same passive linear
Robin sliding law used by the other idealized experiments.  At the right side
an exact linear-viscous SIA-style horizontal profile supplies leftward inflow,
and the vertical velocity is fixed to zero.  Five cases
separate thickness, inflow strength, and the left boundary condition:

* variable thickness with an open prescribed-traction boundary on the left;
* the same geometry with flux-matched horizontal influx from both sides;
* the same geometry with a closed left divide;
* uniform thickness with the same open-left boundary (a sensitivity case); and
* variable thickness with the full local-SIA influx and a closed left divide.

For each steady Stokes solution, markers enter just inside the right boundary
at a range of depths.  Along every path the deformation gradient obeys

    dF/dt = grad(u) F,        F(0) = I.

For an incoming material-line tangent ``(1, m0)``, the horizontal order
Jacobian is ``J(m0) = F00 + m0 F01``.  The script screens a grid of incoming
slopes, emphasizes ``m0=0`` and the right surface slope, checks ``det(F)=1``,
records exits and age caps, and integrates the two contributions
``dJ/dt = u_x T_x + u_z T_z`` separately.

Outputs (created only when the script is run) are

* ``data/allan_hills_idealized_case_screen.csv``;
* ``data/allan_hills_idealized_source_screen.csv.gz``;
* ``data/allan_hills_idealized_surface_balance.csv``;
* ``figures/allan_hills_idealized_comparison.{png,pdf}``; and
* ``notebooks/allan_hills_idealized_report.md``.

Lengths are metres, time is years, velocity is m/yr, viscosity is Pa yr, and
stress is Pa.  This is a fixed-geometry diagnostic: the surface mass balance
needed to hold the represented surface steady is diagnosed, not prescribed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

try:
    import firedrake as fd
except ImportError:  # Let --help and static inspection work without Firedrake.
    fd = None


warnings.filterwarnings("ignore", message="Unable to import recommended hash")
warnings.filterwarnings("ignore", message="The .*Function.at.* method")
logging.getLogger("tsfc").setLevel(logging.ERROR)


# Geometry and rheology -----------------------------------------------------

LENGTH = 2_000.0
LEFT_THICKNESS = 70.0
RIGHT_THICKNESS = 360.0
MID_THICKNESS = 300.0
FIRST_THICKENING_START = 0.0
FIRST_THICKENING_END = 1_100.0
SECOND_THICKENING_START = 800.0
# Continue the second thickening ramp beyond the modeled window.  After
# normalization below, h(L)=RIGHT_THICKNESS exactly, but h_x(L)>s_x(L), so
# the represented bed is still descending at the right influx boundary as in
# the supplied Allan Hills cross-section.
SECOND_THICKENING_END = 2_200.0

# The surface slope is m_mid + m_amp*tanh((x-x0)/width).  These nominal
# asymptotes make the thin left side flow weakly rightward and the thick right
# side flow leftward.  Their meeting creates the desired convergence.
LEFT_SURFACE_SLOPE = -0.035
RIGHT_SURFACE_SLOPE = 0.012
SURFACE_TRANSITION_X = 520.0
SURFACE_TRANSITION_WIDTH = 140.0
LEFT_SURFACE_ELEVATION = 400.0

DENSITY = 917.0
GRAVITY = 9.81
VISCOSITY = 2.5e8
BASAL_DRAG = 1.0e5
NITSCHE_PENALTY = 3.0e4

REFERENCE_INFLOW_FLUX = -(
    DENSITY
    * GRAVITY
    * RIGHT_SURFACE_SLOPE
    * (
        RIGHT_THICKNESS**2 / BASAL_DRAG
        + RIGHT_THICKNESS**3 / (3.0 * VISCOSITY)
    )
)
REDUCED_INFLOW_FRACTION = 0.01
SOURCE_SIGMA_MIN = 0.01
SOURCE_SIGMA_MAX = 0.95
SAMPLER_SIGMA_LOWER = 0.0005
SAMPLER_SIGMA_UPPER = 0.99

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = ROOT / "data"
FIGURE_DIRECTORY = ROOT / "figures"
REPORT_PATH = ROOT / "notebooks" / "allan_hills_idealized_report.md"

SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


@dataclass(frozen=True)
class Case:
    """One controlled geometry/boundary-condition experiment."""

    key: str
    title: str
    variable_thickness: bool
    left_boundary: str  # ``divide``, ``open_traction``, or ``matched_inflow``
    inflow_fraction: float


CASES = (
    Case(
        "two_sided_variable",
        "Variable thickness, open-left convergence",
        True,
        "open_traction",
        REDUCED_INFLOW_FRACTION,
    ),
    Case(
        "matched_two_sided_variable",
        "Variable thickness, flux-matched convergence",
        True,
        "matched_inflow",
        REDUCED_INFLOW_FRACTION,
    ),
    Case(
        "closed_left_variable",
        "Variable thickness, closed-left control",
        True,
        "divide",
        REDUCED_INFLOW_FRACTION,
    ),
    Case(
        "two_sided_uniform",
        "Uniform thickness, open-left sensitivity",
        False,
        "open_traction",
        REDUCED_INFLOW_FRACTION,
    ),
    Case(
        "closed_left_full_inflow",
        "Variable thickness, full reference influx",
        True,
        "divide",
        1.0,
    ),
)


def require_firedrake() -> None:
    if fd is None:
        raise RuntimeError(
            "Firedrake is required to run this experiment. Activate the "
            "project Firedrake environment and rerun the script."
        )
    if fd.COMM_WORLD.size != 1:
        raise RuntimeError(
            "This screening script currently writes serial trajectory data; "
            "run it with one MPI rank."
        )


def _smoothstep(t):
    """Quintic smoothstep on [0, 1], with zero first derivative at its ends."""
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def _smoothstep_derivative(t):
    return 30.0 * t**2 * (t - 1.0) ** 2


def thickness_weight(x, start, end):
    """Quintic thickening weight, exactly zero/one outside its ramp."""
    x = np.asarray(x, dtype=float)
    width = end - start
    t = (x - start) / width
    clipped = np.clip(t, 0.0, 1.0)
    return _smoothstep(clipped)


def thickness_weight_slope(x, start, end):
    x = np.asarray(x, dtype=float)
    width = end - start
    t = (x - start) / width
    inside = (t > 0.0) & (t < 1.0)
    result = np.zeros_like(t)
    result[inside] = _smoothstep_derivative(t[inside]) / width
    return result


def variable_ice_thickness(x):
    second_weight_at_right = thickness_weight(
        LENGTH, SECOND_THICKENING_START, SECOND_THICKENING_END
    )
    return (
        LEFT_THICKNESS
        + (MID_THICKNESS - LEFT_THICKNESS)
        * thickness_weight(x, FIRST_THICKENING_START, FIRST_THICKENING_END)
        + (RIGHT_THICKNESS - MID_THICKNESS)
        * thickness_weight(x, SECOND_THICKENING_START, SECOND_THICKENING_END)
        / second_weight_at_right
    )


def variable_ice_thickness_slope(x):
    second_weight_at_right = thickness_weight(
        LENGTH, SECOND_THICKENING_START, SECOND_THICKENING_END
    )
    return (
        (MID_THICKNESS - LEFT_THICKNESS)
        * thickness_weight_slope(
            x, FIRST_THICKENING_START, FIRST_THICKENING_END
        )
        + (RIGHT_THICKNESS - MID_THICKNESS)
        * thickness_weight_slope(
            x, SECOND_THICKENING_START, SECOND_THICKENING_END
        )
        / second_weight_at_right
    )


def volume_mean_thickness() -> float:
    """Mean thickness of the variable geometry."""
    x = np.linspace(0.0, LENGTH, 20_001)
    return float(np.trapz(variable_ice_thickness(x), x) / LENGTH)


UNIFORM_THICKNESS = volume_mean_thickness()


def ice_thickness(x, case: Case):
    x = np.asarray(x, dtype=float)
    if case.variable_thickness:
        return variable_ice_thickness(x)
    return np.full_like(x, UNIFORM_THICKNESS, dtype=float)


def ice_thickness_slope(x, case: Case):
    x = np.asarray(x, dtype=float)
    if case.variable_thickness:
        return variable_ice_thickness_slope(x)
    return np.zeros_like(x)


def surface_slope(x):
    x = np.asarray(x, dtype=float)
    midpoint = 0.5 * (LEFT_SURFACE_SLOPE + RIGHT_SURFACE_SLOPE)
    amplitude = 0.5 * (RIGHT_SURFACE_SLOPE - LEFT_SURFACE_SLOPE)
    return midpoint + amplitude * np.tanh(
        (x - SURFACE_TRANSITION_X) / SURFACE_TRANSITION_WIDTH
    )


def _log_cosh(value):
    """Stable log(cosh(value)) for scalar or array NumPy input."""
    value = np.asarray(value, dtype=float)
    absolute = np.abs(value)
    return absolute + np.log1p(np.exp(-2.0 * absolute)) - np.log(2.0)


def surface_elevation(x):
    """Integral of :func:`surface_slope`, anchored at x=0."""
    x = np.asarray(x, dtype=float)
    midpoint = 0.5 * (LEFT_SURFACE_SLOPE + RIGHT_SURFACE_SLOPE)
    amplitude = 0.5 * (RIGHT_SURFACE_SLOPE - LEFT_SURFACE_SLOPE)
    scaled = (x - SURFACE_TRANSITION_X) / SURFACE_TRANSITION_WIDTH
    scaled_zero = -SURFACE_TRANSITION_X / SURFACE_TRANSITION_WIDTH
    return (
        LEFT_SURFACE_ELEVATION
        + midpoint * x
        + amplitude
        * SURFACE_TRANSITION_WIDTH
        * (_log_cosh(scaled) - _log_cosh(scaled_zero))
    )


def bed_elevation(x, case: Case):
    return surface_elevation(x) - ice_thickness(x, case)


def bed_slope(x, case: Case):
    return surface_slope(x) - ice_thickness_slope(x, case)


def _ufl_thickness_weight(x, start, end):
    width = end - start
    t = (x - start) / width
    middle = _smoothstep(t)
    return fd.conditional(
        x <= start,
        0.0,
        fd.conditional(x >= end, 1.0, middle),
    )


def _ufl_surface(x):
    midpoint = 0.5 * (LEFT_SURFACE_SLOPE + RIGHT_SURFACE_SLOPE)
    amplitude = 0.5 * (RIGHT_SURFACE_SLOPE - LEFT_SURFACE_SLOPE)
    scaled = (x - SURFACE_TRANSITION_X) / SURFACE_TRANSITION_WIDTH
    scaled_zero = -SURFACE_TRANSITION_X / SURFACE_TRANSITION_WIDTH
    return (
        LEFT_SURFACE_ELEVATION
        + midpoint * x
        + amplitude
        * SURFACE_TRANSITION_WIDTH
        * (fd.ln(fd.cosh(scaled)) - float(_log_cosh(scaled_zero)))
    )


def _ufl_thickness(x, case: Case):
    if case.variable_thickness:
        return (
            LEFT_THICKNESS
            + (MID_THICKNESS - LEFT_THICKNESS)
            * _ufl_thickness_weight(
                x, FIRST_THICKENING_START, FIRST_THICKENING_END
            )
            + (RIGHT_THICKNESS - MID_THICKNESS)
            * _ufl_thickness_weight(
                x, SECOND_THICKENING_START, SECOND_THICKENING_END
            )
            / float(
                thickness_weight(
                    LENGTH,
                    SECOND_THICKENING_START,
                    SECOND_THICKENING_END,
                )
            )
        )
    return fd.Constant(UNIFORM_THICKNESS)


def make_mesh(case: Case, nx: int, nz: int):
    """Terrain-following mesh retaining RectangleMesh boundary markers."""
    reference_mesh = fd.RectangleMesh(nx, nz, LENGTH, 1.0)
    coordinate_space = fd.VectorFunctionSpace(reference_mesh, "CG", 2)
    x, sigma = fd.SpatialCoordinate(reference_mesh)
    surface = _ufl_surface(x)
    thickness = _ufl_thickness(x, case)
    bed = surface - thickness
    coordinates = fd.Function(coordinate_space).interpolate(
        fd.as_vector((x, bed + sigma * thickness))
    )
    return fd.Mesh(coordinates)


def raw_sia_flux(thickness: float, slope: float) -> float:
    """Depth-integrated horizontal flux of the linear-viscous slab profile."""
    return -DENSITY * GRAVITY * slope * (
        thickness**2 / BASAL_DRAG
        + thickness**3 / (3.0 * VISCOSITY)
    )


def target_right_flux(case: Case) -> float:
    return case.inflow_fraction * REFERENCE_INFLOW_FLUX


def right_inflow_horizontal(mesh, case: Case):
    """SIA-shaped horizontal speed with a prescribed target flux.

    This expression supplies the horizontal Dirichlet datum.  The solve
    separately imposes zero vertical velocity on the right inlet.
    """
    _, z = fd.SpatialCoordinate(mesh)
    h_right = float(ice_thickness(LENGTH, case))
    b_right = float(bed_elevation(LENGTH, case))
    y = z - b_right
    drive = DENSITY * GRAVITY * RIGHT_SURFACE_SLOPE
    raw_u = -drive * (
        h_right / BASAL_DRAG
        + (h_right * y - 0.5 * y**2) / VISCOSITY
    )
    raw_flux = raw_sia_flux(h_right, RIGHT_SURFACE_SLOPE)
    scale = target_right_flux(case) / raw_flux
    return scale * raw_u


def left_inflow_horizontal(mesh, case: Case):
    """SIA-shaped rightward speed with flux matching the right influx.

    Only the horizontal component is prescribed.  Its depth integral is the
    negative of the right outward flux, so the two boundaries import equal
    amounts of ice before the diagnosed surface balance is applied.
    """
    _, z = fd.SpatialCoordinate(mesh)
    h_left = float(ice_thickness(0.0, case))
    b_left = float(bed_elevation(0.0, case))
    y = z - b_left
    drive = DENSITY * GRAVITY * LEFT_SURFACE_SLOPE
    raw_u = -drive * (
        h_left / BASAL_DRAG
        + (h_left * y - 0.5 * y**2) / VISCOSITY
    )
    raw_flux = raw_sia_flux(h_left, LEFT_SURFACE_SLOPE)
    target_flux = -target_right_flux(case)
    return target_flux / raw_flux * raw_u


def open_left_traction(mesh, case: Case):
    """Hydrostatic-plus-shear traction for a reference parallel slab.

    This is an open prescribed-traction boundary.  The effective slab slope
    is calibrated so an *isolated parallel slab* would carry a rightward
    influx equal in magnitude to the target right-side influx; the actual
    full-domain left flux is neither prescribed nor generally matched.
    Its magnitude and sign must be diagnosed from the solved velocity.
    """
    _, z = fd.SpatialCoordinate(mesh)
    h_left = float(ice_thickness(0.0, case))
    surface_left = float(surface_elevation(0.0))
    coefficient = DENSITY * GRAVITY * (
        h_left**2 / BASAL_DRAG
        + h_left**3 / (3.0 * VISCOSITY)
    )
    # target_right_flux is negative (right-boundary inflow).  Using that
    # value here gives a negative reference slope and hence rightward flow at
    # the thin-left boundary, matching the sign of the local surface slope.
    effective_slope = target_right_flux(case) / coefficient
    depth = surface_left - z
    pressure = DENSITY * GRAVITY * depth
    # For slab flow, sigma_xz=-rho*g*m_eff*depth.  With outward normal
    # (-1,0), sigma*n=(p, +rho*g*m_eff*depth).  A negative m_eff therefore
    # applies the downward shear traction associated with rightward inflow.
    return fd.as_vector(
        (pressure, DENSITY * GRAVITY * effective_slope * depth)
    ), effective_slope


def solve_stokes(case: Case, nx: int, nz: int):
    """Solve full, incompressible, linear Stokes flow for one case."""
    mesh = make_mesh(case, nx, nz)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)
    u, p = fd.split(solution)
    v, q = fd.TestFunctions(mixed_space)
    body_force = fd.Constant((0.0, -DENSITY * GRAVITY))
    stress = 2.0 * VISCOSITY * fd.sym(fd.grad(u)) - p * fd.Identity(2)
    test_stress = (
        2.0 * VISCOSITY * fd.sym(fd.grad(v)) - q * fd.Identity(2)
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
    normal_speed = fd.dot(u, normal)
    test_normal_speed = fd.dot(v, normal)
    nominal_cell_size = min(
        LENGTH / nx,
        float(np.min(ice_thickness(np.linspace(0.0, LENGTH, 2001), case)))
        / nz,
    )
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

    effective_left_slope = np.nan
    right_boundary_conditions = [
        fd.DirichletBC(
            mixed_space.sub(0).sub(0),
            right_inflow_horizontal(mesh, case),
            2,
        ),
        fd.DirichletBC(
            mixed_space.sub(0).sub(1), fd.Constant(0.0), 2
        ),
    ]
    if case.left_boundary == "open_traction":
        left_traction, effective_left_slope = open_left_traction(
            mesh, case
        )
        residual -= fd.dot(left_traction, v) * fd.ds(1)
        boundary_conditions = right_boundary_conditions
    elif case.left_boundary == "matched_inflow":
        boundary_conditions = [
            fd.DirichletBC(
                mixed_space.sub(0).sub(0),
                left_inflow_horizontal(mesh, case),
                1,
            ),
        ] + right_boundary_conditions
    elif case.left_boundary == "divide":
        boundary_conditions = [
            fd.DirichletBC(
                mixed_space.sub(0).sub(0), fd.Constant(0.0), 1
            ),
        ] + right_boundary_conditions
    else:
        raise ValueError(f"unknown left boundary {case.left_boundary!r}")

    fd.solve(
        residual == 0,
        solution,
        bcs=boundary_conditions,
        solver_parameters=SOLVER_PARAMETERS,
    )
    velocity, pressure = solution.subfunctions

    # Exact finite-element boundary-flux and constraint diagnostics.
    boundary_flux = {
        name: float(fd.assemble(fd.dot(velocity, normal) * fd.ds(marker)))
        for name, marker in (
            ("left", 1),
            ("right", 2),
            ("bed", 3),
            ("surface", 4),
        )
    }
    bed_length = float(fd.assemble(fd.Constant(1.0) * fd.ds(3, domain=mesh)))
    right_length = float(
        fd.assemble(fd.Constant(1.0) * fd.ds(2, domain=mesh))
    )
    domain_area = float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))
    diagnostics = {
        "case": case.key,
        "title": case.title,
        "variable thickness": case.variable_thickness,
        "left boundary": case.left_boundary,
        "right inflow fraction": case.inflow_fraction,
        "mesh nx": nx,
        "mesh nz": nz,
        "cells": mesh.num_cells(),
        "minimum thickness (m)": float(
            np.min(ice_thickness(np.linspace(0.0, LENGTH, 4001), case))
        ),
        "maximum thickness (m)": float(
            np.max(ice_thickness(np.linspace(0.0, LENGTH, 4001), case))
        ),
        "mean thickness (m)": float(
            np.trapz(
                ice_thickness(np.linspace(0.0, LENGTH, 4001), case),
                np.linspace(0.0, LENGTH, 4001),
            )
            / LENGTH
        ),
        "target right flux (m2/yr)": target_right_flux(case),
        "computed right outward flux (m2/yr)": boundary_flux["right"],
        "right flux relative error": (
            (boundary_flux["right"] - target_right_flux(case))
            / abs(target_right_flux(case))
        ),
        "left outward flux (m2/yr)": boundary_flux["left"],
        "bed outward flux (m2/yr)": boundary_flux["bed"],
        "surface outward flux (m2/yr)": boundary_flux["surface"],
        "required integrated accumulation (m2/yr)": -boundary_flux[
            "surface"
        ],
        "flux closure (m2/yr)": sum(boundary_flux.values()),
        "relative flux closure": sum(boundary_flux.values())
        / max(abs(target_right_flux(case)), 1.0e-12),
        "rms divergence (1/yr)": np.sqrt(
            float(fd.assemble(fd.div(velocity) ** 2 * fd.dx)) / domain_area
        ),
        "rms basal normal speed (m/yr)": np.sqrt(
            float(
                fd.assemble(fd.dot(velocity, normal) ** 2 * fd.ds(3))
            )
            / bed_length
        ),
        "rms right-boundary w (m/yr)": np.sqrt(
            float(fd.assemble(velocity[1] ** 2 * fd.ds(2)))
            / right_length
        ),
        "effective open-left reference slope": effective_left_slope,
    }
    return mesh, velocity, diagnostics


# Terrain-coordinate sampling ----------------------------------------------


def make_sampler(
    case: Case,
    velocity,
    nx_sample: int = 501,
    nz_sample: int = 161,
    horizontal_margin: float = 0.5,
    sigma_lower: float = SAMPLER_SIGMA_LOWER,
    sigma_upper: float = SAMPLER_SIGMA_UPPER,
    domain_left: float = 0.0,
    domain_right: float = LENGTH,
    analytic_bed_function=None,
    analytic_surface_function=None,
):
    """Sample velocity and its physical gradient on a terrain grid.

    Velocity is sampled in ``(x, sigma)`` coordinates fitted to the boundaries
    actually represented by the finite-element mesh.  Derivatives are first
    taken on that regular grid and then transformed to physical ``(x, z)``
    derivatives.  This avoids ambiguous point evaluation of a discontinuous
    projected finite-element gradient and does not substitute the analytic
    geometry for its discrete representation.
    """
    if not 0.0 < sigma_lower < sigma_upper < 1.0:
        raise ValueError("require 0 < sigma_lower < sigma_upper < 1")
    if domain_right - domain_left <= 2.0 * horizontal_margin:
        raise ValueError("sampling domain is narrower than its two margins")
    mesh = velocity.function_space().mesh()
    coordinates = mesh.coordinates.dat.data_ro
    zero = fd.Constant((0.0, 0.0))

    def boundary_interpolator(marker):
        nodes = fd.DirichletBC(velocity.function_space(), zero, marker).nodes
        order = np.argsort(coordinates[nodes, 0])
        xy = coordinates[nodes][order]
        keep = np.r_[True, np.diff(xy[:, 0]) > 1.0e-10]
        return PchipInterpolator(xy[keep, 0], xy[keep, 1])

    represented_bed = boundary_interpolator(3)
    represented_surface = boundary_interpolator(4)
    represented_bed_slope = represented_bed.derivative()
    represented_surface_slope = represented_surface.derivative()

    x = np.linspace(
        domain_left + horizontal_margin,
        domain_right - horizontal_margin,
        nx_sample,
    )
    sigma = np.linspace(sigma_lower, sigma_upper, nz_sample)
    bed = represented_bed(x)
    surface = represented_surface(x)
    thickness = surface - bed
    if analytic_bed_function is None:
        analytic_bed_function = lambda values: bed_elevation(values, case)
    if analytic_surface_function is None:
        analytic_surface_function = surface_elevation
    analytic_bed = analytic_bed_function(x)
    analytic_surface = analytic_surface_function(x)
    X = np.tile(x, (nz_sample, 1))
    Z = bed[None, :] + sigma[:, None] * thickness[None, :]
    points = np.column_stack((X.ravel(), Z.ravel()))
    evaluated = velocity.at(points, tolerance=1.0e-7, dont_raise=True)
    point_is_valid = np.array(
        [value is not None for value in evaluated], dtype=bool
    ).reshape(Z.shape)
    values = np.full(Z.shape + (2,), np.nan)
    if np.any(point_is_valid):
        values[point_is_valid] = np.asarray(
            [value for value in evaluated if value is not None], dtype=float
        )
    missing_point_count = int(np.count_nonzero(~point_is_valid))
    if missing_point_count:
        if missing_point_count > 0.01 * point_is_valid.size:
            raise RuntimeError(
                f"point location failed at {missing_point_count}/"
                f"{point_is_valid.size} terrain samples"
            )
        nearest = distance_transform_edt(
            ~point_is_valid,
            return_distances=False,
            return_indices=True,
        )
        for component in range(2):
            component_values = values[..., component]
            component_values[~point_is_valid] = component_values[
                tuple(nearest[:, ~point_is_valid])
            ]
    if not np.all(np.isfinite(values)):
        raise RuntimeError("nonfinite velocity while sampling terrain grid")

    velocity_interpolator = RegularGridInterpolator(
        (sigma, x), values, bounds_error=False, fill_value=None
    )

    # Grid derivatives at fixed sigma, followed by the terrain-coordinate
    # chain rule.  grad_values[..., i, j] = d velocity_i / d coordinate_j.
    derivative_sigma = np.gradient(values, sigma, axis=0, edge_order=2)
    derivative_x_sigma = np.gradient(values, x, axis=1, edge_order=2)
    local_bed_slope = represented_bed_slope(x)
    local_thickness_slope = (
        represented_surface_slope(x) - local_bed_slope
    )
    z_x_at_sigma = (
        local_bed_slope[None, :]
        + sigma[:, None] * local_thickness_slope[None, :]
    )
    sigma_x_at_z = -z_x_at_sigma / thickness[None, :]
    sigma_z = 1.0 / thickness[None, :]
    gradient_values = np.empty(Z.shape + (2, 2))
    for component in range(2):
        gradient_values[..., component, 0] = (
            derivative_x_sigma[..., component]
            + derivative_sigma[..., component] * sigma_x_at_z
        )
        gradient_values[..., component, 1] = (
            derivative_sigma[..., component] * sigma_z
        )
    gradient_interpolator = RegularGridInterpolator(
        (sigma, x), gradient_values, bounds_error=False, fill_value=None
    )

    def physical_to_terrain(points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        local_bed = represented_bed(points[:, 0])
        local_thickness = (
            represented_surface(points[:, 0]) - local_bed
        )
        local_sigma = (points[:, 1] - local_bed) / local_thickness
        return np.column_stack((local_sigma, points[:, 0]))

    def sample(points):
        return velocity_interpolator(physical_to_terrain(points))

    def sample_gradient(points):
        return gradient_interpolator(physical_to_terrain(points))

    surface_velocity = sample(
        np.column_stack(
            (
                x,
                bed + sigma_upper * thickness,
            )
        )
    )
    surface_mass_balance = (
        surface_velocity[:, 0] * represented_surface_slope(x)
        - surface_velocity[:, 1]
    )
    integrated_surface_mass_balance = float(
        np.trapz(surface_mass_balance, x)
    )
    sampled_divergence = (
        gradient_values[..., 0, 0] + gradient_values[..., 1, 1]
    )
    diagnostics = {
        "minimum sampled u (m/yr)": float(np.min(values[..., 0])),
        "maximum sampled u (m/yr)": float(np.max(values[..., 0])),
        "minimum sampled w (m/yr)": float(np.min(values[..., 1])),
        "maximum sampled w (m/yr)": float(np.max(values[..., 1])),
        "point-locator nearest-fill count": missing_point_count,
        "maximum represented bed mismatch (m)": float(
            np.max(np.abs(bed - analytic_bed))
        ),
        "maximum represented surface mismatch (m)": float(
            np.max(np.abs(surface - analytic_surface))
        ),
        "maximum abs sampled divergence (1/yr)": float(
            np.max(np.abs(sampled_divergence))
        ),
        "minimum sampled ux (1/yr)": float(
            np.min(gradient_values[..., 0, 0])
        ),
        "fraction of sampled domain with ux < 0": float(
            np.mean(gradient_values[..., 0, 0] < 0.0)
        ),
        "minimum near-surface mass balance (m/yr)": float(
            np.min(surface_mass_balance)
        ),
        "maximum near-surface mass balance (m/yr)": float(
            np.max(surface_mass_balance)
        ),
        "integrated near-surface mass balance (m2/yr)": (
            integrated_surface_mass_balance
        ),
    }
    left_mask = x <= domain_left + 0.25 * (domain_right - domain_left)
    left_w = values[:, left_mask, 1]
    diagnostics.update(
        {
            "mean w in left quarter (m/yr)": float(np.mean(left_w)),
            "fraction w < 0 in left quarter": float(np.mean(left_w < 0.0)),
        }
    )
    return {
        "sample": sample,
        "sample gradient": sample_gradient,
        "x": x,
        "sigma": sigma,
        "X": X,
        "Z": Z,
        "velocity": values,
        "gradient": gradient_values,
        "surface mass balance": surface_mass_balance,
        "bed": represented_bed,
        "surface": represented_surface,
        "bed slope": represented_bed_slope,
        "surface slope": represented_surface_slope,
        "thickness": lambda x_value: (
            represented_surface(x_value) - represented_bed(x_value)
        ),
        "diagnostics": diagnostics,
        "horizontal margin": horizontal_margin,
        "domain left": domain_left,
        "domain right": domain_right,
        "sigma lower": sigma_lower,
        "sigma upper": sigma_upper,
    }


# Steady path and deformation-gradient diagnostic --------------------------


def _state_rhs(state, sampler):
    """Velocity, deformation-gradient, and tangent-budget right-hand side."""
    points = state[:, :2]
    velocity = sampler["sample"](points)
    gradient = sampler["sample gradient"](points)
    deformation = state[:, 2:6].reshape((-1, 2, 2))
    deformation_rate = np.einsum("nij,njk->nik", gradient, deformation)

    # Four budgets are the ux*Tx and uz*Tz contributions for the two basis
    # tangents e_x and e_z.  Any incoming slope m0 is their linear combination.
    tangent_x = deformation[:, :, 0]
    tangent_z = deformation[:, :, 1]
    budget_rate = np.column_stack(
        (
            gradient[:, 0, 0] * tangent_x[:, 0],
            gradient[:, 0, 1] * tangent_x[:, 1],
            gradient[:, 0, 0] * tangent_z[:, 0],
            gradient[:, 0, 1] * tangent_z[:, 1],
        )
    )
    return np.column_stack(
        (velocity, deformation_rate.reshape((-1, 4)), budget_rate)
    )


def _terrain_sigma(points, sampler):
    points = np.atleast_2d(points)
    return (
        points[:, 1] - sampler["bed"](points[:, 0])
    ) / sampler["thickness"](points[:, 0])


def trace_incoming_material(
    case: Case,
    sampler,
    slopes,
    source_count: int,
    maximum_age: float,
    requested_time_step: float,
    source_sigma_values=None,
    store_history: bool = True,
    source_sigma_min: float = SOURCE_SIGMA_MIN,
    source_sigma_max: float = SOURCE_SIGMA_MAX,
):
    """Trace right-boundary sources and screen ``J(m0)`` over slope."""
    slopes = np.unique(
        np.sort(
            np.r_[np.asarray(slopes, dtype=float), 0.0, RIGHT_SURFACE_SLOPE]
        )
    )
    if source_sigma_values is None:
        source_sigma = np.linspace(
            source_sigma_min, source_sigma_max, source_count
        )
    else:
        source_sigma = np.unique(np.asarray(source_sigma_values, dtype=float))
        if (
            source_sigma.ndim != 1
            or len(source_sigma) < 1
            or np.any(source_sigma <= sampler["sigma lower"])
            or np.any(source_sigma >= sampler["sigma upper"])
        ):
            raise ValueError(
                "explicit source sigma values must lie strictly inside the "
                "terrain-sampler buffer"
            )
        source_count = len(source_sigma)
    start_x = sampler["domain right"] - 2.0 * sampler["horizontal margin"]
    start_bed = float(sampler["bed"](start_x))
    start_thickness = float(sampler["thickness"](start_x))
    positions = np.column_stack(
        (
            np.full(source_count, start_x),
            start_bed + source_sigma * start_thickness,
        )
    )
    state = np.zeros((source_count, 10), dtype=float)
    state[:, :2] = positions
    state[:, 2:6] = np.array((1.0, 0.0, 0.0, 1.0))

    # Limit RK4 steps by independent horizontal and vertical sampling scales.
    velocity = sampler["velocity"]
    dx = np.min(np.diff(sampler["x"]))
    dz_min = (
        float(np.min(sampler["thickness"](sampler["x"])))
        * np.min(np.diff(sampler["sigma"]))
    )
    horizontal_limit = 0.35 * dx / max(
        float(np.max(np.abs(velocity[..., 0]))), 1.0e-12
    )
    vertical_limit = 0.35 * dz_min / max(
        float(np.max(np.abs(velocity[..., 1]))), 1.0e-12
    )
    time_step = min(requested_time_step, horizontal_limit, vertical_limit)
    if time_step <= 0.0 or not np.isfinite(time_step):
        raise RuntimeError("could not determine a finite positive time step")

    n_slopes = len(slopes)
    minimum_j = np.full((source_count, n_slopes), np.inf)
    minimum_age = np.full_like(minimum_j, np.nan)
    minimum_x = np.full_like(minimum_j, np.nan)
    minimum_z = np.full_like(minimum_j, np.nan)
    minimum_sigma = np.full_like(minimum_j, np.nan)
    minimum_det_error = np.full_like(minimum_j, np.nan)
    minimum_ux_contribution = np.full_like(minimum_j, np.nan)
    minimum_uz_contribution = np.full_like(minimum_j, np.nan)
    minimum_budget_closure = np.full_like(minimum_j, np.nan)
    maximum_det_error = np.zeros(source_count)
    active = np.ones(source_count, dtype=bool)
    exit_status = np.full(source_count, "active", dtype=object)
    exit_age = np.full(source_count, np.nan)

    history_stride = max(1, int(round(50.0 / time_step)))
    history_times = []
    history_positions = []
    history_states = []

    def record_values(indices, values, ages):
        """Record minima for active states or interpolated exit states."""
        indices = np.asarray(indices, dtype=int)
        values = np.atleast_2d(values)
        ages = np.broadcast_to(np.asarray(ages, dtype=float), (len(indices),))
        if not len(indices):
            return
        deformation = values[:, 2:6].reshape((-1, 2, 2))
        determinant = np.linalg.det(deformation)
        maximum_det_error[indices] = np.maximum(
            maximum_det_error[indices], np.abs(determinant - 1.0)
        )
        current_j = (
            deformation[:, 0, 0, None]
            + deformation[:, 0, 1, None] * slopes[None, :]
        )
        budgets = values[:, 6:10]
        ux_contribution = (
            budgets[:, 0, None] + budgets[:, 2, None] * slopes[None, :]
        )
        uz_contribution = (
            budgets[:, 1, None] + budgets[:, 3, None] * slopes[None, :]
        )
        closure = current_j - (1.0 + ux_contribution + uz_contribution)
        local_sigma = _terrain_sigma(values[:, :2], sampler)
        improved = current_j < minimum_j[indices]
        local_rows, slope_columns = np.nonzero(improved)
        if not len(local_rows):
            return
        global_rows = indices[local_rows]
        minimum_j[global_rows, slope_columns] = current_j[
            local_rows, slope_columns
        ]
        minimum_age[global_rows, slope_columns] = ages[local_rows]
        minimum_x[global_rows, slope_columns] = values[local_rows, 0]
        minimum_z[global_rows, slope_columns] = values[local_rows, 1]
        minimum_sigma[global_rows, slope_columns] = local_sigma[local_rows]
        minimum_det_error[global_rows, slope_columns] = np.abs(
            determinant[local_rows] - 1.0
        )
        minimum_ux_contribution[global_rows, slope_columns] = ux_contribution[
            local_rows, slope_columns
        ]
        minimum_uz_contribution[global_rows, slope_columns] = uz_contribution[
            local_rows, slope_columns
        ]
        minimum_budget_closure[global_rows, slope_columns] = closure[
            local_rows, slope_columns
        ]

    def record(age):
        indices = np.flatnonzero(active)
        record_values(indices, state[indices], np.full(len(indices), age))

    steps = int(np.ceil(maximum_age / time_step))
    for step in range(steps + 1):
        age = min(step * time_step, maximum_age)
        record(age)
        if store_history and (step % history_stride == 0 or step == steps):
            recorded = state[:, :2].copy()
            recorded[~active] = np.nan
            history_times.append(age)
            history_positions.append(recorded)
            recorded_state = state.copy()
            recorded_state[~active] = np.nan
            history_states.append(recorded_state)
        if step == steps or not np.any(active):
            break

        indices = np.flatnonzero(active)
        current = state[indices]
        local_step = min(time_step, maximum_age - age)
        k1 = _state_rhs(current, sampler)
        k2 = _state_rhs(current + 0.5 * local_step * k1, sampler)
        k3 = _state_rhs(current + 0.5 * local_step * k2, sampler)
        k4 = _state_rhs(current + local_step * k3, sampler)
        candidate = current + local_step * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
        state[indices] = candidate

        current_sigma = _terrain_sigma(current[:, :2], sampler)
        local_sigma = _terrain_sigma(candidate[:, :2], sampler)
        nonfinite = ~np.all(np.isfinite(candidate), axis=1)
        left_boundary = (
            sampler["domain left"] + sampler["horizontal margin"]
        )
        right_boundary = (
            sampler["domain right"] - sampler["horizontal margin"]
        )
        left = candidate[:, 0] <= left_boundary
        right = candidate[:, 0] >= right_boundary
        bed = local_sigma <= sampler["sigma lower"]
        surface = local_sigma >= sampler["sigma upper"]
        leaving = nonfinite | left | right | bed | surface
        for local_index in np.flatnonzero(leaving):
            source_index = indices[local_index]
            if nonfinite[local_index]:
                status = "nonfinite"
                fraction = 0.0
            elif left[local_index]:
                status = "left_exit"
                fraction = (
                    current[local_index, 0] - left_boundary
                ) / (
                    current[local_index, 0] - candidate[local_index, 0]
                )
            elif right[local_index]:
                status = "right_exit"
                fraction = (
                    right_boundary - current[local_index, 0]
                ) / (
                    candidate[local_index, 0] - current[local_index, 0]
                )
            elif bed[local_index]:
                status = "bed_buffer"
                fraction = (
                    current_sigma[local_index] - sampler["sigma lower"]
                ) / (
                    current_sigma[local_index] - local_sigma[local_index]
                )
            else:
                status = "surface_exit"
                fraction = (
                    sampler["sigma upper"] - current_sigma[local_index]
                ) / (
                    local_sigma[local_index] - current_sigma[local_index]
                )
            fraction = float(np.clip(fraction, 0.0, 1.0))
            if status != "nonfinite":
                exit_state = current[local_index] + fraction * (
                    candidate[local_index] - current[local_index]
                )
                state[source_index] = exit_state
                interpolated_age = age + fraction * local_step
                record_values(
                    np.array((source_index,)),
                    exit_state[None, :],
                    np.array((interpolated_age,)),
                )
            else:
                interpolated_age = age
            exit_status[source_index] = status
            exit_age[source_index] = interpolated_age
        active[indices[leaving]] = False

    exit_status[active] = "age_cap"
    exit_age[active] = maximum_age
    rows = []
    for source_index, sigma0 in enumerate(source_sigma):
        for slope_index, slope in enumerate(slopes):
            rows.append(
                {
                    "case": case.key,
                    "title": case.title,
                    "source index": source_index,
                    "source sigma": sigma0,
                    "incoming slope m0": slope,
                    "is primary horizontal": bool(np.isclose(slope, 0.0)),
                    "is primary surface parallel": bool(
                        np.isclose(slope, RIGHT_SURFACE_SLOPE)
                    ),
                    "minimum J": minimum_j[source_index, slope_index],
                    "age at minimum J (yr)": minimum_age[
                        source_index, slope_index
                    ],
                    "x at minimum J (m)": minimum_x[
                        source_index, slope_index
                    ],
                    "z at minimum J (m)": minimum_z[
                        source_index, slope_index
                    ],
                    "sigma at minimum J": minimum_sigma[
                        source_index, slope_index
                    ],
                    "ux Tx integrated contribution": (
                        minimum_ux_contribution[source_index, slope_index]
                    ),
                    "uz Tz integrated contribution": (
                        minimum_uz_contribution[source_index, slope_index]
                    ),
                    "J budget closure at minimum": (
                        minimum_budget_closure[source_index, slope_index]
                    ),
                    "abs detF-1 at minimum": minimum_det_error[
                        source_index, slope_index
                    ],
                    "maximum abs detF-1 on path": maximum_det_error[
                        source_index
                    ],
                    "path status": exit_status[source_index],
                    "path exit/cap age (yr)": exit_age[source_index],
                    "time step (yr)": time_step,
                }
            )
    return {
        "rows": rows,
        "slopes": slopes,
        "source sigma": source_sigma,
        "exit status": exit_status,
        "exit age": exit_age,
        "maximum determinant error": maximum_det_error,
        "time step": time_step,
        "history times": np.asarray(history_times),
        "history positions": np.asarray(history_positions),
        "history states": np.asarray(history_states),
    }


def _minimum_row(frame: pd.DataFrame, slope: float) -> pd.Series:
    subset = frame[np.isclose(frame["incoming slope m0"], slope)]
    if subset.empty:
        raise RuntimeError(f"incoming slope {slope:g} missing from screen")
    return subset.loc[subset["minimum J"].idxmin()]


def run_case(case: Case, args):
    """Solve, sample, trace, and reduce one controlled experiment."""
    print(f"Solving {case.key}: {case.title}", flush=True)
    mesh, velocity, diagnostics = solve_stokes(case, args.nx, args.nz)
    sampler = make_sampler(
        case,
        velocity,
        nx_sample=args.sample_nx,
        nz_sample=args.sample_nz,
        sigma_lower=getattr(args, "sigma_lower", SAMPLER_SIGMA_LOWER),
        sigma_upper=getattr(args, "sigma_upper", SAMPLER_SIGMA_UPPER),
    )
    diagnostics.update(sampler["diagnostics"])
    diagnosed_accumulation = diagnostics[
        "integrated near-surface mass balance (m2/yr)"
    ]
    flux_accumulation = diagnostics["required integrated accumulation (m2/yr)"]
    diagnostics["surface mass-balance integral error (m2/yr)"] = (
        diagnosed_accumulation - flux_accumulation
    )
    diagnostics["relative surface mass-balance integral error"] = (
        (diagnosed_accumulation - flux_accumulation)
        / max(abs(flux_accumulation), 1.0e-12)
    )
    slopes = np.linspace(-args.slope_limit, args.slope_limit, args.slope_count)
    trace = trace_incoming_material(
        case,
        sampler,
        slopes,
        args.sources,
        args.maximum_age,
        args.time_step,
        source_sigma_values=getattr(args, "source_sigma_values", None),
        store_history=getattr(args, "store_history", True),
        source_sigma_min=getattr(
            args, "source_sigma_min", SOURCE_SIGMA_MIN
        ),
        source_sigma_max=getattr(
            args, "source_sigma_max", SOURCE_SIGMA_MAX
        ),
    )
    source_frame = pd.DataFrame(trace["rows"])
    horizontal = _minimum_row(source_frame, 0.0)
    surface_parallel = _minimum_row(source_frame, RIGHT_SURFACE_SLOPE)
    statuses = pd.Series(trace["exit status"]).value_counts()
    diagnostics.update(
        {
            "minimum J over all incoming slopes": float(
                source_frame["minimum J"].min()
            ),
            "minimum J, m0=0": float(horizontal["minimum J"]),
            "age at minimum J, m0=0 (yr)": float(
                horizontal["age at minimum J (yr)"]
            ),
            "source sigma at minimum J, m0=0": float(
                horizontal["source sigma"]
            ),
            "ux Tx contribution at minimum, m0=0": float(
                horizontal["ux Tx integrated contribution"]
            ),
            "uz Tz contribution at minimum, m0=0": float(
                horizontal["uz Tz integrated contribution"]
            ),
            "minimum J, m0=m_right": float(surface_parallel["minimum J"]),
            "age at minimum J, m0=m_right (yr)": float(
                surface_parallel["age at minimum J (yr)"]
            ),
            "source sigma at minimum J, m0=m_right": float(
                surface_parallel["source sigma"]
            ),
            "ux Tx contribution at minimum, m0=m_right": float(
                surface_parallel["ux Tx integrated contribution"]
            ),
            "uz Tz contribution at minimum, m0=m_right": float(
                surface_parallel["uz Tz integrated contribution"]
            ),
            "maximum abs detF-1": float(
                np.max(trace["maximum determinant error"])
            ),
            "maximum abs J budget closure": float(
                source_frame["J budget closure at minimum"].abs().max()
            ),
            "trajectory time step (yr)": trace["time step"],
            "trajectory sources": len(trace["source sigma"]),
            "left exits": int(statuses.get("left_exit", 0)),
            "surface exits": int(statuses.get("surface_exit", 0)),
            "bed-buffer terminations": int(statuses.get("bed_buffer", 0)),
            "right exits": int(statuses.get("right_exit", 0)),
            "nonfinite paths": int(statuses.get("nonfinite", 0)),
            "age-capped paths": int(statuses.get("age_cap", 0)),
        }
    )

    x = sampler["x"]
    near_surface_velocity = sampler["sample"](
        np.column_stack(
            (
                x,
                sampler["bed"](x)
                + sampler["sigma upper"] * sampler["thickness"](x),
            )
        )
    )
    surface_frame = pd.DataFrame(
        {
            "case": case.key,
            "x (m)": x,
            "bed elevation (m)": sampler["bed"](x),
            "surface elevation (m)": sampler["surface"](x),
            "thickness (m)": sampler["thickness"](x),
            "surface slope": sampler["surface slope"](x),
            "near-surface u (m/yr)": near_surface_velocity[:, 0],
            "near-surface w (m/yr)": near_surface_velocity[:, 1],
            "diagnosed near-surface mass balance (m/yr)": sampler[
                "surface mass balance"
            ],
        }
    )
    print(
        f"  Jmin(m0=0)={diagnostics['minimum J, m0=0']:.4g}; "
        f"Jmin(m0=m_right)={diagnostics['minimum J, m0=m_right']:.4g}; "
        f"max|detF-1|={diagnostics['maximum abs detF-1']:.2e}",
        flush=True,
    )
    return {
        "case": case,
        "mesh": mesh,
        "velocity": velocity,
        "sampler": sampler,
        "trace": trace,
        "diagnostics": diagnostics,
        "source frame": source_frame,
        "surface frame": surface_frame,
    }


# Figure and report ---------------------------------------------------------


def _draw_geometry(axis):
    x = np.linspace(0.0, LENGTH, 1001)
    reference_case = CASES[0]
    bed = bed_elevation(x, reference_case)
    surface = surface_elevation(x)
    axis.fill_between(x / 1000.0, np.min(bed) - 45.0, bed, color="0.72")
    axis.fill_between(
        x / 1000.0, bed, surface, color="#d8edf3", alpha=0.85
    )
    axis.plot(x / 1000.0, bed, color="0.15", linewidth=1.2)
    axis.plot(x / 1000.0, surface, color="#174a6e", linewidth=1.5)
    for x_km in (0.16, 1.84):
        x_m = 1000.0 * x_km
        axis.annotate(
            "",
            xy=(x_km, float(surface_elevation(x_m))),
            xytext=(x_km, float(bed_elevation(x_m, reference_case))),
            arrowprops={"arrowstyle": "<->", "linewidth": 0.8},
        )
    axis.text(
        0.25,
        377.0,
        rf"$h_L\simeq{LEFT_THICKNESS:.0f}$ m",
        ha="left",
        va="center",
    )
    axis.text(
        1.52,
        220.0,
        "thick right\n" + rf"$h_R\simeq{RIGHT_THICKNESS:.0f}$ m",
        ha="right",
        va="center",
    )
    # Define the source-depth coordinate used on panel D directly on the
    # geometry.  The source itself lies on the exact right boundary; the
    # adjacent bracket shows its height above the bed as a fraction of h_R.
    source_sigma_example = 0.28
    right_bed = float(bed_elevation(LENGTH, reference_case))
    right_surface = float(surface_elevation(LENGTH))
    source_elevation = right_bed + source_sigma_example * (
        right_surface - right_bed
    )
    axis.plot(
        LENGTH / 1000.0,
        source_elevation,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor="0.1",
        clip_on=False,
        zorder=6,
    )
    axis.annotate(
        "",
        xy=(1.72, source_elevation),
        xytext=(1.72, right_bed),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.75},
    )
    axis.plot(
        (1.72, LENGTH / 1000.0),
        (source_elevation, source_elevation),
        color="0.35",
        linewidth=0.7,
        linestyle=":",
    )
    axis.text(
        1.69,
        0.5 * (right_bed + source_elevation),
        r"$\sigma_0h_R$",
        ha="right",
        va="center",
        fontsize=8.0,
    )
    axis.text(
        1.30,
        source_elevation - 10.0,
        r"source: $z_0=b_R+\sigma_0h_R$",
        ha="left",
        va="top",
        fontsize=7.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )
    axis.annotate(
        "$u=u_{\\mathrm{SIA}}(z)$\n$w=0$",
        xy=(1.62, 310.0),
        xytext=(1.96, 310.0),
        arrowprops={"arrowstyle": "->", "linewidth": 1.3},
        ha="right",
        va="center",
    )
    axis.annotate(
        "open traction",
        xy=(0.28, 340.0),
        xytext=(0.03, 340.0),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        ha="left",
        va="center",
    )
    axis.text(
        0.55,
        0.96,
        "traction-free surface",
        transform=axis.transAxes,
        ha="center",
        color="#174a6e",
    )
    axis.text(
        0.03,
        0.05,
        "bed: no penetration + linear sliding\nvertical scale exaggerated",
        transform=axis.transAxes,
        fontsize=8.0,
        va="bottom",
    )
    axis.set(
        xlim=(0.0, 2.0),
        ylim=(np.min(bed) - 35.0, np.max(surface) + 20.0),
        xlabel="Distance (km)",
        ylabel="Elevation (m)",
        title="Idealized Allan Hills geometry and boundaries",
    )


def _representative_path_indices(source_count):
    return np.unique(
        np.linspace(1, max(1, source_count - 2), min(7, source_count), dtype=int)
    )


def _draw_flow(axis, result, color_norm, levels):
    case = result["case"]
    sampler = result["sampler"]
    contour = axis.contourf(
        sampler["X"] / 1000.0,
        sampler["Z"],
        sampler["velocity"][..., 0],
        levels=levels,
        cmap="RdBu_r",
        norm=color_norm,
        extend="both",
    )
    x = sampler["x"]
    bed = sampler["bed"](x)
    surface = sampler["surface"](x)
    axis.fill_between(
        x / 1000.0,
        np.min(bed) - 45.0,
        bed,
        color="0.70",
        zorder=4,
    )
    axis.plot(x / 1000.0, bed, color="0.12", linewidth=0.8, zorder=5)
    axis.plot(x / 1000.0, surface, color="0.12", linewidth=0.9, zorder=5)
    horizontal = sampler["velocity"][..., 0]
    if np.min(horizontal) < 0.0 < np.max(horizontal):
        axis.contour(
            sampler["X"] / 1000.0,
            sampler["Z"],
            horizontal,
            levels=(0.0,),
            colors="0.12",
            linewidths=0.9,
            zorder=5,
        )

    x_indices = np.linspace(
        1, len(x) - 2, min(10, max(2, len(x) - 2)), dtype=int
    )
    z_indices = np.linspace(
        1,
        len(sampler["sigma"]) - 2,
        min(4, max(2, len(sampler["sigma"]) - 2)),
        dtype=int,
    )
    arrows = sampler["velocity"][np.ix_(z_indices, x_indices)]
    speeds = np.linalg.norm(arrows, axis=2)
    directions = arrows / np.maximum(speeds[..., None], 1.0e-12)
    arrow_length_m = 60.0
    axis.quiver(
        sampler["X"][np.ix_(z_indices, x_indices)] / 1000.0,
        sampler["Z"][np.ix_(z_indices, x_indices)],
        arrow_length_m * directions[..., 0] / 1000.0,
        arrow_length_m * directions[..., 1],
        color="0.08",
        edgecolor="white",
        linewidth=0.25,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        zorder=6,
    )

    history = result["trace"]["history positions"]
    for source_index in _representative_path_indices(history.shape[1]):
        path = history[:, source_index]
        valid = np.all(np.isfinite(path), axis=1)
        axis.plot(
            path[valid, 0] / 1000.0,
            path[valid, 1],
            color="#f0cf5b",
            linewidth=0.9,
            alpha=0.9,
            zorder=7,
        )
    source_frame = result["source frame"]
    primary = source_frame[source_frame["is primary horizontal"]]
    negative = primary[primary["minimum J"] < 0.0]
    if not negative.empty:
        selected = negative.loc[negative["minimum J"].idxmin()]
        axis.scatter(
            [selected["x at minimum J (m)"] / 1000.0],
            [selected["z at minimum J (m)"]],
            s=28,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.0,
            zorder=8,
        )
        axis.annotate(
            "$J_{min}$",
            xy=(
                selected["x at minimum J (m)"] / 1000.0,
                selected["z at minimum J (m)"],
            ),
            xytext=(8, 8),
            textcoords="offset points",
            color="#b2182b",
            fontsize=8.0,
            zorder=9,
        )
    diagnostic = result["diagnostics"]
    boundary_label = (
        "left divide"
        if case.left_boundary == "divide"
        else (
            "flux-matched left"
            if case.left_boundary == "matched_inflow"
            else "open-traction left"
        )
    )
    axis.text(
        0.02,
        0.04,
        f"{boundary_label}; $Q_R/Q_0={case.inflow_fraction:g}$\n"
        f"$J_{{min}}(0)={diagnostic['minimum J, m0=0']:.3f}$",
        transform=axis.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        zorder=9,
    )
    axis.text(
        0.985,
        0.73,
        "right inlet: $w=0$",
        transform=axis.transAxes,
        rotation=90,
        ha="right",
        va="center",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.60},
        zorder=9,
    )
    axis.set(
        xlim=(0.0, 2.0),
        ylim=(np.min(bed) - 25.0, np.max(surface) + 10.0),
        xlabel="Distance (km)",
        ylabel="Elevation (m)",
        title=case.title,
    )
    return contour


def _draw_vertical_velocity(axis, result):
    """Show the downwelling/upwelling structure of the primary case."""
    sampler = result["sampler"]
    vertical = sampler["velocity"][..., 1]
    limit = max(float(np.max(np.abs(vertical))), 1.0e-8)
    levels = np.linspace(-limit, limit, 25)
    contour = axis.contourf(
        sampler["X"] / 1000.0,
        sampler["Z"],
        vertical,
        levels=levels,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        extend="both",
    )
    if np.min(vertical) < 0.0 < np.max(vertical):
        axis.contour(
            sampler["X"] / 1000.0,
            sampler["Z"],
            vertical,
            levels=(0.0,),
            colors="0.15",
            linewidths=0.8,
        )
    x = sampler["x"]
    bed = sampler["bed"](x)
    surface = sampler["surface"](x)
    axis.fill_between(
        x / 1000.0,
        np.min(bed) - 45.0,
        bed,
        color="0.70",
        zorder=4,
    )
    axis.plot(x / 1000.0, bed, color="0.12", linewidth=0.8, zorder=5)
    axis.plot(x / 1000.0, surface, color="0.12", linewidth=0.9, zorder=5)
    diagnostic = result["diagnostics"]
    axis.text(
        0.02,
        0.04,
        "black: $w=0$\n"
        f"left-quarter $w<0$: "
        f"{100.0 * diagnostic['fraction w < 0 in left quarter']:.0f}%",
        transform=axis.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        zorder=6,
    )
    axis.set(
        xlim=(0.0, 2.0),
        ylim=(np.min(bed) - 25.0, np.max(surface) + 10.0),
        xlabel="Distance (km)",
        ylabel="Elevation (m)",
        title="Vertical velocity and zero contour",
    )
    return contour


def _draw_inlet_j_map(axis, result):
    """Depth--incoming-slope map for the primary side-inlet diagnostic."""
    frame = result["source frame"]
    grid = frame.pivot(
        index="source sigma", columns="incoming slope m0", values="minimum J"
    )
    slopes = grid.columns.to_numpy(dtype=float)
    source_sigma = grid.index.to_numpy(dtype=float)
    values = grid.to_numpy(dtype=float)
    display_limit = 0.60
    image = axis.pcolormesh(
        slopes,
        source_sigma,
        values,
        shading="nearest",
        cmap="RdBu",
        norm=TwoSlopeNorm(
            vmin=-display_limit, vcenter=0.0, vmax=display_limit
        ),
        rasterized=True,
    )
    if np.nanmin(values) < 0.0 < np.nanmax(values):
        axis.contour(slopes, source_sigma, values, levels=(0.0,), colors="0.15")
    axis.axvline(0.0, color="0.35", linewidth=0.7, linestyle="--")
    axis.axvline(
        RIGHT_SURFACE_SLOPE,
        color="0.35",
        linewidth=0.7,
        linestyle=":",
    )
    minimum = frame.loc[frame["minimum J"].idxmin()]
    axis.plot(
        minimum["incoming slope m0"],
        minimum["source sigma"],
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="0.1",
    )
    axis.annotate(
        f"minimum $J={minimum['minimum J']:.3f}$",
        xy=(minimum["incoming slope m0"], minimum["source sigma"]),
        xytext=(-0.088, 0.17),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8.2,
    )
    axis.text(
        0.98,
        0.03,
        f"colors clipped at $|J|={display_limit:.2f}$",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
    )
    axis.set(
        xlabel="Incoming layer slope, $m_0$",
        ylabel=(
            "Right-inlet normalized source height\n"
            "$\\sigma_0=(z_0-b_R)/h_R$\n"
            "(0 = bed; 1 = surface)"
        ),
        title="Where do incoming ordered layers overturn?",
    )
    return image


def _draw_tangent_history(axis, result):
    """Show the two cumulative terms that create the primary fold."""
    frame = result["source frame"]
    horizontal = frame[frame["is primary horizontal"]]
    minimum = horizontal.loc[horizontal["minimum J"].idxmin()]
    source_index = int(minimum["source index"])
    ages = result["trace"]["history times"]
    states = result["trace"]["history states"][:, source_index]
    valid = np.all(np.isfinite(states), axis=1)
    ages = ages[valid] / 1000.0
    states = states[valid]
    j_values = states[:, 2]
    ux_terms = states[:, 6]
    uz_terms = states[:, 7]

    axis.plot(ages, j_values, color="0.10", linewidth=1.8, label="$J$")
    axis.plot(
        ages,
        ux_terms,
        color="#2c7fb8",
        linewidth=1.4,
        label=r"$\int u_xT_x\,dt$",
    )
    axis.plot(
        ages,
        uz_terms,
        color="#d95f0e",
        linewidth=1.4,
        label=r"$\int u_zT_z\,dt$",
    )
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    negative = np.flatnonzero(j_values < 0.0)
    if len(negative):
        first = negative[0]
        axis.axvline(
            ages[first], color="#b2182b", linewidth=0.8, linestyle="--"
        )
        axis.text(
            ages[first],
            0.03,
            "$J=0$",
            color="#b2182b",
            fontsize=8.0,
            ha="right",
            va="bottom",
        )
    minimum_index = int(np.argmin(j_values))
    axis.scatter(
        ages[minimum_index],
        j_values[minimum_index],
        s=28,
        facecolors="none",
        edgecolors="#b2182b",
        zorder=4,
    )
    axis.legend(frameon=False, fontsize=8.0, loc="best")
    axis.set(
        xlabel="Travel time from right inlet (kyr)",
        ylabel="Tangent diagnostic / cumulative contribution",
        title="Vertical shear drives the fold",
    )


def make_comparison_figure(results):
    """Geometry, primary velocity mechanics, and controlled J comparisons."""
    primary = results[0]
    primary_u = primary["sampler"]["velocity"][..., 0]
    color_limit = max(float(np.max(np.abs(primary_u))), 1.0e-6)
    color_norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    levels = np.linspace(-color_limit, color_limit, 25)
    figure, axes = plt.subplots(
        2, 3, figsize=(13.4, 8.0), constrained_layout=True
    )
    axes = axes.ravel()
    _draw_geometry(axes[0])
    horizontal_contour = _draw_flow(
        axes[1], primary, color_norm, levels
    )
    vertical_contour = _draw_vertical_velocity(axes[2], primary)
    j_image = _draw_inlet_j_map(axes[3], primary)

    summary_axis = axes[4]
    positions = np.arange(len(results), dtype=float)
    horizontal_minima = np.array(
        [result["diagnostics"]["minimum J, m0=0"] for result in results]
    )
    bar_colors = np.where(horizontal_minima < 0.0, "#b2182b", "#2166ac")
    summary_axis.bar(
        positions,
        horizontal_minima,
        width=0.68,
        color=bar_colors,
        alpha=0.88,
    )
    summary_axis.axhline(0.0, color="0.25", linewidth=0.8)
    summary_axis.set_xticks(
        positions,
        (
            "open\ntraction",
            "matched\ninflows",
            "closed\nleft",
            "uniform\n(open)",
            "full-right\ninflux",
        ),
    )
    summary_axis.tick_params(axis="x", labelsize=7.5)
    for position, value in zip(positions, horizontal_minima):
        summary_axis.text(
            position,
            value + (0.035 if value >= 0.0 else -0.035),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=7.5,
        )
    summary_axis.set(
        ylim=(-0.47, 0.12),
        ylabel="Minimum $J$ for horizontal incoming layers",
        title="Thin-left counterflow permits folding",
    )
    _draw_tangent_history(axes[5], primary)

    for label, axis in zip("ABCDEF", axes):
        axis.text(
            -0.10,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )
    horizontal_colorbar = figure.colorbar(
        horizontal_contour,
        ax=axes[1],
        orientation="horizontal",
        shrink=0.9,
        pad=0.10,
        ticks=np.linspace(-color_limit, color_limit, 5),
    )
    horizontal_colorbar.set_label("$u$ (m yr$^{-1}$)")
    vertical_colorbar = figure.colorbar(
        vertical_contour,
        ax=axes[2],
        orientation="horizontal",
        shrink=0.9,
        pad=0.10,
        ticks=np.linspace(
            vertical_contour.norm.vmin, vertical_contour.norm.vmax, 5
        ),
    )
    vertical_colorbar.set_label("$w$ (m yr$^{-1}$)")
    j_colorbar = figure.colorbar(
        j_image,
        ax=axes[3],
        orientation="horizontal",
        shrink=0.9,
        pad=0.10,
        extend="both",
        ticks=np.linspace(j_image.norm.vmin, j_image.norm.vmax, 5),
    )
    j_colorbar.set_label("Minimum $J$")
    figure.suptitle(
        "Idealized Allan Hills: can thick-to-thin convergence fold layers?",
        fontsize=14,
    )
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIRECTORY / "allan_hills_idealized_comparison.png"
    pdf = FIGURE_DIRECTORY / "allan_hills_idealized_comparison.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def _markdown_table(frame: pd.DataFrame, columns, formats):
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            formatter = formats.get(column, str)
            values.append(formatter(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join((header, separator, *rows))


def write_report(case_frame: pd.DataFrame, source_frame: pd.DataFrame, args):
    overturned_horizontal = case_frame[case_frame["minimum J, m0=0"] < 0.0]
    overturned_surface = case_frame[
        case_frame["minimum J, m0=m_right"] < 0.0
    ]

    def names(frame):
        if frame.empty:
            return "none"
        return ", ".join(frame["title"].tolist())

    table = _markdown_table(
        case_frame,
        (
            "title",
            "minimum J, m0=0",
            "minimum J, m0=m_right",
            "left outward flux (m2/yr)",
            "computed right outward flux (m2/yr)",
            "surface outward flux (m2/yr)",
            "age-capped paths",
            "maximum abs detF-1",
        ),
        {
            "title": str,
            "minimum J, m0=0": lambda v: f"{v:.4f}",
            "minimum J, m0=m_right": lambda v: f"{v:.4f}",
            "left outward flux (m2/yr)": lambda v: f"{v:.3f}",
            "computed right outward flux (m2/yr)": lambda v: f"{v:.3f}",
            "surface outward flux (m2/yr)": lambda v: f"{v:.3f}",
            "age-capped paths": lambda v: f"{int(v)}",
            "maximum abs detF-1": lambda v: f"{v:.2e}",
        },
    )
    maximum_budget_error = source_frame["J budget closure at minimum"].abs().max()
    primary = case_frame.loc[
        case_frame["case"] == "two_sided_variable"
    ].iloc[0]
    matched = case_frame.loc[
        case_frame["case"] == "matched_two_sided_variable"
    ].iloc[0]
    closed = case_frame.loc[
        case_frame["case"] == "closed_left_variable"
    ].iloc[0]
    full_right = case_frame.loc[
        case_frame["case"] == "closed_left_full_inflow"
    ].iloc[0]
    primary_horizontal = source_frame[
        (source_frame["case"] == "two_sided_variable")
        & source_frame["is primary horizontal"]
    ]
    negative_primary = primary_horizontal[
        primary_horizontal["minimum J"] < 0.0
    ]
    primary_minimum = primary_horizontal.loc[
        primary_horizontal["minimum J"].idxmin()
    ]

    validation_path = DATA_DIRECTORY / "allan_hills_idealized_validation.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        validation_table = _markdown_table(
            validation,
            (
                "variant",
                "source count",
                "bed buffer sigma",
                "minimum J, m0=0",
                "source sigma at minimum J, m0=0",
                "negative horizontal sources",
                "bed-buffer terminations",
                "rms right-boundary w (m/yr)",
                "maximum abs detF-1",
                "relative flux closure",
            ),
            {
                "variant": str,
                "source count": lambda v: f"{int(v)}",
                "bed buffer sigma": lambda v: f"{v:.5g}",
                "minimum J, m0=0": lambda v: f"{v:.5f}",
                "source sigma at minimum J, m0=0": lambda v: f"{v:.5f}",
                "negative horizontal sources": lambda v: f"{int(v)}",
                "bed-buffer terminations": lambda v: f"{int(v)}",
                "rms right-boundary w (m/yr)": lambda v: f"{v:.2e}",
                "maximum abs detF-1": lambda v: f"{v:.2e}",
                "relative flux closure": lambda v: f"{v:.2e}",
            },
        )
        validation_rows = validation.set_index("variant")
        production_validation = validation_rows.loc["production"]
        dense_validation = validation_rows.loc["production-dense-depth"]
        refined_validation = validation_rows.loc["refined"]
        buffer_high = validation_rows.loc["buffer-0.001"]
        buffer_low = validation_rows.loc["buffer-0.00025"]
        validation_note = (
            "At the fixed source cutoff "
            f"$\\sigma_0={SOURCE_SIGMA_MIN:g}$, the production, doubled "
            "source-density, and refined-mesh values are "
            f"{production_validation['minimum J, m0=0']:.5f}, "
            f"{dense_validation['minimum J, m0=0']:.5f}, and "
            f"{refined_validation['minimum J, m0=0']:.5f}. Moving the "
            "bed buffer from "
            f"$\\sigma={buffer_high['bed buffer sigma']:.4g}$ to "
            f"$\\sigma={buffer_low['bed buffer sigma']:.4g}$ changes the "
            "minimum from "
            f"{buffer_high['minimum J, m0=0']:.5f} to "
            f"{buffer_low['minimum J, m0=0']:.5f}."
        )
    else:
        validation_table = "Validation file not yet generated."
        validation_note = "Validation file not yet generated."
    report = f"""# Idealized Allan Hills steady-flow experiment

## Question and setup

This calculation tests whether an idealized thin-left/thick-right geometry, together with leftward influx from the thick side and alternative treatments of the thin-left boundary, can overturn material lines. It is a two-dimensional, fixed-geometry, incompressible, linear-viscous full-Stokes calculation, not a free-surface evolution.

The domain is {LENGTH / 1000:.1f} km long. The variable-thickness geometry grows smoothly from {LEFT_THICKNESS:.0f} to {RIGHT_THICKNESS:.0f} m; the surface slope transitions from {LEFT_SURFACE_SLOPE:.3f} to {RIGHT_SURFACE_SLOPE:.3f}. The bed has no penetration and linear Robin drag $\\beta={BASAL_DRAG:.2e}$ Pa yr m$^{{-1}}$, and the surface is traction free. At the right boundary, $u$ is prescribed using an $n=1$ SIA-shaped profile scaled to the target flux and $w=0$ is imposed explicitly. The reference right flux is {REFERENCE_INFLOW_FLUX:.3f} m$^2$ yr$^{{-1}}$, and the reduced cases use {100.0 * REDUCED_INFLOW_FRACTION:.0f}% of it.

The thickness construction reaches {MID_THICKNESS:.0f} m through a smooth 0--{FIRST_THICKENING_END / 1000.0:.1f} km transition, then adds a second transition beginning at {SECOND_THICKENING_START / 1000.0:.1f} km. That second ramp formally ends at {SECOND_THICKENING_END / 1000.0:.1f} km outside the domain and is normalized to give exactly {RIGHT_THICKNESS:.0f} m at the right boundary. Consequently, the bed is still descending at the right inlet rather than turning upward there.

The principal case uses an open left boundary with hydrostatic-plus-shear traction corresponding to a reference parallel slab. This condition does not prescribe the left flux: its magnitude and sign are outputs. A second case prescribes an SIA-shaped horizontal left influx exactly matched in magnitude to the right influx. Closed-left cases impose $u=0$ at the left boundary. The uniform-thickness run is only a sensitivity case because changing its left thickness also changes the open-boundary response.

The surface mass balance needed to keep the surface fixed is diagnosed as $a_s=u s_x-w$ and checked against the exact finite-element surface-normal flux. Boundary fluxes below use the outward normal: negative lateral flux is inflow, while positive surface-outward flux requires negative steady surface mass balance (net ablation).

## Steady material-line diagnostic

Along {args.sources} right-inlet depth sources, the code integrates $\\dot F=\\nabla u F$ from $F=I$. For each incoming tangent $(1,m_0)$ it evaluates $J(m_0)=F_{{00}}+m_0F_{{01}}$. Negative $J$ is overturning. The reported decomposition integrates $u_xT_x$ and $u_zT_z$ separately, and `det(F)` checks incompressibility along paths.

The source coordinate is $\\sigma_0=(z_0-b_R)/h_R$: $\\sigma_0=0$ is the right-inlet bed and $\\sigma_0=1$ is the right-inlet surface. It is a normalized source height, not either boundary thickness $h_L$ or $h_R$.

{table}

For initially horizontal material lines, overturning occurs in: **{names(overturned_horizontal)}**. For material lines initially parallel to the right surface, overturning occurs in: **{names(overturned_surface)}**.

## What the experiment shows

The open-left variable-thickness case overturns: $J_{{min}}(m_0=0)={primary['minimum J, m0=0']:.4f}$ at {primary['age at minimum J, m0=0 (yr)'] / 1000.0:.2f} kyr. {len(negative_primary)} of {len(primary_horizontal)} sampled inlet depths overturn for horizontal incoming layers. The reported minimum comes from the explicit basal-source cutoff $\\sigma_0={primary_minimum['source sigma']:.5f}$ and occurs at $x={primary_minimum['x at minimum J (m)'] / 1000.0:.3f}$ km, local $\\sigma={primary_minimum['sigma at minimum J']:.3f}$; it is therefore a cutoff-conditioned screening value, not a claimed continuum-global minimum. The tangent budget identifies the mechanism: vertical shear contributes {primary['uz Tz contribution at minimum, m0=0']:.3f}, while longitudinal deformation contributes {primary['ux Tx contribution at minimum, m0=0']:+.3f} and therefore opposes the fold.

The flux-matched two-sided case also overturns ($J_{{min}}={matched['minimum J, m0=0']:.4f}$), so the positive result does not depend on using the open-traction boundary. In contrast, closing the thin-left boundary leaves the same variable-thickness geometry ordered ($J_{{min}}={closed['minimum J, m0=0']:.4f}$), and even the full reference right influx with a closed left remains ordered ($J_{{min}}={full_right['minimum J, m0=0']:.4f}$). Thus thickness contrast plus right-side influx is not sufficient in these tests; thin-left counterflow and the resulting two-sided convergence are the decisive ingredients.

The principal case imports {-primary['left outward flux (m2/yr)']:.3f} m$^2$ yr$^{{-1}}$ from the left and {-primary['computed right outward flux (m2/yr)']:.3f} m$^2$ yr$^{{-1}}$ from the right, balanced by {primary['surface outward flux (m2/yr)']:.3f} m$^2$ yr$^{{-1}}$ of surface outflux. The solved RMS right-boundary vertical velocity is {primary['rms right-boundary w (m/yr)']:.2e} m yr$^{{-1}}$. Its left quarter is not uniformly downwelling: {100.0 * primary['fraction w < 0 in left quarter']:.0f}% of sampled points have $w<0$, while the mean is {primary['mean w in left quarter (m/yr)']:.3e} m yr$^{{-1}}$. The idealized vertical-velocity field therefore only partly reproduces the attached 3-D section.

The largest tangent-budget closure error is {maximum_budget_error:.2e}.

## Resolution and source-depth validation

{validation_table}

{validation_note}

## Interpretation limits

This is a sufficient-mechanism test, not a complete representation of Allan Hills. The specialized calculation satisfies $u_x+v_y+w_z=0$, whereas this section sets the omitted lateral strain $v_y$ to zero. A positive 2-D result shows that the represented cross-sectional convergence can suffice; a negative result would not rule out folding driven by 3-D lateral convergence. The incoming tangent $(1,m_0)$ is assumed boundary stratigraphy, and the depth sources are trajectories carrying that local deformation diagnostic, not a claim that a simultaneous isochrone lies along the vertical inlet boundary. The open-left traction is a reference mechanical boundary condition, not an observational constraint. The fixed geometry also requires the diagnosed surface mass balance.

## Numerical settings

Mesh: {args.nx} by {args.nz}; terrain sampler: {args.sample_nx} by {args.sample_nz} over $\\sigma=[{args.sigma_lower:g},{args.sigma_upper:g}]$; trajectory sources: {args.sources} uniformly spaced over $\\sigma_0=[{args.source_sigma_min:g},{args.source_sigma_max:g}]$; requested RK4 step: {args.time_step:g} yr (automatically reduced by a terrain-grid CFL bound); maximum age: {args.maximum_age:g} yr; incoming-slope range: $[-{args.slope_limit:g},{args.slope_limit:g}]$.
"""
    REPORT_PATH.write_text(report)


# Command-line driver -------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0]
    )
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--sample-nx", type=int, default=501)
    parser.add_argument("--sample-nz", type=int, default=161)
    parser.add_argument("--sources", type=int, default=193)
    parser.add_argument("--maximum-age", type=float, default=500_000.0)
    parser.add_argument("--time-step", type=float, default=5.0)
    parser.add_argument("--slope-limit", type=float, default=0.10)
    parser.add_argument("--slope-count", type=int, default=41)
    parser.add_argument(
        "--sigma-lower", type=float, default=SAMPLER_SIGMA_LOWER
    )
    parser.add_argument(
        "--sigma-upper", type=float, default=SAMPLER_SIGMA_UPPER
    )
    parser.add_argument(
        "--source-sigma-min", type=float, default=SOURCE_SIGMA_MIN
    )
    parser.add_argument(
        "--source-sigma-max", type=float, default=SOURCE_SIGMA_MAX
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a coarse smoke-test mesh and shorter trajectory window.",
    )
    return parser.parse_args()


def validate_arguments(args):
    if args.quick:
        args.nx = 80
        args.nz = 16
        args.sample_nx = 241
        args.sample_nz = 51
        args.sources = 13
        args.maximum_age = min(args.maximum_age, 8_000.0)
        args.time_step = max(args.time_step, 3.0)
        args.slope_count = 21
    for name in ("nx", "nz", "sample_nx", "sample_nz", "sources"):
        if getattr(args, name) < 3:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 3")
    if args.maximum_age <= 0.0 or args.time_step <= 0.0:
        raise ValueError("maximum age and time step must be positive")
    if args.slope_limit <= 0.0 or args.slope_count < 3:
        raise ValueError("incoming-slope grid must have positive extent")
    if not 0.0 < args.sigma_lower < args.sigma_upper < 1.0:
        raise ValueError("require 0 < --sigma-lower < --sigma-upper < 1")
    if not (
        args.sigma_lower
        < args.source_sigma_min
        < args.source_sigma_max
        < args.sigma_upper
    ):
        raise ValueError(
            "source-sigma limits must lie strictly inside sampler limits"
        )
    return args


def main():
    args = validate_arguments(parse_arguments())
    require_firedrake()
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    results = []
    case_rows = []
    source_frames = []
    surface_frames = []
    for case in CASES:
        result = run_case(case, args)
        results.append(result)
        case_rows.append(result["diagnostics"])
        source_frames.append(result["source frame"])
        surface_frames.append(result["surface frame"])

        # Incremental screen files preserve completed cases if a later solve is
        # interrupted.  The final write below has the same paths and schema.
        pd.DataFrame(case_rows).to_csv(
            DATA_DIRECTORY / "allan_hills_idealized_case_screen.csv",
            index=False,
        )
        pd.concat(source_frames, ignore_index=True).to_csv(
            DATA_DIRECTORY / "allan_hills_idealized_source_screen.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.concat(surface_frames, ignore_index=True).to_csv(
            DATA_DIRECTORY / "allan_hills_idealized_surface_balance.csv",
            index=False,
        )

    case_frame = pd.DataFrame(case_rows)
    source_frame = pd.concat(source_frames, ignore_index=True)
    make_comparison_figure(results)
    write_report(case_frame, source_frame, args)
    print("\nCase screen:\n", case_frame.to_string(index=False), flush=True)
    print(
        "\nWrote Allan Hills case/surface CSVs, compressed source screen, "
        "comparison.{png,pdf}, and report.md.",
        flush=True,
    )


if __name__ == "__main__":
    main()
