"""Steady full-Stokes experiment with a true Allan Hills nunatak margin.

The earlier thin-left experiment ended the ice at a finite vertical cut.  That
is not the geometry of exposed rock.  Here the bed and free surface meet at a
single point at x=0, so there is no left boundary facet, no prescribed left
flux, and no source of ice through the nunatak.  The fixed right boundary has
the same SIA-shaped horizontal influx and w=0 condition as the companion
experiment.  Surface mass balance is diagnosed after the solve.

The point-margin mesh is unstructured and triangular.  A family of positive-
thickness truncations is also available solely as a numerical convergence
test; their shrinking artificial face is traction free and carries no
prescribed flux.  Material-line diagnostics exclude a stated thickness near
the contact point, where terrain coordinates are singular.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

try:
    import firedrake as fd
    import gmsh
except ImportError:
    fd = None
    gmsh = None

import allan_hills_idealized_flow as core


warnings.filterwarnings("ignore", message="Unable to import recommended hash")
warnings.filterwarnings("ignore", message="The .*Function.at.* method")
logging.getLogger("tsfc").setLevel(logging.ERROR)


LENGTH = core.LENGTH
RIGHT_THICKNESS = core.RIGHT_THICKNESS
THICKENING_LENGTH = 600.0
DEFAULT_INFLOW_FRACTION = core.REDUCED_INFLOW_FRACTION
DEFAULT_CHARACTERISTIC_RESOLUTION = 20.0
DEFAULT_TIP_RESOLUTION = 4.0
DEFAULT_DIAGNOSTIC_THICKNESS_CUTOFF = 25.0

SECONDS_PER_YEAR = 365.25 * 86400.0
GLEN_EXPONENT = 3.0
GLEN_FLUIDITY_PA_S = 6.8e-26  # Pa^-3 s^-1; Cuffey & Paterson (2010), T = -25 C
GLEN_FLUIDITY = GLEN_FLUIDITY_PA_S * SECONDS_PER_YEAR  # Pa^-3 yr^-1
GLEN_STRAIN_RATE_REG = 1.0e-10  # yr^-1

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = ROOT / "data"
FIGURE_DIRECTORY = ROOT / "figures"
REPORT_PATH = ROOT / "notebooks/allan_hills_nunatak_margin_report.md"


@dataclass(frozen=True)
class MarginConfiguration:
    """One mesh/diagnostic configuration for the margin-limit study."""

    key: str
    title: str
    left_thickness: float
    resolution: float
    tip_resolution: float
    diagnostic_thickness_cutoff: float
    inflow_fraction: float = DEFAULT_INFLOW_FRACTION
    store_history: bool = False
    surface_mode: str = "traction_free"
    rheology: str = "linear"


def require_environment():
    core.require_firedrake()
    if gmsh is None:
        raise RuntimeError("Gmsh is required for the point-margin mesh")


def ice_thickness(x, thickening_length=THICKENING_LENGTH):
    """Finite-angle thickness profile with H(0)=0 and H(L)=H_R."""
    values = np.asarray(x, dtype=float)
    denominator = 1.0 - np.exp(-LENGTH / thickening_length)
    return RIGHT_THICKNESS * (
        1.0 - np.exp(-values / thickening_length)
    ) / denominator


def ice_thickness_slope(x, thickening_length=THICKENING_LENGTH):
    values = np.asarray(x, dtype=float)
    denominator = 1.0 - np.exp(-LENGTH / thickening_length)
    return (
        RIGHT_THICKNESS
        * np.exp(-values / thickening_length)
        / (thickening_length * denominator)
    )


def x_at_thickness(thickness, thickening_length=THICKENING_LENGTH):
    """Invert the monotone margin thickness profile."""
    thickness = float(thickness)
    if not 0.0 <= thickness < RIGHT_THICKNESS:
        if np.isclose(thickness, RIGHT_THICKNESS):
            return LENGTH
        raise ValueError("thickness must lie in [0, H_R]")
    denominator = 1.0 - np.exp(-LENGTH / thickening_length)
    return float(
        -thickening_length
        * np.log(1.0 - thickness * denominator / RIGHT_THICKNESS)
    )


def surface_elevation(x):
    return core.surface_elevation(x)


def bed_elevation(x):
    return surface_elevation(x) - ice_thickness(x)


def _boundary_x_values(x_left, count=181):
    """Resolve the contact point while retaining the full 2 km section."""
    fraction = np.linspace(0.0, 1.0, count)
    stretched = fraction**1.45
    return x_left + (LENGTH - x_left) * stretched


def make_margin_mesh(configuration: MarginConfiguration):
    """Create a triangular wedge or a positive-thickness truncation."""
    x_left = x_at_thickness(configuration.left_thickness)
    x_values = _boundary_x_values(x_left)
    mesh_file = Path(tempfile.NamedTemporaryFile(suffix=".msh", delete=False).name)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.add(f"nunatak_{configuration.key}")

        def local_size(x_value):
            transition = 1.0 - np.exp(-max(x_value - x_left, 0.0) / 160.0)
            return configuration.tip_resolution + (
                configuration.resolution - configuration.tip_resolution
            ) * transition

        if configuration.left_thickness == 0.0:
            margin = gmsh.model.geo.addPoint(
                0.0,
                float(surface_elevation(0.0)),
                0.0,
                configuration.tip_resolution,
            )
            bed_points = [margin]
            surface_points = [margin]
            start = 1
        else:
            bed_points = []
            surface_points = []
            start = 0

        for x_value in x_values[start:]:
            size = local_size(float(x_value))
            bed_points.append(
                gmsh.model.geo.addPoint(
                    float(x_value),
                    float(bed_elevation(x_value)),
                    0.0,
                    size,
                )
            )
            surface_points.append(
                gmsh.model.geo.addPoint(
                    float(x_value),
                    float(surface_elevation(x_value)),
                    0.0,
                    size,
                )
            )

        bed_curve = gmsh.model.geo.addSpline(bed_points)
        right_curve = gmsh.model.geo.addLine(
            bed_points[-1], surface_points[-1]
        )
        surface_curve = gmsh.model.geo.addSpline(surface_points[::-1])
        curves = [bed_curve, right_curve, surface_curve]
        left_curve = None
        if configuration.left_thickness > 0.0:
            left_curve = gmsh.model.geo.addLine(
                surface_points[0], bed_points[0]
            )
            curves.append(left_curve)
        loop = gmsh.model.geo.addCurveLoop(curves)
        area = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        if left_curve is not None:
            gmsh.model.addPhysicalGroup(1, [left_curve], 1)
        gmsh.model.addPhysicalGroup(1, [right_curve], 2)
        gmsh.model.addPhysicalGroup(1, [bed_curve], 3)
        gmsh.model.addPhysicalGroup(1, [surface_curve], 4)
        gmsh.model.addPhysicalGroup(2, [area], 1)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh_file))
    finally:
        gmsh.finalize()

    linear_mesh = fd.Mesh(str(mesh_file))
    mesh_file.unlink(missing_ok=True)
    coordinate_space = fd.VectorFunctionSpace(linear_mesh, "CG", 2)
    coordinates = fd.Function(coordinate_space).interpolate(
        fd.SpatialCoordinate(linear_mesh)
    )
    zero = fd.Constant((0.0, 0.0))
    bed_nodes = fd.DirichletBC(coordinate_space, zero, 3).nodes
    surface_nodes = fd.DirichletBC(coordinate_space, zero, 4).nodes
    coordinates.dat.data[bed_nodes, 1] = bed_elevation(
        coordinates.dat.data[bed_nodes, 0]
    )
    coordinates.dat.data[surface_nodes, 1] = surface_elevation(
        coordinates.dat.data[surface_nodes, 0]
    )
    return fd.Mesh(coordinates), x_left


def glen_sia_flux(
    thickness: float,
    slope: float,
    fluidity: float = GLEN_FLUIDITY,
    exponent: float = GLEN_EXPONENT,
) -> float:
    """Depth-integrated horizontal flux of an isotropic Glen slab with linear slip."""
    drive = core.DENSITY * core.GRAVITY * slope
    drive_abs = abs(drive)
    return float(
        -drive * thickness**2 / core.BASAL_DRAG
        - (2.0 * fluidity / (exponent + 2.0))
        * drive_abs ** (exponent - 1.0)
        * drive
        * thickness ** (exponent + 2.0)
    )


def _glen_sia_profile_ufl(
    y,
    thickness: float,
    slope: float,
    fluidity: float,
    exponent: float,
):
    drive = core.DENSITY * core.GRAVITY * slope
    drive_abs = abs(drive)
    return -drive * thickness / core.BASAL_DRAG - (
        2.0
        * fluidity
        / (exponent + 1.0)
        * drive_abs ** (exponent - 1.0)
        * drive
        * (thickness ** (exponent + 1.0) - (thickness - y) ** (exponent + 1.0))
    )


def _reference_inflow_flux(rheology: str) -> float:
    if rheology == "linear":
        return core.REFERENCE_INFLOW_FLUX
    if rheology == "glen":
        return glen_sia_flux(RIGHT_THICKNESS, core.RIGHT_SURFACE_SLOPE)
    raise ValueError(f"unknown rheology {rheology!r}")


def _right_inflow_horizontal(mesh, configuration):
    """SIA-plus-slip horizontal inflow profile matching the active rheology."""
    _, z = fd.SpatialCoordinate(mesh)
    h_right = RIGHT_THICKNESS
    b_right = float(bed_elevation(LENGTH))
    y = z - b_right
    if configuration.rheology == "linear":
        drive = core.DENSITY * core.GRAVITY * core.RIGHT_SURFACE_SLOPE
        raw_u = -drive * (
            h_right / core.BASAL_DRAG
            + (h_right * y - 0.5 * y**2) / core.VISCOSITY
        )
        raw_flux = core.raw_sia_flux(h_right, core.RIGHT_SURFACE_SLOPE)
        target_flux = configuration.inflow_fraction * core.REFERENCE_INFLOW_FLUX
        return target_flux / raw_flux * raw_u
    if configuration.rheology == "glen":
        raw_u = _glen_sia_profile_ufl(
            y, h_right, core.RIGHT_SURFACE_SLOPE, GLEN_FLUIDITY, GLEN_EXPONENT
        )
        raw_flux = glen_sia_flux(h_right, core.RIGHT_SURFACE_SLOPE)
        target_flux = configuration.inflow_fraction * raw_flux
        return target_flux / raw_flux * raw_u
    raise ValueError(f"unknown rheology {configuration.rheology!r}")


def _cofunction_norm(form, bcs=None) -> float:
    assembled = fd.assemble(form)
    if bcs:
        for bc in bcs:
            bc.zero(assembled)
    return float(
        np.sqrt(
            sum(
                np.dot(part.ravel(), part.ravel())
                for part in assembled.dat.data_ro
            )
        )
    )


def _cell_diameter_field(mesh):
    """DG0 maximum node-to-node distance for the curved P2 cells."""
    coordinate_nodes = (
        mesh.coordinates.function_space().cell_node_map().values
    )
    cell_coordinates = mesh.coordinates.dat.data_ro[coordinate_nodes]
    differences = (
        cell_coordinates[:, :, None, :] - cell_coordinates[:, None, :, :]
    )
    diameters = np.sqrt(np.max(np.sum(differences**2, axis=-1), axis=(1, 2)))
    space = fd.FunctionSpace(mesh, "DG", 0)
    field = fd.Function(space, name="local cell diameter")
    field.dat.data[:] = diameters
    return field


def solve_stokes(configuration: MarginConfiguration):
    """Solve the point-margin or truncated-margin Stokes problem."""
    mesh, x_left = make_margin_mesh(configuration)
    velocity_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = fd.FunctionSpace(mesh, "CG", 1)
    mixed_space = velocity_space * pressure_space
    solution = fd.Function(mixed_space)
    u, p = fd.split(solution)
    v, q = fd.TestFunctions(mixed_space)
    body_force = fd.Constant((0.0, -core.DENSITY * core.GRAVITY))
    normal = fd.FacetNormal(mesh)
    tangent = fd.as_vector((-normal[1], normal[0]))
    normal_speed = fd.dot(u, normal)
    test_normal_speed = fd.dot(v, normal)
    cell_diameter = _cell_diameter_field(mesh)
    reference_flux = _reference_inflow_flux(configuration.rheology)
    target_flux = configuration.inflow_fraction * reference_flux

    prescribed_surface_balance = np.nan
    target_surface_normal_speed = None
    if configuration.surface_mode == "uniform_ablation":
        # For a fixed surface, a_s = u*s_x-w and therefore
        # u.n = -a_s/sqrt(1+s_x^2).  Choose uniform a_s so that its
        # integrated outward flux exactly balances the right influx.  The
        # normal velocity is imposed weakly while tangential traction remains
        # natural (zero).
        prescribed_surface_balance = target_flux / LENGTH
        x_coordinate = fd.SpatialCoordinate(mesh)[0]
        midpoint = 0.5 * (
            core.LEFT_SURFACE_SLOPE + core.RIGHT_SURFACE_SLOPE
        )
        amplitude = 0.5 * (
            core.RIGHT_SURFACE_SLOPE - core.LEFT_SURFACE_SLOPE
        )
        surface_slope_ufl = midpoint + amplitude * fd.tanh(
            (x_coordinate - core.SURFACE_TRANSITION_X)
            / core.SURFACE_TRANSITION_WIDTH
        )
        target_surface_normal_speed = -fd.Constant(
            prescribed_surface_balance
        ) / fd.sqrt(1.0 + surface_slope_ufl**2)
    elif configuration.surface_mode != "traction_free":
        raise ValueError(
            f"unknown surface mode {configuration.surface_mode!r}"
        )

    if configuration.rheology == "linear":
        stress = 2.0 * core.VISCOSITY * fd.sym(fd.grad(u)) - p * fd.Identity(2)
        test_stress = (
            2.0 * core.VISCOSITY * fd.sym(fd.grad(v)) - q * fd.Identity(2)
        )
        residual = (
            2.0
            * core.VISCOSITY
            * fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v)))
            - p * fd.div(v)
            - q * fd.div(u)
            - fd.dot(body_force, v)
        ) * fd.dx
        residual += (
            core.BASAL_DRAG * fd.dot(u, tangent) * fd.dot(v, tangent)
            - fd.dot(stress * normal, normal) * test_normal_speed
            - fd.dot(test_stress * normal, normal) * normal_speed
            + core.NITSCHE_PENALTY
            * core.VISCOSITY
            / cell_diameter
            * normal_speed
            * test_normal_speed
        ) * fd.ds(3)
        if target_surface_normal_speed is not None:
            normal_mismatch = normal_speed - target_surface_normal_speed
            residual += (
                -fd.dot(stress * normal, normal) * test_normal_speed
                - fd.dot(test_stress * normal, normal) * normal_mismatch
                + core.NITSCHE_PENALTY
                * core.VISCOSITY
                / cell_diameter
                * normal_mismatch
                * test_normal_speed
            ) * fd.ds(4)
        boundary_conditions = [
            fd.DirichletBC(
                mixed_space.sub(0).sub(0),
                _right_inflow_horizontal(mesh, configuration),
                2,
            ),
            fd.DirichletBC(
                mixed_space.sub(0).sub(1), fd.Constant(0.0), 2
            ),
        ]
        fd.solve(
            residual == 0,
            solution,
            bcs=boundary_conditions,
            solver_parameters=core.SOLVER_PARAMETERS,
        )
    elif configuration.rheology == "glen":
        n_const = fd.Constant(1.0)
        a_const = fd.Constant(1.0 / (2.0 * core.VISCOSITY))
        _, z_coord = fd.SpatialCoordinate(mesh)
        y_right = z_coord - float(bed_elevation(LENGTH))
        inflow_fn = fd.Function(velocity_space.sub(0))
        boundary_conditions = [
            fd.DirichletBC(mixed_space.sub(0).sub(0), inflow_fn, 2),
            fd.DirichletBC(mixed_space.sub(0).sub(1), fd.Constant(0.0), 2),
        ]

        def glen_eta(vel_field):
            strain = fd.sym(fd.grad(vel_field))
            d_eff_sq = 0.5 * fd.inner(strain, strain) + fd.Constant(
                GLEN_STRAIN_RATE_REG**2
            )
            return (
                0.5
                * a_const ** (-1.0 / n_const)
                * d_eff_sq ** ((1.0 - n_const) / (2.0 * n_const))
            )

        def build_residual(eta_field, constant_penalty: bool = False):
            strain_u = fd.sym(fd.grad(u))
            strain_v = fd.sym(fd.grad(v))
            stress_field = 2.0 * eta_field * strain_u - p * fd.Identity(2)
            test_stress_field = (
                2.0 * eta_field * strain_v - q * fd.Identity(2)
            )
            penalty_visc = (
                fd.Constant(core.VISCOSITY) if constant_penalty else eta_field
            )
            form = (
                2.0 * eta_field * fd.inner(strain_u, strain_v)
                - p * fd.div(v)
                - q * fd.div(u)
                - fd.dot(body_force, v)
            ) * fd.dx + (
                core.BASAL_DRAG * fd.dot(u, tangent) * fd.dot(v, tangent)
                - fd.dot(stress_field * normal, normal) * test_normal_speed
                - fd.dot(test_stress_field * normal, normal) * normal_speed
                + core.NITSCHE_PENALTY
                * penalty_visc
                / cell_diameter
                * normal_speed
                * test_normal_speed
            ) * fd.ds(3)
            if target_surface_normal_speed is not None:
                normal_mismatch = normal_speed - target_surface_normal_speed
                form += (
                    -fd.dot(stress_field * normal, normal) * test_normal_speed
                    - fd.dot(test_stress_field * normal, normal)
                    * normal_mismatch
                    + core.NITSCHE_PENALTY
                    * penalty_visc
                    / cell_diameter
                    * normal_mismatch
                    * test_normal_speed
                ) * fd.ds(4)
            return form

        def set_continuation(n_value: float):
            n_const.assign(n_value)
            alpha_h = (n_value - 1.0) / (GLEN_EXPONENT - 1.0)
            a_step = (1.0 / (2.0 * core.VISCOSITY)) ** (
                1.0 - alpha_h
            ) * GLEN_FLUIDITY**alpha_h
            a_const.assign(a_step)
            raw_u_step = _glen_sia_profile_ufl(
                y_right,
                RIGHT_THICKNESS,
                core.RIGHT_SURFACE_SLOPE,
                a_step,
                n_value,
            )
            raw_flux_step = glen_sia_flux(
                RIGHT_THICKNESS, core.RIGHT_SURFACE_SLOPE, a_step, n_value
            )
            inflow_fn.interpolate(target_flux / raw_flux_step * raw_u_step)

        use_constant_penalty = target_surface_normal_speed is not None
        nonlinear_residual = build_residual(
            glen_eta(u), constant_penalty=use_constant_penalty
        )
        newton_parameters = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_max_it": 50,
            "snes_rtol": 1.0e-10,
            "snes_atol": 1.0e-8,
            **core.SOLVER_PARAMETERS,
        }
        solved_directly = False
        if not use_constant_penalty:
            try:
                for n_step in (1.0, 2.0, GLEN_EXPONENT):
                    set_continuation(n_step)
                    fd.solve(
                        nonlinear_residual == 0,
                        solution,
                        bcs=boundary_conditions,
                        solver_parameters=newton_parameters,
                    )
                solved_directly = True
            except fd.ConvergenceError:
                solution.assign(0.0)

        if not solved_directly:
            u_prev = fd.Function(velocity_space)
            use_picard = fd.Constant(0.0)
            set_continuation(GLEN_EXPONENT)
            f0_norm = max(
                _cofunction_norm(nonlinear_residual, boundary_conditions), 1.0
            )
            picard_eta = (1.0 - use_picard) * fd.Constant(
                core.VISCOSITY
            ) + use_picard * glen_eta(u_prev)
            picard_residual = build_residual(
                picard_eta, constant_penalty=use_constant_penalty
            )
            picard_solver = fd.NonlinearVariationalSolver(
                fd.NonlinearVariationalProblem(
                    picard_residual, solution, bcs=boundary_conditions
                ),
                solver_parameters={
                    "snes_type": "ksponly",
                    **core.SOLVER_PARAMETERS,
                },
            )
            picard_solver.solve()
            u_prev.assign(solution.subfunctions[0])
            use_picard.assign(1.0)
            for _ in range(30):
                picard_solver.solve()
                u_prev.assign(0.35 * u_prev + 0.65 * solution.subfunctions[0])
                if (
                    _cofunction_norm(nonlinear_residual, boundary_conditions)
                    / f0_norm
                    < 1.0e-5
                ):
                    break
            backup = solution.copy(deepcopy=True)
            for linesearch in ("bt", "l2"):
                try:
                    fd.solve(
                        nonlinear_residual == 0,
                        solution,
                        bcs=boundary_conditions,
                        solver_parameters={
                            **newton_parameters,
                            "snes_linesearch_type": linesearch,
                            "snes_atol": 1.0e-10 * f0_norm,
                            "snes_rtol": 1.0e-12,
                            "snes_max_it": 30,
                        },
                    )
                    break
                except fd.ConvergenceError:
                    solution.assign(backup)
    else:
        raise ValueError(f"unknown rheology {configuration.rheology!r}")
    velocity, _ = solution.subfunctions

    boundary_flux = {
        "right": float(fd.assemble(fd.dot(velocity, normal) * fd.ds(2))),
        "bed": float(fd.assemble(fd.dot(velocity, normal) * fd.ds(3))),
        "surface": float(fd.assemble(fd.dot(velocity, normal) * fd.ds(4))),
    }
    if configuration.left_thickness > 0.0:
        boundary_flux["left"] = float(
            fd.assemble(fd.dot(velocity, normal) * fd.ds(1))
        )
    else:
        boundary_flux["left"] = 0.0
    bed_length = float(fd.assemble(fd.Constant(1.0) * fd.ds(3, domain=mesh)))
    right_length = float(fd.assemble(fd.Constant(1.0) * fd.ds(2, domain=mesh)))
    domain_area = float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))
    surface_length = float(
        fd.assemble(fd.Constant(1.0) * fd.ds(4, domain=mesh))
    )
    diagnostics = {
        "case": configuration.key,
        "rheology": configuration.rheology,
        "title": configuration.title,
        "left thickness (m)": configuration.left_thickness,
        "left x (m)": x_left,
        "mesh cells": mesh.num_cells(),
        "resolution (m)": configuration.resolution,
        "tip resolution (m)": configuration.tip_resolution,
        "diagnostic thickness cutoff (m)": (
            configuration.diagnostic_thickness_cutoff
        ),
        "surface mode": configuration.surface_mode,
        "prescribed uniform surface mass balance (m/yr)": (
            prescribed_surface_balance
        ),
        "target right flux (m2/yr)": target_flux,
        "computed right outward flux (m2/yr)": boundary_flux["right"],
        "right flux relative error": (
            boundary_flux["right"] - target_flux
        ) / abs(target_flux),
        "left outward flux (m2/yr)": boundary_flux["left"],
        "bed outward flux (m2/yr)": boundary_flux["bed"],
        "surface outward flux (m2/yr)": boundary_flux["surface"],
        "flux closure (m2/yr)": sum(boundary_flux.values()),
        "rms divergence (1/yr)": np.sqrt(
            float(fd.assemble(fd.div(velocity) ** 2 * fd.dx)) / domain_area
        ),
        "rms basal normal speed (m/yr)": np.sqrt(
            float(fd.assemble(fd.dot(velocity, normal) ** 2 * fd.ds(3)))
            / bed_length
        ),
        "rms right-boundary w (m/yr)": np.sqrt(
            float(fd.assemble(velocity[1] ** 2 * fd.ds(2))) / right_length
        ),
    }
    if configuration.surface_mode == "uniform_ablation":
        diagnostics["rms surface normal-speed error (m/yr)"] = np.sqrt(
            float(
                fd.assemble(
                    (
                        fd.dot(velocity, normal)
                        - target_surface_normal_speed
                    )
                    ** 2
                    * fd.ds(4)
                )
            )
            / surface_length
        )
    else:
        diagnostics["rms surface normal-speed error (m/yr)"] = np.nan
    return mesh, velocity, diagnostics


def _minimum_row(frame, slope):
    subset = frame[np.isclose(frame["incoming slope m0"], slope)]
    return subset.loc[subset["minimum J"].idxmin()]


def run_configuration(configuration: MarginConfiguration, args):
    print(f"Solving {configuration.key}: {configuration.title}", flush=True)
    mesh, velocity, diagnostics = solve_stokes(configuration)
    cutoff_x = x_at_thickness(configuration.diagnostic_thickness_cutoff)
    sampling_left = max(diagnostics["left x (m)"], cutoff_x)
    sampler = core.make_sampler(
        core.CASES[0],
        velocity,
        nx_sample=args.sample_nx,
        nz_sample=args.sample_nz,
        horizontal_margin=args.horizontal_margin,
        sigma_lower=args.sigma_lower,
        sigma_upper=args.sigma_upper,
        domain_left=sampling_left,
        domain_right=LENGTH,
        analytic_bed_function=bed_elevation,
        analytic_surface_function=surface_elevation,
    )
    diagnostics.update(sampler["diagnostics"])
    case = core.Case(
        configuration.key,
        configuration.title,
        True,
        "nunatak_margin",
        configuration.inflow_fraction,
    )
    slopes = np.linspace(-args.slope_limit, args.slope_limit, args.slope_count)
    trace = core.trace_incoming_material(
        case,
        sampler,
        slopes,
        args.sources,
        args.maximum_age,
        args.time_step,
        store_history=configuration.store_history,
        source_sigma_min=args.source_sigma_min,
        source_sigma_max=args.source_sigma_max,
    )
    source_frame = pd.DataFrame(trace["rows"])
    horizontal = _minimum_row(source_frame, 0.0)
    surface_parallel = _minimum_row(
        source_frame, core.RIGHT_SURFACE_SLOPE
    )
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
            "x at minimum J, m0=0 (m)": float(
                horizontal["x at minimum J (m)"]
            ),
            "source sigma at minimum J, m0=0": float(
                horizontal["source sigma"]
            ),
            "minimum J, m0=m_right": float(surface_parallel["minimum J"]),
            "maximum abs detF-1": float(
                np.max(trace["maximum determinant error"])
            ),
            "maximum abs J budget closure": float(
                source_frame["J budget closure at minimum"].abs().max()
            ),
            "trajectory time step (yr)": trace["time step"],
            "left/cutoff exits": int(statuses.get("left_exit", 0)),
            "surface exits": int(statuses.get("surface_exit", 0)),
            "bed-buffer terminations": int(
                statuses.get("bed_buffer", 0)
            ),
            "right exits": int(statuses.get("right_exit", 0)),
            "nonfinite paths": int(statuses.get("nonfinite", 0)),
            "age-capped paths": int(statuses.get("age_cap", 0)),
        }
    )

    sigma = sampler["sigma"]
    thickness = sampler["thickness"](sampler["x"])
    flux_profile = thickness * _trapezoid(
        sampler["velocity"][..., 0], sigma, axis=0
    )
    surface_frame = pd.DataFrame(
        {
            "case": configuration.key,
            "x (m)": sampler["x"],
            "thickness (m)": thickness,
            "depth-integrated horizontal flux (m2/yr)": flux_profile,
            "diagnosed effective surface mass balance (m/yr)": sampler[
                "surface mass balance"
            ],
        }
    )
    print(
        f"  Jmin(0)={diagnostics['minimum J, m0=0']:.5f}; "
        f"left flux={diagnostics['left outward flux (m2/yr)']:.3e}; "
        f"surface flux={diagnostics['surface outward flux (m2/yr)']:.3e}",
        flush=True,
    )
    return {
        "configuration": configuration,
        "case": case,
        "mesh": mesh,
        "velocity": velocity,
        "sampler": sampler,
        "trace": trace,
        "diagnostics": diagnostics,
        "source frame": source_frame,
        "surface frame": surface_frame,
    }


def default_configurations(args):
    return (
        MarginConfiguration(
            "point_margin_coarse",
            "Point margin, coarse mesh",
            0.0,
            30.0,
            6.0,
            args.diagnostic_thickness_cutoff,
            rheology=args.rheology,
        ),
        MarginConfiguration(
            "point_margin",
            "True nunatak point margin",
            0.0,
            args.resolution,
            args.tip_resolution,
            args.diagnostic_thickness_cutoff,
            store_history=True,
            rheology=args.rheology,
        ),
        MarginConfiguration(
            "point_margin_refined",
            "Point margin, refined mesh",
            0.0,
            13.0,
            2.6,
            args.diagnostic_thickness_cutoff,
            rheology=args.rheology,
        ),
        *(
            MarginConfiguration(
                f"truncated_{thickness:g}m".replace(".", "p"),
                f"Traction-free truncation at {thickness:g} m",
                thickness,
                args.resolution,
                args.tip_resolution,
                args.diagnostic_thickness_cutoff,
                rheology=args.rheology,
            )
            for thickness in (20.0, 10.0, 5.0, 2.5)
        ),
        MarginConfiguration(
            "point_margin_uniform_ablation",
            "Point margin, uniform ablation closure",
            0.0,
            args.resolution,
            args.tip_resolution,
            10.0,
            surface_mode="uniform_ablation",
            rheology=args.rheology,
        ),
    )


def _output_suffix(rheology: str) -> str:
    return "" if rheology == "linear" else f"_{rheology}"


def write_outputs(results, rheology: str = "linear"):
    suffix = _output_suffix(rheology)
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    case_frame = pd.DataFrame([result["diagnostics"] for result in results])
    source_frame = pd.concat(
        [result["source frame"] for result in results], ignore_index=True
    )
    surface_frame = pd.concat(
        [result["surface frame"] for result in results], ignore_index=True
    )
    case_frame.to_csv(
        DATA_DIRECTORY / f"allan_hills_nunatak_margin{suffix}_cases.csv",
        index=False,
    )
    source_frame.to_csv(
        DATA_DIRECTORY / f"allan_hills_nunatak_margin{suffix}_sources.csv.gz",
        index=False,
        compression="gzip",
    )
    surface_frame.to_csv(
        DATA_DIRECTORY / f"allan_hills_nunatak_margin{suffix}_flux.csv",
        index=False,
    )
    return case_frame, source_frame, surface_frame


def _draw_geometry(axis, rheology: str = "linear"):
    x = np.linspace(0.0, LENGTH, 1001)
    surface = surface_elevation(x)
    bed = bed_elevation(x)
    rock_x = np.linspace(-260.0, 0.0, 100)
    margin_bed_slope = float(
        core.surface_slope(0.0) - ice_thickness_slope(0.0)
    )
    exposed_rock = float(bed_elevation(0.0)) + margin_bed_slope * rock_x
    floor = min(np.min(bed), np.min(exposed_rock)) - 35.0
    axis.fill_between(rock_x / 1000.0, floor, exposed_rock, color="0.70")
    axis.fill_between(x / 1000.0, floor, bed, color="0.70")
    axis.fill_between(x / 1000.0, bed, surface, color="#d8edf3")
    axis.plot(rock_x / 1000.0, exposed_rock, color="0.15", linewidth=1.2)
    axis.plot(x / 1000.0, bed, color="0.15", linewidth=1.2)
    axis.plot(x / 1000.0, surface, color="#174a6e", linewidth=1.4)
    axis.text(-0.14, 345.0, "exposed\nrock", ha="center", va="center")
    axis.annotate(
        "point margin\n$H=0$, no left flux",
        xy=(0.0, float(surface_elevation(0.0))),
        xytext=(0.24, 325.0),
        arrowprops={"arrowstyle": "->", "linewidth": 0.9},
        ha="left",
        va="center",
    )

    right_bed = float(bed_elevation(LENGTH))
    right_surface = float(surface_elevation(LENGTH))
    profile_z = np.linspace(right_bed + 18.0, right_surface - 18.0, 8)
    profile_y = profile_z - right_bed
    if rheology == "linear":
        profile_speed = (
            RIGHT_THICKNESS / core.BASAL_DRAG
            + (
                RIGHT_THICKNESS * profile_y - 0.5 * profile_y**2
            ) / core.VISCOSITY
        )
        inflow_label = (
            "scaled $n=1$ SIA inflow + slip\n"
            "$u_R=-r_q\\rho g s_x"
            "[H/\\beta+(Hy-y^2/2)/\\eta]$\n"
            "$y=z-b_R$, $r_q=q_R/q_{\\mathrm{SIA}}$; $w_R=0$"
        )
    else:
        drive_abs = abs(
            core.DENSITY * core.GRAVITY * core.RIGHT_SURFACE_SLOPE
        )
        profile_speed = (
            drive_abs * RIGHT_THICKNESS / core.BASAL_DRAG
            + 0.5
            * GLEN_FLUIDITY
            * drive_abs**3
            * (RIGHT_THICKNESS**4 - (RIGHT_THICKNESS - profile_y) ** 4)
        )
        inflow_label = (
            "scaled $n=3$ Glen SIA inflow + slip\n"
            "$u_R=-r_q[\\tau_R/\\beta+(A_{\\mathrm{G}}/2)"
            "\\tau_R^3 H(1-(1-y/H)^4)]$\n"
            "$\\tau_R=\\rho g s_x H$, $y=z-b_R$; $w_R=0$"
        )
    arrow_lengths = 0.22 * profile_speed / np.max(profile_speed)
    profile_tail = 1.985
    profile_heads = profile_tail - arrow_lengths
    axis.plot(
        profile_heads,
        profile_z,
        color="#174a6e",
        linewidth=0.8,
        zorder=4,
    )
    for head_x, z_value in zip(profile_heads, profile_z):
        axis.annotate(
            "",
            xy=(head_x, z_value),
            xytext=(profile_tail, z_value),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#174a6e",
                "linewidth": 0.85,
                "mutation_scale": 7.0,
            },
            zorder=4,
        )
    axis.plot(
        (LENGTH / 1000.0, LENGTH / 1000.0),
        (right_bed, right_surface),
        color="0.15",
        linewidth=1.0,
        zorder=3,
    )
    axis.text(
        1.10,
        252.0,
        inflow_label,
        ha="center",
        va="center",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        zorder=5,
    )
    axis.text(
        0.70,
        405.0,
        "traction-free surface",
        ha="center",
        va="center",
        color="#174a6e",
    )
    axis.text(
        0.56,
        0.04,
        "bed: no penetration + linear sliding",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    axis.set(
        xlim=(-0.26, 2.0),
        ylim=(floor, np.max(surface) + 20.0),
        xlabel="Distance (km)",
        ylabel="Elevation (m)",
        title="Nunatak geometry and boundaries",
    )


def _draw_flow(axis, result, component, title):
    sampler = result["sampler"]
    values = sampler["velocity"][..., component]
    limit = max(float(np.max(np.abs(values))), 1.0e-8)
    levels = np.linspace(-limit, limit, 25)
    contour = axis.contourf(
        sampler["X"] / 1000.0,
        sampler["Z"],
        values,
        levels=levels,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        extend="both",
    )
    axis.fill_between(
        sampler["x"] / 1000.0,
        np.min(sampler["bed"](sampler["x"])) - 30.0,
        sampler["bed"](sampler["x"]),
        color="0.72",
    )
    axis.plot(
        sampler["x"] / 1000.0,
        sampler["bed"](sampler["x"]),
        color="0.15",
        linewidth=1.0,
    )
    axis.plot(
        sampler["x"] / 1000.0,
        sampler["surface"](sampler["x"]),
        color="0.15",
        linewidth=1.0,
    )
    if component == 0:
        history = result["trace"]["history positions"]
        if history.size:
            for source_index in np.linspace(
                0, history.shape[1] - 1, 11, dtype=int
            ):
                path = history[:, source_index]
                valid = np.all(np.isfinite(path), axis=1)
                axis.plot(
                    path[valid, 0] / 1000.0,
                    path[valid, 1],
                    color="#e6b800",
                    linewidth=0.8,
                )
        axis.annotate(
            "flow from right",
            xy=(1.53, 350.0),
            xytext=(1.88, 350.0),
            arrowprops={"arrowstyle": "->", "linewidth": 0.9},
            ha="right",
            va="center",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )
    axis.set(
        xlim=(0.0, 2.0),
        ylabel="Elevation (m)",
        title=title,
    )
    return contour


def _draw_j_map(axis, result):
    frame = result["source frame"]
    grid = frame.pivot(
        index="source sigma", columns="incoming slope m0", values="minimum J"
    )
    values = grid.to_numpy(dtype=float)
    limit = max(0.2, min(1.0, float(np.nanmax(np.abs(values)))))
    image = axis.pcolormesh(
        grid.columns.to_numpy(dtype=float),
        grid.index.to_numpy(dtype=float),
        values,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        shading="nearest",
        rasterized=True,
    )
    if np.nanmin(values) < 0.0 < np.nanmax(values):
        axis.contour(
            grid.columns.to_numpy(dtype=float),
            grid.index.to_numpy(dtype=float),
            values,
            levels=(0.0,),
            colors="0.15",
            linewidths=0.8,
        )
    horizontal = frame[np.isclose(frame["incoming slope m0"], 0.0)]
    minimum = horizontal.loc[horizontal["minimum J"].idxmin()]
    axis.plot(
        0.0,
        minimum["source sigma"],
        marker="o",
        markerfacecolor="none",
        markeredgecolor="#b2182b",
    )
    axis.axvline(0.0, color="0.35", linewidth=0.7, linestyle="--")
    axis.annotate(
        f"horizontal $J_{{min}}={minimum['minimum J']:.3f}$",
        xy=(0.0, minimum["source sigma"]),
        xytext=(0.012, 0.13),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
    )
    axis.set(
        xlabel="Incoming layer slope, $m_0$",
        ylabel=(
            "Right-inlet normalized source height\n"
            "$\\sigma_0=(z_0-b_R)/h_R$ (0 = bed; 1 = surface)"
        ),
        title="Steady material-line ordering",
    )
    return image


def material_curve_snapshots(
    result,
    target_sigma=0.05,
    initial_length=30.0,
    source_count=121,
):
    """Advect a finite horizontal line whose right endpoint is a traced source.

    The steady diagnostic follows a point at the right inlet together with the
    deformation of an incoming tangent ``(1, m0)``.  To show the corresponding
    fold as a finite curve, initialize

        X0(xi) = (x_source + xi, z_source),  xi <= 0,

    and advect every vertex through the same steady velocity field.  The right
    endpoint therefore has ``sigma0 ~= target_sigma`` and ``m0 = 0``; its
    endpoint derivative is the diagnostic ``J = F00`` in the short-segment
    limit.  A short segment is used so that every vertex represents the same
    near-bed inlet neighborhood rather than mixing a wide range of depths.
    """
    sampler = result["sampler"]
    trace = result["trace"]
    source_sigma = np.asarray(trace["source sigma"], dtype=float)
    source_index = int(np.argmin(np.abs(source_sigma - target_sigma)))
    selected_sigma = float(source_sigma[source_index])
    source_x = (
        sampler["domain right"] - 2.0 * sampler["horizontal margin"]
    )
    source_z = float(
        sampler["bed"](source_x)
        + selected_sigma * sampler["thickness"](source_x)
    )

    initial_x = np.linspace(source_x - initial_length, source_x, source_count)
    initial_positions = np.column_stack(
        (initial_x, np.full(source_count, source_z))
    )
    initial_sigma = core._terrain_sigma(initial_positions, sampler)
    if (
        np.any(initial_sigma <= sampler["sigma lower"])
        or np.any(initial_sigma >= sampler["sigma upper"])
    ):
        raise RuntimeError("initial horizontal material line leaves the ice")

    history_times = np.asarray(trace["history times"], dtype=float)
    endpoint_j = np.asarray(
        trace["history states"][:, source_index, 2], dtype=float
    )
    finite_history = np.isfinite(history_times) & np.isfinite(endpoint_j)
    history_times = history_times[finite_history]
    endpoint_j = endpoint_j[finite_history]
    crossing_indices = np.flatnonzero(endpoint_j <= 0.0)
    if not len(crossing_indices):
        raise RuntimeError("selected horizontal source never reaches J=0")
    crossing_index = int(crossing_indices[0])
    if crossing_index == 0:
        zero_age = float(history_times[0])
    else:
        zero_age = float(
            np.interp(
                0.0,
                endpoint_j[crossing_index - 1 : crossing_index + 1][::-1],
                history_times[crossing_index - 1 : crossing_index + 1][::-1],
            )
        )
    horizontal = result["source frame"]
    horizontal = horizontal[
        np.isclose(horizontal["incoming slope m0"], 0.0)
        & np.isclose(horizontal["source sigma"], selected_sigma)
    ]
    if len(horizontal) != 1:
        raise RuntimeError("could not identify selected horizontal source")
    selected_row = horizontal.iloc[0]
    minimum_age = float(selected_row["age at minimum J (yr)"])
    rheology = result["configuration"].rheology
    if rheology == "linear":
        pre_zero_age = 122_000.0
        snapshot_ages = np.unique(
            np.array(
                (
                    0.0,
                    20_000.0,
                    40_000.0,
                    60_000.0,
                    80_000.0,
                    100_000.0,
                    115_000.0,
                    pre_zero_age,
                    126_000.0,
                    129_000.0,
                    zero_age,
                    133_000.0,
                    minimum_age,
                ),
                dtype=float,
            )
        )
    else:
        pre_zero_age = float(np.round(0.935 * zero_age, -3))
        early_limit = 20_000.0 * np.floor(0.82 * zero_age / 20_000.0)
        early_ages = np.arange(0.0, early_limit + 1.0, 20_000.0)
        snapshot_ages = np.unique(
            np.r_[
                early_ages,
                np.round(0.88 * zero_age, -3),
                pre_zero_age,
                np.round(0.965 * zero_age, -3),
                np.round(0.988 * zero_age, -3),
                zero_age,
                np.round(0.5 * (zero_age + minimum_age), -3),
                minimum_age,
            ].astype(float)
        )

    positions = initial_positions.copy()
    active = np.ones(source_count, dtype=bool)
    snapshots = []
    age = 0.0
    requested_step = float(trace["time step"])

    def record_snapshot(snapshot_age):
        valid = active.copy()
        segment_j = np.full(source_count - 1, np.nan)
        valid_pairs = valid[:-1] & valid[1:]
        segment_j[valid_pairs] = (
            np.diff(positions[:, 0])[valid_pairs]
            / np.diff(initial_x)[valid_pairs]
        )
        snapshots.append(
            {
                "age": float(snapshot_age),
                "positions": positions.copy(),
                "valid": valid,
                "segment J": segment_j,
                "endpoint J": float(
                    np.interp(snapshot_age, history_times, endpoint_j)
                ),
            }
        )

    for target_age in snapshot_ages:
        while age < target_age - 1.0e-10:
            indices = np.flatnonzero(active)
            if not len(indices):
                raise RuntimeError("material curve exited before final snapshot")
            local_step = min(requested_step, target_age - age)
            current = positions[indices]
            k1 = sampler["sample"](current)
            k2 = sampler["sample"](current + 0.5 * local_step * k1)
            k3 = sampler["sample"](current + 0.5 * local_step * k2)
            k4 = sampler["sample"](current + local_step * k3)
            candidate = current + local_step * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
            positions[indices] = candidate
            local_sigma = core._terrain_sigma(candidate, sampler)
            left_boundary = (
                sampler["domain left"] + sampler["horizontal margin"]
            )
            right_boundary = (
                sampler["domain right"] - sampler["horizontal margin"]
            )
            leaving = (
                ~np.all(np.isfinite(candidate), axis=1)
                | (candidate[:, 0] <= left_boundary)
                | (candidate[:, 0] >= right_boundary)
                | (local_sigma <= sampler["sigma lower"])
                | (local_sigma >= sampler["sigma upper"])
            )
            active[indices[leaving]] = False
            age += local_step
        record_snapshot(target_age)

    endpoint_finite_difference = float(snapshots[-1]["segment J"][-1])
    endpoint_finite_difference_error = abs(
        endpoint_finite_difference - snapshots[-1]["endpoint J"]
    )
    if endpoint_finite_difference_error > 0.01:
        raise RuntimeError(
            "finite-curve endpoint does not reproduce tangent diagnostic"
        )
    return {
        "initial x": initial_x,
        "source index": source_index,
        "source sigma": selected_sigma,
        "source x": source_x,
        "source z": source_z,
        "initial positions": initial_positions,
        "initial sigma range": (
            float(np.min(initial_sigma)),
            float(np.max(initial_sigma)),
        ),
        "zero age": zero_age,
        "pre-zero closeup age": pre_zero_age,
        "minimum age": minimum_age,
        "minimum J": float(selected_row["minimum J"]),
        "endpoint finite-difference J": endpoint_finite_difference,
        "endpoint finite-difference error": endpoint_finite_difference_error,
        "snapshots": snapshots,
    }


def _consecutive_groups(indices):
    """Split integer indices so plotting never bridges exited vertices."""
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return ()
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def _draw_material_curve_evolution(axis, result, curve, cmap, age_norm):
    """Show the full advection history of the near-bed material segment."""
    sampler = result["sampler"]
    plotted_positions = np.vstack(
        [
            snapshot["positions"][snapshot["valid"]]
            for snapshot in curve["snapshots"]
        ]
    )
    x_padding = max(2.0, 0.08 * np.ptp(plotted_positions[:, 0]))
    x_limits_m = (
        float(np.min(plotted_positions[:, 0]) - x_padding),
        float(np.max(plotted_positions[:, 0]) + x_padding),
    )
    x = np.linspace(*x_limits_m, 801)
    floor = float(
        min(
            np.min(sampler["bed"](x)) - 8.0,
            np.min(plotted_positions[:, 1]) - 15.0,
        )
    )
    axis.fill_between(
        x / 1000.0, floor, sampler["bed"](x), color="0.72", zorder=0
    )
    axis.plot(x / 1000.0, sampler["bed"](x), color="0.15", linewidth=1.0)
    axis.plot(
        x / 1000.0, sampler["surface"](x), color="0.15", linewidth=1.0
    )
    colors = [
        cmap(age_norm(snapshot["age"] / 1000.0))
        for snapshot in curve["snapshots"]
    ]
    for snapshot_index, (snapshot, color) in enumerate(
        zip(curve["snapshots"], colors)
    ):
        valid_indices = np.flatnonzero(snapshot["valid"])
        for group in _consecutive_groups(valid_indices):
            if len(group) < 2:
                continue
            axis.plot(
                snapshot["positions"][group, 0] / 1000.0,
                snapshot["positions"][group, 1],
                color=color,
                linewidth=(
                    1.8
                    if np.isclose(snapshot["age"], curve["minimum age"])
                    else 1.25
                ),
                zorder=3 + snapshot_index,
            )
        if len(valid_indices):
            endpoint = valid_indices[-1]
            axis.plot(
                snapshot["positions"][endpoint, 0] / 1000.0,
                snapshot["positions"][endpoint, 1],
                marker="o",
                markersize=3.1,
                color=color,
                zorder=8,
            )
    axis.text(
        0.54,
        0.64,
        (
            f"30 m horizontal inlet segment: $\\sigma_0="
            f"{curve['source sigma']:.3f}$, $m_0=0$\n"
            f"$J_{{\\mathrm{{end}}}}=0$ at "
            f"{curve['zero age'] / 1000.0:.1f} kyr\n"
            f"$J_{{\\mathrm{{end,min}}}}={curve['minimum J']:.3f}$ at "
            f"{curve['minimum age'] / 1000.0:.1f} kyr"
        ),
        transform=axis.transAxes,
        fontsize=7.2,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    axis.set(
        xlim=np.asarray(x_limits_m) / 1000.0,
        ylim=(floor, float(np.max(sampler["surface"](x)) + 8.0)),
        xlabel="Distance (km)",
        ylabel="Elevation (m)",
        title="Near-bed material-line evolution",
    )

    return colors


def _draw_fold_closeup(axis, result, curve, colors):
    """Resolve the narrow re-entrant hook once the endpoint reaches J < 0."""
    sampler = result["sampler"]
    closeup_ages = (
        curve["pre-zero closeup age"],
        curve["zero age"],
        curve["minimum age"],
    )
    late = [
        (snapshot, color)
        for snapshot, color in zip(curve["snapshots"], colors)
        if any(
            np.isclose(snapshot["age"], selected_age)
            for selected_age in closeup_ages
        )
    ]
    if len(late) != len(closeup_ages):
        raise RuntimeError("fold close-up snapshots are missing")
    fold_positions = np.vstack(
        [snapshot["positions"][snapshot["valid"]] for snapshot, _ in late]
    )
    x_padding = max(0.15, 0.08 * np.ptp(fold_positions[:, 0]))
    y_padding = max(4.0, 0.05 * np.ptp(fold_positions[:, 1]))
    x_limits = (
        float(np.min(fold_positions[:, 0]) - x_padding),
        float(np.max(fold_positions[:, 0]) + x_padding),
    )
    y_limits = (
        float(np.min(fold_positions[:, 1]) - y_padding),
        float(np.max(fold_positions[:, 1]) + y_padding),
    )
    x = np.linspace(*x_limits, 401)
    surface = sampler["surface"](x)
    if np.min(surface) <= y_limits[1]:
        axis.plot(x, surface, color="0.15", linewidth=1.0, label="surface")

    negative_snapshots = [
        snapshot
        for snapshot, _ in late
        if np.any(snapshot["segment J"] < 0.0)
    ]
    negative_colors = (
        "#0072B2",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#332288",
    )
    negative_markers = ("o", "s", "^", "D", "P", "X")
    negative_style = {
        snapshot["age"]: (
            negative_colors[index % len(negative_colors)],
            negative_markers[index % len(negative_markers)],
        )
        for index, snapshot in enumerate(negative_snapshots)
    }
    for snapshot, color in late:
        valid_indices = np.flatnonzero(snapshot["valid"])
        for group in _consecutive_groups(valid_indices):
            if len(group) < 2:
                continue
            axis.plot(
                snapshot["positions"][group, 0],
                snapshot["positions"][group, 1],
                color=color,
                linewidth=1.8,
                zorder=2,
            )
        negative = np.flatnonzero(snapshot["segment J"] < 0.0)
        negative = negative[:: max(1, len(negative) // 10)]
        if len(negative):
            marker_color, marker = negative_style[snapshot["age"]]
            midpoints = 0.5 * (
                snapshot["positions"][negative]
                + snapshot["positions"][negative + 1]
            )
            axis.scatter(
                midpoints[:, 0],
                midpoints[:, 1],
                s=24,
                marker=marker,
                facecolors="white",
                edgecolors=marker_color,
                linewidths=1.05,
                label=f"{snapshot['age'] / 1000.0:.1f} kyr",
                zorder=4,
            )
    axis.set(
        xlim=x_limits,
        ylim=y_limits,
        xlabel="Distance, $x$ (m; horizontal scale exaggerated)",
        ylabel="Elevation (m)",
        title="Fold close-up",
    )
    if negative_snapshots:
        axis.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.88,
            fontsize=6.5,
            title=r"$J<0$ markers (elapsed time)",
            title_fontsize=6.7,
            loc="upper right",
            borderpad=0.35,
            handletextpad=0.35,
            labelspacing=0.25,
        )
    return x_limits, y_limits


def _convergence_cases(case_frame):
    """Return truncation, production, and exact-margin mesh sequences."""
    ordered = case_frame.sort_values(["left thickness (m)", "resolution (m)"])
    traction_free = ordered[ordered["surface mode"] == "traction_free"]
    exact = traction_free[traction_free["left thickness (m)"] == 0.0]
    truncated = traction_free[traction_free["left thickness (m)"] > 0.0]
    production = exact[exact["case"] == "point_margin"]
    if len(production) != 1:
        raise RuntimeError("production point-margin case is missing")
    return truncated, production.iloc[0], exact


def make_convergence_figure(case_frame, rheology: str = "linear"):
    """Write the point-margin numerical-convergence supplement figure."""
    truncated, production, exact = _convergence_cases(case_frame)
    figure, axes = plt.subplots(
        1, 2, figsize=(9.0, 3.9), constrained_layout=True
    )
    axes[0].plot(
        truncated["left thickness (m)"],
        truncated["minimum J, m0=0"],
        marker="o",
        color="0.20",
        label="traction-free truncations",
    )
    axes[0].scatter(
        0.0,
        production["minimum J, m0=0"],
        marker="s",
        s=42,
        facecolors="none",
        edgecolors="#2166ac",
        label="true point margin",
        zorder=4,
    )
    if rheology == "linear":
        trunc_ylim = (-0.205, -0.115)
    else:
        trunc_vals = np.r_[
            truncated["minimum J, m0=0"].to_numpy(float),
            float(production["minimum J, m0=0"]),
        ]
        span = max(0.06, float(np.ptp(trunc_vals)))
        trunc_ylim = (
            float(np.min(trunc_vals) - 0.18 * span),
            min(-0.01, float(np.max(trunc_vals) + 0.18 * span)),
        )
    axes[0].set(
        ylim=trunc_ylim,
        xlabel="Thickness of artificial left truncation (m)",
        ylabel="Minimum $J$ for horizontal incoming layers",
        title="Truncation limit",
    )
    axes[0].legend(frameon=False, fontsize=8.0, loc="upper left")

    exact_by_resolution = exact.sort_values("resolution (m)")
    axes[1].plot(
        exact_by_resolution["resolution (m)"],
        exact_by_resolution["minimum J, m0=0"],
        marker="s",
        color="#2166ac",
        linewidth=1.2,
    )
    axes[1].scatter(
        production["resolution (m)"],
        production["minimum J, m0=0"],
        s=48,
        facecolors="white",
        edgecolors="#2166ac",
        linewidths=1.2,
        zorder=4,
    )
    axes[1].annotate(
        "production mesh",
        xy=(
            production["resolution (m)"],
            production["minimum J, m0=0"],
        ),
        xytext=(4, 9),
        textcoords="offset points",
        fontsize=8.0,
    )
    axes[1].set(
        xlabel="Nominal mesh resolution (m)",
        ylabel="Minimum $J$ for horizontal incoming layers",
        title="Exact point-margin mesh sequence",
    )
    for label, axis in zip("AB", axes):
        axis.text(
            -0.12,
            1.03,
            label,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    suffix = _output_suffix(rheology)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png = (
        FIGURE_DIRECTORY
        / f"allan_hills_nunatak_margin{suffix}_convergence.png"
    )
    pdf = (
        FIGURE_DIRECTORY
        / f"allan_hills_nunatak_margin{suffix}_convergence.pdf"
    )
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def make_figure(results, rheology: str = "linear"):
    primary = next(
        result for result in results if result["configuration"].key == "point_margin"
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(13.4, 8.4), constrained_layout=True
    )
    axes = axes.ravel()
    _draw_geometry(axes[0], rheology=rheology)
    u_contour = _draw_flow(axes[1], primary, 0, "Horizontal flow and paths")
    w_contour = _draw_flow(axes[2], primary, 1, "Vertical velocity")
    j_image = _draw_j_map(axes[3], primary)
    curve = material_curve_snapshots(primary)
    age_cmap = plt.get_cmap("viridis")
    age_norm = Normalize(0.0, curve["minimum age"] / 1000.0)
    colors = _draw_material_curve_evolution(
        axes[4], primary, curve, age_cmap, age_norm
    )
    zoom_x_limits, zoom_y_limits = _draw_fold_closeup(
        axes[5], primary, curve, colors
    )
    exact_zoom_width_km = (
        zoom_x_limits[1] - zoom_x_limits[0]
    ) / 1000.0
    # The exact 3-4 m span is sub-pixel on panel E's kilometre-scale axis.
    # Center a minimum-width locator on it so the requested box is visible.
    locator_width_km = max(exact_zoom_width_km, 0.03)
    locator_left_km = (
        0.5 * (zoom_x_limits[0] + zoom_x_limits[1]) / 1000.0
        - 0.5 * locator_width_km
    )
    zoom_box = Rectangle(
        (locator_left_km, zoom_y_limits[0]),
        locator_width_km,
        zoom_y_limits[1] - zoom_y_limits[0],
        facecolor="none",
        edgecolor="black",
        linewidth=0.7,
        linestyle="--",
        zorder=10,
    )
    axes[4].add_patch(zoom_box)
    axes[4].text(
        locator_left_km + locator_width_km + 0.012,
        zoom_y_limits[1] - 12.0,
        "Panel F.",
        color="black",
        fontsize=7.2,
        ha="left",
        va="top",
        zorder=11,
    )

    for label, contour, axis in (
        ("$u$ (m yr$^{-1}$)", u_contour, axes[1]),
        ("$w$ (m yr$^{-1}$)", w_contour, axes[2]),
        ("Minimum $J$", j_image, axes[3]),
    ):
        figure.colorbar(
            contour,
            ax=axis,
            orientation="horizontal",
            pad=0.14 if axis is not axes[3] else 0.11,
            label=label,
            ticks=np.linspace(contour.norm.vmin, contour.norm.vmax, 5),
            format="%.3g",
        )
    age_mappable = plt.cm.ScalarMappable(norm=age_norm, cmap=age_cmap)
    age_mappable.set_array([])
    max_kyr = curve["minimum age"] / 1000.0
    if rheology == "linear":
        age_ticks = (0.0, 30.0, 60.0, 90.0, 120.0, max_kyr)
    else:
        base_ticks = np.arange(0.0, max_kyr - 14.0, 30.0)
        age_ticks = tuple(np.r_[base_ticks, max_kyr])
    figure.colorbar(
        age_mappable,
        ax=[axes[4], axes[5]],
        orientation="horizontal",
        pad=0.12,
        label="Elapsed time (kyr)",
        ticks=age_ticks,
        format="%.0f",
    )
    for label, axis in zip("ABCDEF", axes):
        axis.text(
            -0.10,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    suffix = _output_suffix(rheology)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIRECTORY / f"allan_hills_nunatak_margin{suffix}.png"
    pdf = FIGURE_DIRECTORY / f"allan_hills_nunatak_margin{suffix}.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf, curve


def write_report(case_frame, rheology: str = "linear", curve=None):
    suffix = _output_suffix(rheology)
    primary = case_frame.loc[case_frame["case"] == "point_margin"].iloc[0]
    refined = case_frame.loc[
        case_frame["case"] == "point_margin_refined"
    ].iloc[0]
    uniform = case_frame.loc[
        case_frame["case"] == "point_margin_uniform_ablation"
    ].iloc[0]
    source_frame = pd.read_csv(
        DATA_DIRECTORY / f"allan_hills_nunatak_margin{suffix}_sources.csv.gz"
    )
    primary_horizontal = source_frame[
        (source_frame["case"] == "point_margin")
        & source_frame["is primary horizontal"]
    ]
    negative_horizontal = primary_horizontal[
        primary_horizontal["minimum J"] < 0.0
    ]
    minimum_above_five_percent = primary_horizontal.loc[
        primary_horizontal["source sigma"] >= 0.05, "minimum J"
    ].min()
    validation_path = (
        DATA_DIRECTORY / f"allan_hills_nunatak{suffix}_validation.csv"
    )
    if validation_path.exists():
        validation_frame = pd.read_csv(validation_path)
        validation_table = core._markdown_table(
            validation_frame,
            (
                "variant",
                "sample nx",
                "sample nz",
                "source count",
                "source plane x (m)",
                "actual time step (yr)",
                "minimum J, m0=0",
                "minimum J for source sigma >= 0.05",
                "negative horizontal sources",
                "maximum abs detF-1",
                "age-capped paths",
            ),
            {
                "variant": str,
                "sample nx": lambda value: f"{int(value)}",
                "sample nz": lambda value: f"{int(value)}",
                "source count": lambda value: f"{int(value)}",
                "source plane x (m)": lambda value: f"{value:.1f}",
                "actual time step (yr)": lambda value: f"{value:.3f}",
                "minimum J, m0=0": lambda value: f"{value:.5f}",
                "minimum J for source sigma >= 0.05": (
                    lambda value: f"{value:.5f}"
                ),
                "negative horizontal sources": lambda value: f"{int(value)}",
                "maximum abs detF-1": lambda value: f"{value:.2e}",
                "age-capped paths": lambda value: f"{int(value)}",
            },
        )
    else:
        validation_table = "Validation file not yet generated."
    columns = [
        "case",
        "left thickness (m)",
        "mesh cells",
        "resolution (m)",
        "surface mode",
        "minimum J, m0=0",
        "left outward flux (m2/yr)",
        "surface outward flux (m2/yr)",
        "age-capped paths",
        "maximum abs detF-1",
    ]
    table = core._markdown_table(
        case_frame,
        columns,
        {
            "case": str,
            "left thickness (m)": lambda value: f"{value:.3g}",
            "mesh cells": lambda value: f"{int(value)}",
            "resolution (m)": lambda value: f"{value:.3g}",
            "surface mode": str,
            "minimum J, m0=0": lambda value: f"{value:.5f}",
            "left outward flux (m2/yr)": lambda value: f"{value:.3e}",
            "surface outward flux (m2/yr)": lambda value: f"{value:.5f}",
            "age-capped paths": lambda value: f"{int(value)}",
            "maximum abs detF-1": lambda value: f"{value:.2e}",
        },
    )
    rheology_note = (
        f"linear viscosity $\\eta={core.VISCOSITY:.2e}$ Pa yr"
        if rheology == "linear"
        else (
            f"isotropic Glen rheology ($n={GLEN_EXPONENT:g}$, "
            f"$A_{{\\mathrm{{G}}}}={GLEN_FLUIDITY_PA_S:.2e}$ Pa$^{{-3}}$ s$^{{-1}}$ "
            f"$={GLEN_FLUIDITY:.3e}$ Pa$^{{-3}}$ yr$^{{-1}}$ at $-25^\\circ$C, "
            f"regularization $D_0={GLEN_STRAIN_RATE_REG:.1e}$ yr$^{{-1}}$)"
        )
    )
    curve_note = ""
    if curve is not None:
        curve_note = (
            f"\n\nFor the plotted 30 m horizontal segment at $\\sigma_0={curve['source sigma']:.3f}$, "
            f"the endpoint reaches $J=0$ at {curve['zero age'] / 1000.0:.1f} kyr and "
            f"$J={curve['minimum J']:.3f}$ at {curve['minimum age'] / 1000.0:.1f} kyr."
        )
    displaced_note = ""
    if validation_path.exists():
        displaced_rows = validation_frame[
            validation_frame["variant"] == "source-plane-20m-inside"
        ]
        if len(displaced_rows) == 1:
            displaced_note = (
                f" The displaced source plane gives "
                f"$J_{{min}}={displaced_rows.iloc[0]['minimum J, m0=0']:.5f}$, "
                f"ruling out the lower-right boundary corner as the sole cause."
            )
    text = f"""# Allan Hills nunatak-margin experiment ({rheology})

## Setup

The ice domain ends at a true grounded point margin: bed and free surface meet at $x=0$, and exposed rock lies to the left. There is no left boundary facet, no left boundary condition, and no prescribed left influx. Ice obeys {rheology_note}. The bed has no penetration plus the same linear sliding law as the other experiments. The primary surface is fully traction free; a control instead imposes the normal velocity corresponding to uniform steady ablation while retaining zero tangential traction. At the right inlet, the horizontal velocity is the reduced SIA-shaped profile and $w=0$.

The thickness is $H(x)=H_R[1-\\exp(-x/\\ell_H)]/[1-\\exp(-L/\\ell_H)]$, with $H_R={RIGHT_THICKNESS:.0f}$ m, $L={LENGTH / 1000:.1f}$ km, and $\\ell_H={THICKENING_LENGTH:.0f}$ m. Thus $H(0)=0$ with a finite opening angle. The immediate contact-line region is excluded from the material-line minimum because terrain coordinates are singular at the apex: the primary uses $H>{primary['diagnostic thickness cutoff (m)']:.0f}$ m and the uniform-ablation control uses the more permissive $H>{uniform['diagnostic thickness cutoff (m)']:.0f}$ m.

For the fixed geometry, the effective surface mass balance $a_{{\\mathrm{{eff}}}}=u_s s_x-w_s$ is diagnosed. In strict two dimensions it absorbs the right-side influx; in the real Allan Hills section it may also represent omitted lateral flux divergence.

## Results

{table}

With the fully traction-free surface, the production point-margin value is $J_{{min}}={primary['minimum J, m0=0']:.5f}$ for initially horizontal layers; the refined value is {refined['minimum J, m0=0']:.5f}. {len(negative_horizontal)} of {len(primary_horizontal)} sampled right-inlet source heights overturn. The sign does not depend solely on the lowest source: after excluding all sources below $\\sigma_0=0.05$, the minimum is still {minimum_above_five_percent:.5f}. The global horizontal-layer minimum occurs at $x={primary['x at minimum J, m0=0 (m)']:.0f}$ m, where $H={float(ice_thickness(primary['x at minimum J, m0=0 (m)'])):.0f}$ m, well outside the contact-line exclusion. The exact point margin has zero left flux by construction. However, that solution diagnoses a spatially alternating surface balance that supplies a small rightward flow near the nunatak. When a uniform steady ablation rate is imposed instead, the minimum is {uniform['minimum J, m0=0']:.5f}, so the layers remain ordered. All control paths exit and the minimum stays positive. The margin geometry alone is therefore not sufficient: the surface/lateral mass-flux closure controls whether the convergent counterflow exists.{curve_note}

The production right-boundary RMS vertical velocity is {primary['rms right-boundary w (m/yr)']:.2e} m yr$^{{-1}}$. The uniform-ablation control prescribes $a_s={uniform['prescribed uniform surface mass balance (m/yr)']:.3e}$ m yr$^{{-1}}$ and has RMS surface normal-speed error {uniform['rms surface normal-speed error (m/yr)']:.2e} m yr$^{{-1}}$.

## Diagnostic validation

{validation_table}

The overturning sign is unchanged by halving the trajectory step, increasing terrain and source resolution, enlarging the terrain buffer, or moving the source plane from 1 m to 20 m inside the inlet.{displaced_note}

## Interpretation limits

The margin corner can carry a local stress singularity, so only minima outside the stated thickness cutoff are interpreted. Positive-thickness truncations and the exact-margin mesh sequence test this sensitivity; their shrinking artificial faces are traction free and can carry small fluxes, so they are numerical checks rather than physical boundary models. This remains a two-dimensional fixed-geometry calculation: any implausible diagnosed surface balance is evidence that lateral convergence around the nunatak matters dynamically.
"""
    report_path = (
        ROOT / f"notebooks/allan_hills_nunatak_margin{suffix}_report.md"
    )
    report_path.write_text(text)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rheology",
        choices=("linear", "glen"),
        default="linear",
        help="Constitutive law: linear viscous (n=1) or isotropic Glen (n=3).",
    )
    parser.add_argument("--resolution", type=float, default=DEFAULT_CHARACTERISTIC_RESOLUTION)
    parser.add_argument("--tip-resolution", type=float, default=DEFAULT_TIP_RESOLUTION)
    parser.add_argument(
        "--diagnostic-thickness-cutoff",
        type=float,
        default=DEFAULT_DIAGNOSTIC_THICKNESS_CUTOFF,
    )
    parser.add_argument("--sample-nx", type=int, default=501)
    parser.add_argument("--sample-nz", type=int, default=161)
    parser.add_argument("--horizontal-margin", type=float, default=0.5)
    parser.add_argument("--sigma-lower", type=float, default=0.0005)
    parser.add_argument("--sigma-upper", type=float, default=0.99)
    parser.add_argument("--sources", type=int, default=193)
    parser.add_argument("--source-sigma-min", type=float, default=0.01)
    parser.add_argument("--source-sigma-max", type=float, default=0.95)
    parser.add_argument("--slope-limit", type=float, default=0.10)
    parser.add_argument("--slope-count", type=int, default=41)
    parser.add_argument("--maximum-age", type=float, default=3_000_000.0)
    parser.add_argument("--time-step", type=float, default=20.0)
    parser.add_argument("--single", action="store_true",
                        help="run only the production point-margin case")
    return parser.parse_args()


def main():
    require_environment()
    args = parse_args()
    configurations = default_configurations(args)
    if args.single:
        configurations = tuple(
            configuration
            for configuration in configurations
            if configuration.key == "point_margin"
        )
    results = [
        run_configuration(configuration, args)
        for configuration in configurations
    ]
    if args.single:
        print(
            pd.DataFrame(
                [result["diagnostics"] for result in results]
            ).to_string(index=False),
            flush=True,
        )
        return
    case_frame, _, _ = write_outputs(results, rheology=args.rheology)
    _, _, curve = make_figure(results, rheology=args.rheology)
    make_convergence_figure(case_frame, rheology=args.rheology)
    write_report(case_frame, rheology=args.rheology, curve=curve)
    print(case_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
