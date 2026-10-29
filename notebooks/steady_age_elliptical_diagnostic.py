"""Diagnose isochrone overturning from stationary Eulerian transport.

For a steady incompressible velocity ``v = (u, w)``, material age satisfies

    v . grad(tau) = 1.

On a surface-sourced streamline, let ``a0`` be the positive accumulation rate
at deposition.  The horizontal order Jacobian of a common-age layer is then

    J = partial x / partial x0 |_tau = -a0 * partial tau / partial z.

Consequently ``partial tau / partial z > 0`` is an exact overturn criterion.
The same Jacobian is obtained without differentiating age by transporting the
equal-age tangent ``T``:

    v . grad(T) = grad(v) T,       T_surface = (1, 0),       J = T_x.

This script solves both stationary equations with upwind DG; it does not
advance particles or an age field in time.  A provenance equation separates
surface-sourced ice from ice entering through the hydrostatic lateral boundary.

The calculation retains the full-Stokes flow and linear sliding law used by the
elliptical-geometry study.  Two trajectory results are recomputed only as an
independent validation of the stationary diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import firedrake as fd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from realistic_geometry_elliptical_family import (
    bed_height,
    fixed_interior_end_age_column,
    fixed_interior_minimum_column,
    make_sampler,
    side_locations,
    solve_stokes,
    trace_surface_map,
)
from realistic_geometry_elliptical_parameter_map import (
    geometry_from_thicknesses,
)


DIRECT_SOLVER = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

CASES = (
    {
        "case": "overturned",
        "boundary thickness (m)": 70.0,
        "center thickness (m)": 150.0,
    },
    {
        "case": "ordered",
        "boundary thickness (m)": 70.0,
        "center thickness (m)": 270.0,
    },
)
PROVENANCE_THRESHOLDS = (0.90, 0.95, 0.99, 0.9999)

plt.rcParams.update(
    {
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def solve_upwind_transport(
    velocity,
    degree,
    volume_source,
    surface_inflow_value,
):
    """Solve ``v.grad(q)=source`` with weak upwind inflow data.

    The given surface value is applied only where the surface is inflow.
    Unspecified inflow boundaries receive zero.  In particular, age is zero at
    every inflow boundary, while provenance is one only at surface inflow.
    The latter field identifies the part of the age solution that has physical
    surface-deposition data.
    """
    mesh = velocity.function_space().mesh()
    space = fd.FunctionSpace(mesh, "DG", degree)
    trial = fd.TrialFunction(space)
    test = fd.TestFunction(space)
    normal = fd.FacetNormal(mesh)
    normal_velocity = fd.dot(velocity, normal)
    outgoing = 0.5 * (normal_velocity + abs(normal_velocity))
    incoming = 0.5 * (normal_velocity - abs(normal_velocity))

    bilinear = (
        -trial * fd.div(test * velocity) * fd.dx
        + fd.inner(
            outgoing("+") * trial("+")
            - outgoing("-") * trial("-"),
            fd.jump(test),
        )
        * fd.dS
        + outgoing * trial * test * fd.ds
    )
    linear = (
        volume_source * test * fd.dx
        - surface_inflow_value * incoming * test * fd.ds(2)
    )
    solution = fd.Function(space)
    fd.solve(
        bilinear == linear,
        solution,
        solver_parameters=DIRECT_SOLVER,
    )
    return solution


def solve_stationary_tangent(velocity):
    """Solve ``v.grad(T)=grad(v) T`` with the deposition tangent at inflow.

    The modeled surface is horizontal, so the equal-age material tangent is
    ``T=(1, 0)`` at deposition.  Other inflow boundaries receive zero; the
    provenance field below excludes their streamtubes from the diagnosis.
    """
    mesh = velocity.function_space().mesh()
    space = fd.VectorFunctionSpace(mesh, "DG", 1)
    trial = fd.TrialFunction(space)
    test = fd.TestFunction(space)
    normal = fd.FacetNormal(mesh)
    normal_velocity = fd.dot(velocity, normal)
    outgoing = 0.5 * (normal_velocity + abs(normal_velocity))
    incoming = 0.5 * (normal_velocity - abs(normal_velocity))

    # This is the non-conservative transport equation integrated by parts.
    # Retaining div(test (x) velocity) also retains the small pointwise
    # divergence of the discrete Taylor--Hood velocity.
    bilinear = (
        -fd.inner(trial, fd.div(fd.outer(test, velocity))) * fd.dx
        + fd.inner(
            outgoing("+") * trial("+")
            - outgoing("-") * trial("-"),
            fd.jump(test),
        )
        * fd.dS
        + fd.inner(outgoing * trial, test) * fd.ds
        - fd.inner(fd.dot(fd.grad(velocity), trial), test) * fd.dx
    )
    surface_tangent = fd.as_vector((1.0, 0.0))
    linear = -incoming * fd.inner(surface_tangent, test) * fd.ds(2)
    tangent = fd.Function(space)
    fd.solve(
        bilinear == linear,
        tangent,
        solver_parameters=DIRECT_SOLVER,
    )
    return tangent


def solve_stationary_age(velocity):
    """Return age, provenance, source labels, and both exact forms of J."""
    mesh = velocity.function_space().mesh()
    x, _ = fd.SpatialCoordinate(mesh)
    normal = fd.FacetNormal(mesh)
    normal_velocity = fd.dot(velocity, normal)

    age = solve_upwind_transport(
        velocity,
        degree=1,
        volume_source=fd.Constant(1.0),
        surface_inflow_value=fd.Constant(0.0),
    )
    provenance = solve_upwind_transport(
        velocity,
        degree=0,
        volume_source=fd.Constant(0.0),
        surface_inflow_value=fd.Constant(1.0),
    )
    source_x = solve_upwind_transport(
        velocity,
        degree=1,
        volume_source=fd.Constant(0.0),
        surface_inflow_value=x,
    )
    deposition_accumulation = solve_upwind_transport(
        velocity,
        degree=1,
        volume_source=fd.Constant(0.0),
        surface_inflow_value=-normal_velocity,
    )
    tangent = solve_stationary_tangent(velocity)

    cell_space = fd.FunctionSpace(mesh, "DG", 0)
    vector_cell_space = fd.VectorFunctionSpace(mesh, "DG", 0)
    cell_coordinates = fd.Function(vector_cell_space).interpolate(
        fd.SpatialCoordinate(mesh)
    )
    cell_velocity = fd.project(velocity, vector_cell_space)
    age_cell = fd.project(age, cell_space)
    source_x_cell = fd.project(source_x, cell_space)
    accumulation_cell = fd.project(deposition_accumulation, cell_space)
    age_z_cell = fd.project(age.dx(1), cell_space)
    jacobian_cell = fd.project(
        -deposition_accumulation * age.dx(1), cell_space
    )
    tangent_cell = fd.project(tangent, vector_cell_space)
    tangent_jacobian_cell = fd.project(tangent[0], cell_space)
    return {
        "age": age,
        "provenance": provenance,
        "source x": source_x,
        "deposition accumulation": deposition_accumulation,
        "tangent": tangent,
        "cell coordinates": cell_coordinates,
        "cell velocity": cell_velocity,
        "cell age": age_cell,
        "cell provenance": provenance,
        "cell source x": source_x_cell,
        "cell accumulation": accumulation_cell,
        "cell age z derivative": age_z_cell,
        "cell J": jacobian_cell,
        "cell tangent": tangent_cell,
        "cell tangent J": tangent_jacobian_cell,
    }


def fixed_source_interval(geometry, sample, source_count=401):
    """Match the central 80% deposition interval of the trajectory diagnostic."""
    left_side, _ = side_locations(geometry)
    sources = np.linspace(0.96 * left_side, 0.04 * left_side, source_count)
    accumulation = -sample("w", sources, np.full_like(sources, -0.05))
    sources = sources[accumulation > 1.0e-6]
    margin = int(np.ceil(0.10 * len(sources)))
    interior = sources[margin : len(sources) - margin]
    return float(np.min(interior)), float(np.max(interior))


def summarize_stationary_fields(
    fields,
    source_interval,
    maximum_age,
    geometry,
    resolution,
    provenance_threshold=0.95,
):
    """Summarize the surface-fed fixed-source part of a stationary solution."""
    values = {
        name: np.asarray(function.dat.data_ro).reshape(-1)
        for name, function in fields.items()
        if name.startswith("cell ")
        and name not in ("cell coordinates", "cell velocity", "cell tangent")
    }
    coordinates = np.asarray(fields["cell coordinates"].dat.data_ro)
    velocity = np.asarray(fields["cell velocity"].dat.data_ro)
    source_min, source_max = source_interval
    left_side, right_side = side_locations(geometry)
    outflow_buffer = (
        (
            (coordinates[:, 0] < left_side + 2.0 * resolution)
            & (velocity[:, 0] < 0.0)
        )
        | (
            (coordinates[:, 0] > right_side - 2.0 * resolution)
            & (velocity[:, 0] > 0.0)
        )
    ) | (
        (coordinates[:, 1] > -2.0 * resolution)
        & (velocity[:, 1] > 0.0)
    )
    surface_streamtube = (
        np.isfinite(values["cell age"])
        & np.isfinite(values["cell provenance"])
        & np.isfinite(values["cell source x"])
        & np.isfinite(values["cell age z derivative"])
        & np.isfinite(values["cell J"])
        & np.isfinite(values["cell tangent J"])
        & (values["cell provenance"] >= provenance_threshold)
        & (values["cell source x"] >= source_min)
        & (values["cell source x"] <= source_max)
        & (values["cell age"] >= 0.0)
        & (values["cell age"] <= maximum_age)
        & (coordinates[:, 1] < -0.25)
    )
    valid = surface_streamtube & ~outflow_buffer
    if not np.any(valid):
        raise RuntimeError("stationary mask contains no surface-fed cells")
    raw_jacobian = values["cell J"][surface_streamtube]
    raw_tangent_jacobian = values["cell tangent J"][surface_streamtube]
    age_z = values["cell age z derivative"][valid]
    jacobian = values["cell J"][valid]
    tangent_jacobian = values["cell tangent J"][valid]
    accumulation = values["cell accumulation"][valid]
    age = values["cell age"][valid]
    negative = tangent_jacobian < 0.0
    resolved_negative = tangent_jacobian < -0.05
    identity_error = tangent_jacobian - jacobian
    sign_test = (np.abs(tangent_jacobian) > 0.1) & (
        np.abs(jacobian) > 0.1
    )
    return {
        "stationary valid cell count": int(np.count_nonzero(valid)),
        "stationary raw age-gradient minimum J": float(np.min(raw_jacobian)),
        "stationary raw age-gradient J 1st percentile": float(
            np.quantile(raw_jacobian, 0.01)
        ),
        "stationary raw tangent minimum J": float(
            np.min(raw_tangent_jacobian)
        ),
        "stationary raw tangent J 1st percentile": float(
            np.quantile(raw_tangent_jacobian, 0.01)
        ),
        "stationary age-gradient minimum J": float(np.min(jacobian)),
        "stationary age-gradient J 1st percentile": float(
            np.quantile(jacobian, 0.01)
        ),
        "stationary tangent minimum J": float(np.min(tangent_jacobian)),
        "stationary tangent J 1st percentile": float(
            np.quantile(tangent_jacobian, 0.01)
        ),
        "stationary tangent/age-gradient J RMSE": float(
            np.sqrt(np.mean(identity_error**2))
        ),
        "stationary tangent/age-gradient J correlation": float(
            np.corrcoef(tangent_jacobian, jacobian)[0, 1]
        ),
        "stationary tangent/age-gradient sign disagreements": int(
            np.count_nonzero(
                sign_test
                & (np.sign(tangent_jacobian) != np.sign(jacobian))
            )
        ),
        "stationary maximum tau_z (yr/m)": float(np.max(age_z)),
        "stationary tau_z 99th percentile (yr/m)": float(
            np.quantile(age_z, 0.99)
        ),
        "stationary tangent negative-J cell count": int(
            np.count_nonzero(tangent_jacobian < 0)
        ),
        "stationary tangent first negative-cell age (yr)": (
            float(np.min(age[negative])) if np.any(negative) else np.nan
        ),
        "stationary tangent first J<-0.05 cell age (yr)": (
            float(np.min(age[resolved_negative]))
            if np.any(resolved_negative)
            else np.nan
        ),
        "stationary oldest retained age (yr)": float(np.max(age)),
        "stationary minimum retained accumulation (m/yr)": float(
            np.min(accumulation)
        ),
        "stationary maximum retained accumulation (m/yr)": float(
            np.max(accumulation)
        ),
    }


def first_stationary_zero_crossing_age(
    fields,
    source_interval,
    maximum_age,
    geometry,
    resolution,
    provenance_threshold=0.90,
):
    """Estimate the earliest ``T_x=0`` contour crossing on mesh edges."""
    mesh = fields["age"].function_space().mesh()
    vertex_space = fd.FunctionSpace(mesh, "CG", 1)
    vector_vertex_space = fd.VectorFunctionSpace(mesh, "CG", 1)
    age = fd.project(fields["age"], vertex_space)
    tangent_jacobian = fd.project(fields["tangent"][0], vertex_space)
    provenance = fd.project(fields["provenance"], vertex_space)
    source_x = fd.project(fields["source x"], vertex_space)
    coordinates = fd.Function(vector_vertex_space).interpolate(
        fd.SpatialCoordinate(mesh)
    )
    cell_nodes = vertex_space.cell_node_map().values
    age_values = np.asarray(age.dat.data_ro)
    tangent_values = np.asarray(tangent_jacobian.dat.data_ro)
    provenance_values = np.asarray(provenance.dat.data_ro)
    source_values = np.asarray(source_x.dat.data_ro)
    coordinate_values = np.asarray(coordinates.dat.data_ro)
    source_min, source_max = source_interval
    left_side, right_side = side_locations(geometry)

    crossing_ages = []
    for nodes in cell_nodes:
        for local_i, local_j in ((0, 1), (1, 2), (2, 0)):
            i = nodes[local_i]
            j = nodes[local_j]
            gi = tangent_values[i]
            gj = tangent_values[j]
            if not np.isfinite(gi + gj) or gi * gj > 0.0 or gi == gj:
                continue
            fraction = -gi / (gj - gi)
            if fraction < 0.0 or fraction > 1.0:
                continue
            crossing_age = (1.0 - fraction) * age_values[i] + fraction * age_values[j]
            crossing_source = (
                (1.0 - fraction) * source_values[i]
                + fraction * source_values[j]
            )
            crossing_provenance = (
                (1.0 - fraction) * provenance_values[i]
                + fraction * provenance_values[j]
            )
            crossing_coordinate = (
                (1.0 - fraction) * coordinate_values[i]
                + fraction * coordinate_values[j]
            )
            if not (
                0.0 <= crossing_age <= maximum_age
                and source_min <= crossing_source <= source_max
                and crossing_provenance >= provenance_threshold
                and crossing_coordinate[0] >= left_side + resolution
                and crossing_coordinate[0] <= right_side - resolution
                and crossing_coordinate[1] <= -resolution
            ):
                continue
            crossing_ages.append(float(crossing_age))
    return min(crossing_ages) if crossing_ages else np.nan


def prefixed(values, prefix):
    """Prefix dictionary labels for side-by-side horizon and full-path results."""
    return {f"{prefix} {key}": value for key, value in values.items()}


def make_plot_payload(case, geometry, resolution, fields, source_interval):
    """Extract unstructured field arrays for the validation figure."""
    mesh = fields["age"].function_space().mesh()
    vertex_space = fd.FunctionSpace(mesh, "CG", 1)
    vector_vertex_space = fd.VectorFunctionSpace(mesh, "CG", 1)
    coordinates = fd.Function(vector_vertex_space).interpolate(
        fd.SpatialCoordinate(mesh)
    )
    vertex_coordinates = np.asarray(coordinates.dat.data_ro).copy()
    triangles = vertex_space.cell_node_map().values.copy()
    vertex_age = np.asarray(
        fd.project(fields["age"], vertex_space).dat.data_ro
    ).copy()

    cell_coordinates = np.asarray(fields["cell coordinates"].dat.data_ro)
    cell_velocity = np.asarray(fields["cell velocity"].dat.data_ro)
    cell_age = np.asarray(fields["cell age"].dat.data_ro)
    provenance = np.asarray(fields["cell provenance"].dat.data_ro)
    source_x = np.asarray(fields["cell source x"].dat.data_ro)
    tangent_jacobian = np.asarray(fields["cell tangent J"].dat.data_ro)
    age_jacobian = np.asarray(fields["cell J"].dat.data_ro)
    source_min, source_max = source_interval
    left_side, right_side = side_locations(geometry)
    outflow_buffer = (
        (
            (cell_coordinates[:, 0] < left_side + 2.0 * resolution)
            & (cell_velocity[:, 0] < 0.0)
        )
        | (
            (cell_coordinates[:, 0] > right_side - 2.0 * resolution)
            & (cell_velocity[:, 0] > 0.0)
        )
    ) | (
        (cell_coordinates[:, 1] > -2.0 * resolution)
        & (cell_velocity[:, 1] > 0.0)
    )
    valid = (
        np.isfinite(cell_age)
        & np.isfinite(provenance)
        & np.isfinite(source_x)
        & np.isfinite(tangent_jacobian)
        & np.isfinite(age_jacobian)
        & (provenance >= 0.95)
        & (source_x >= source_min)
        & (source_x <= source_max)
        & (cell_age >= 0.0)
        & (cell_coordinates[:, 1] < -0.25)
        & ~outflow_buffer
    )
    return {
        "case": case,
        "geometry": geometry,
        "resolution": resolution,
        "coordinates": vertex_coordinates,
        "triangles": triangles,
        "vertex age": vertex_age,
        "cell age": cell_age,
        "tangent J": tangent_jacobian,
        "age-gradient J": age_jacobian,
        "valid": valid,
    }


def plot_stationary_fields(payloads, output_path):
    """Plot the stationary J fields and the independent age-gradient check."""
    ordered_payloads = [
        next(payload for payload in payloads if payload["case"]["case"] == name)
        for name in ("ordered", "overturned")
    ]
    valid_jacobians = np.concatenate(
        [payload["tangent J"][payload["valid"]] for payload in ordered_payloads]
    )
    limit = float(
        np.clip(np.quantile(np.abs(valid_jacobians), 0.99), 1.0, 1.5)
    )
    normalization = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.55),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.0, 0.90)},
    )
    field_image = None
    for panel, (axis, payload) in enumerate(zip(axes[:2], ordered_payloads)):
        coordinates = payload["coordinates"]
        triangulation = mtri.Triangulation(
            coordinates[:, 0],
            coordinates[:, 1],
            triangles=payload["triangles"],
        )
        triangulation.set_mask(~payload["valid"])
        field_image = axis.tripcolor(
            triangulation,
            facecolors=payload["tangent J"],
            shading="flat",
            cmap="RdBu",
            norm=normalization,
            rasterized=True,
        )

        valid_age = payload["cell age"][payload["valid"]]
        if np.max(valid_age) < 4_000.0:
            levels = np.arange(500.0, np.max(valid_age), 500.0)
        else:
            levels = np.array([2_000.0, 5_000.0, 10_000.0, 20_000.0, 30_000.0])
            levels = levels[levels < np.max(valid_age)]
        if len(levels):
            contours = axis.tricontour(
                triangulation,
                payload["vertex age"],
                levels=levels,
                colors="0.22",
                linewidths=0.55,
                alpha=0.75,
            )
            axis.clabel(
                contours,
                fmt=lambda age: f"{age / 1000:g}",
                fontsize=7,
                inline_spacing=2,
            )

        geometry = payload["geometry"]
        left_side, right_side = side_locations(geometry)
        bed_x = np.linspace(left_side, right_side, 501)
        axis.plot(bed_x, bed_height(bed_x, geometry), color="k", lw=0.9)
        axis.plot([left_side, right_side], [0.0, 0.0], color="k", lw=0.9)
        case = payload["case"]
        valid_j = payload["tangent J"][payload["valid"]]
        axis.set_title(
            f"{'AB'[panel]}   {case['case'].capitalize()}\n"
            rf"$H_b={case['boundary thickness (m)']:.0f}$ m, "
            rf"$H_c={case['center thickness (m)']:.0f}$ m; "
            rf"retained $\min J={np.min(valid_j):.2f}$",
            loc="left",
        )
        axis.set_xlabel("Horizontal position (m)")
        if panel == 0:
            axis.set_ylabel("Elevation (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(left_side, right_side)
        axis.set_ylim(float(np.min(bed_height(bed_x, geometry))) - 3.0, 3.0)
        axis.text(
            0.98,
            0.03,
            "age contours (kyr)",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.25",
        )
        axis.tick_params(direction="in")

    colorbar = figure.colorbar(
        field_image,
        ax=axes[:2],
        orientation="horizontal",
        fraction=0.08,
        pad=0.16,
        aspect=35,
    )
    colorbar.set_label(
        r"Stationary tangent Jacobian $J=T_x$ "
        r"(surface provenance $\geq0.95$; outflow buffer applied)"
    )

    identity_axis = axes[2]
    colors = {"ordered": "#2166ac", "overturned": "#b2182b"}
    all_values = []
    for payload in ordered_payloads:
        valid = payload["valid"]
        x_values = payload["age-gradient J"][valid]
        y_values = payload["tangent J"][valid]
        all_values.extend((x_values, y_values))
        error = y_values - x_values
        identity_axis.scatter(
            x_values,
            y_values,
            s=8,
            alpha=0.55,
            linewidths=0.0,
            color=colors[payload["case"]["case"]],
            label=(
                f"{payload['case']['case']} "
                f"(RMSE {np.sqrt(np.mean(error**2)):.3f})"
            ),
        )
    combined = np.concatenate(all_values)
    identity_limit = float(
        np.clip(np.quantile(np.abs(combined), 0.995), 1.0, 1.5)
    )
    identity_axis.plot(
        [-identity_limit, identity_limit],
        [-identity_limit, identity_limit],
        color="0.25",
        lw=0.9,
        zorder=0,
    )
    identity_axis.axhline(0.0, color="0.75", lw=0.6, zorder=0)
    identity_axis.axvline(0.0, color="0.75", lw=0.6, zorder=0)
    identity_axis.set_xlim(-identity_limit, identity_limit)
    identity_axis.set_ylim(-identity_limit, identity_limit)
    identity_axis.set_aspect("equal", adjustable="box")
    identity_axis.set_xlabel(r"Age-gradient Jacobian $-\dot a_0\tau_z$")
    identity_axis.set_ylabel(r"Tangent Jacobian $T_x$")
    identity_axis.set_title("C   Independent stationary identity", loc="left")
    identity_axis.legend(frameon=False, loc="upper left")
    identity_axis.tick_params(direction="in")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_benchmark(
    case, resolution, report_age=8_000.0, return_plot_payload=False
):
    """Compare one stationary solve with the established particle diagnostic."""
    boundary = case["boundary thickness (m)"]
    center = case["center thickness (m)"]
    geometry = geometry_from_thicknesses(boundary, center)
    start = perf_counter()
    velocity, stokes = solve_stokes(geometry, resolution)
    stokes_seconds = perf_counter() - start
    sample, speed_range = make_sampler(geometry, velocity)

    start = perf_counter()
    trajectory = trace_surface_map(
        geometry,
        sample,
        maximum_age=report_age,
        time_step=1.0,
        source_count=401,
        fixed_interior_report_ages=(report_age,),
    )
    trajectory_seconds = perf_counter() - start
    evaluation_age = float(
        trajectory[fixed_interior_end_age_column(report_age)]
    )
    source_interval = fixed_source_interval(geometry, sample)
    start = perf_counter()
    fields = solve_stationary_age(velocity)
    stationary_seconds = perf_counter() - start
    comparison_stationary = summarize_stationary_fields(
        fields,
        source_interval,
        maximum_age=evaluation_age,
        geometry=geometry,
        resolution=resolution,
    )
    full_stationary = summarize_stationary_fields(
        fields,
        source_interval,
        maximum_age=np.inf,
        geometry=geometry,
        resolution=resolution,
    )
    stationary_zero_age = first_stationary_zero_crossing_age(
        fields,
        source_interval,
        maximum_age=np.inf,
        geometry=geometry,
        resolution=resolution,
    )
    sensitivity = {}
    for threshold in PROVENANCE_THRESHOLDS:
        summary = (
            full_stationary
            if threshold == 0.95
            else summarize_stationary_fields(
                fields,
                source_interval,
                maximum_age=np.inf,
                geometry=geometry,
                resolution=resolution,
                provenance_threshold=threshold,
            )
        )
        label = f"provenance>={threshold:g}"
        sensitivity[f"{label} retained cell count"] = summary[
            "stationary valid cell count"
        ]
        sensitivity[f"{label} tangent minimum J"] = summary[
            "stationary tangent minimum J"
        ]
        sensitivity[f"{label} tangent J 1st percentile"] = summary[
            "stationary tangent J 1st percentile"
        ]
        sensitivity[f"{label} oldest retained age (yr)"] = summary[
            "stationary oldest retained age (yr)"
        ]
        sensitivity[f"{label} projected J=0 age (yr)"] = (
            stationary_zero_age
            if threshold == 0.95
            else first_stationary_zero_crossing_age(
                fields,
                source_interval,
                maximum_age=np.inf,
                geometry=geometry,
                resolution=resolution,
                provenance_threshold=threshold,
            )
        )

    row = {
        **case,
        "resolution (m)": resolution,
        "mesh cells": stokes["cells"],
        "Stokes solve wall time (s)": stokes_seconds,
        "stationary transport wall time (s)": stationary_seconds,
        "validation trajectory wall time (s)": trajectory_seconds,
        "minimum grid u (m/yr)": speed_range[0],
        "maximum grid u (m/yr)": speed_range[1],
        "fixed source minimum x0 (m)": source_interval[0],
        "fixed source maximum x0 (m)": source_interval[1],
        "comparison end age (yr)": evaluation_age,
        "trajectory minimum J": trajectory[
            fixed_interior_minimum_column(report_age)
        ],
        "trajectory first J=0 age (yr)": trajectory[
            "first fixed-interior zero-crossing age (yr)"
        ],
        "stationary tangent projected J=0 age (yr)": stationary_zero_age,
        **prefixed(comparison_stationary, "comparison-horizon"),
        **prefixed(full_stationary, "full-streamtube"),
        **sensitivity,
    }
    if return_plot_payload:
        return row, make_plot_payload(
            case, geometry, resolution, fields, source_interval
        )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=float,
        action="append",
        help="Target mesh spacing in m; may be repeated (default: 10, 7.5)",
    )
    parser.add_argument(
        "--case",
        choices=[case["case"] for case in CASES],
        action="append",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "data"
            / "steady_age_elliptical_diagnostic.csv"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "figures"
            / "steady_age_elliptical_diagnostic.png"
        ),
    )
    args = parser.parse_args()
    resolutions = args.resolution or [10.0, 7.5]
    selected = set(args.case or [case["case"] for case in CASES])

    rows = []
    plot_payloads = []
    plot_resolution = min(resolutions)
    for resolution in resolutions:
        for case in CASES:
            if case["case"] not in selected:
                continue
            print(
                f"Solving {case['case']} case at {resolution:g} m",
                flush=True,
            )
            result = run_benchmark(
                case,
                resolution,
                return_plot_payload=(resolution == plot_resolution),
            )
            if resolution == plot_resolution:
                row, payload = result
                plot_payloads.append(payload)
            else:
                row = result
            rows.append(row)
            print(pd.Series(row).to_string(), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, na_rep="NA")
    print(f"Wrote {args.output}", flush=True)
    if {payload["case"]["case"] for payload in plot_payloads} == {
        case["case"] for case in CASES
    }:
        plot_stationary_fields(plot_payloads, args.figure)
        print(f"Wrote {args.figure}", flush=True)


if __name__ == "__main__":
    main()
