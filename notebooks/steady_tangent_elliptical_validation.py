"""Validate an age-independent overturn diagnostic in elliptical troughs.

For a steady velocity field v, the material tangent T associated with a
surface-deposition coordinate x0 obeys

    dX/dt = v(X),       dT/dt = grad(v)(X) T,
    X(0) = (x0, -0.05 m),  T(0) = (1, 0).

Thus T_x is the horizontal order Jacobian d x / d x0 without differencing
neighboring trajectories.  This script integrates that variational equation
adaptively to the first sampled-domain exit and compares T_x with the
centered neighboring-source diagnostic used by the parameter-map scripts.
The glacier and its velocity are steady throughout; integration time is
particle age, not evolution time of the glacier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parent
if str(NOTEBOOK_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIRECTORY))

from realistic_geometry_elliptical_family import (  # noqa: E402
    bed_height,
    material_curve_snapshots,
    side_locations,
    solve_stokes,
)
from realistic_geometry_elliptical_parameter_map import (  # noqa: E402
    geometry_from_thicknesses,
)


CASES = (
    {
        "case": "ordered slow flow",
        "boundary thickness (m)": 70.0,
        "center thickness (m)": 270.0,
        "comparison age (yr)": 8_000.0,
    },
    {
        "case": "overturned unidirectional flow",
        "boundary thickness (m)": 70.0,
        "center thickness (m)": 150.0,
        "comparison age (yr)": 1_900.0,
    },
    {
        "case": "ordered early exit",
        "boundary thickness (m)": 40.0,
        "center thickness (m)": 130.0,
        "comparison age (yr)": 1_600.0,
    },
    {
        "case": "overturned counterflow",
        "boundary thickness (m)": 180.0,
        "center thickness (m)": 240.0,
        "comparison age (yr)": 2_800.0,
    },
)


def bed_slope(x, geometry):
    """Derivative of the piecewise-elliptical bed elevation."""
    x = np.asarray(x, dtype=float)
    radius = np.where(
        x < 0.0, geometry["left radius"], geometry["right radius"]
    )
    inside = np.maximum(1.0e-14, 1.0 - (x / radius) ** 2)
    return (
        geometry["vertical radius"]
        * x
        / (radius**2 * np.sqrt(inside))
    )


def make_variational_sampler(
    geometry, velocity, nx=601, nz=181
):
    """Return the same bilinear velocity sampler and its exact gradient.

    The velocity is sampled on the bed-following grid used by the existing
    material-map calculation.  Within each bilinear grid cell, the returned
    physical gradient is the analytical derivative of that same interpolant.
    """
    left_side, right_side = side_locations(geometry)
    x_grid = np.linspace(0.995 * left_side, 0.995 * right_side, nx)
    sigma_grid = np.linspace(0.005, 0.99995, nz)
    bed = bed_height(x_grid, geometry)
    X = np.tile(x_grid, (nz, 1))
    Z = bed[None, :] * (1.0 - sigma_grid[:, None])
    points = np.column_stack((X.ravel(), Z.ravel()))
    values = np.asarray(
        velocity.at(points, tolerance=1.0e-7), dtype=float
    ).reshape(nz, nx, 2)

    def interpolate(points):
        """Bilinear velocity and coordinate derivatives at physical points."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        x = points[:, 0]
        z = points[:, 1]
        local_bed = bed_height(x, geometry)
        thickness = -local_bed
        sigma = (z - local_bed) / thickness

        ix = np.searchsorted(x_grid, x, side="right") - 1
        iz = np.searchsorted(sigma_grid, sigma, side="right") - 1
        ix = np.clip(ix, 0, nx - 2)
        iz = np.clip(iz, 0, nz - 2)
        dx = x_grid[ix + 1] - x_grid[ix]
        ds = sigma_grid[iz + 1] - sigma_grid[iz]
        fx = (x - x_grid[ix]) / dx
        fs = (sigma - sigma_grid[iz]) / ds

        v00 = values[iz, ix]
        v01 = values[iz, ix + 1]
        v10 = values[iz + 1, ix]
        v11 = values[iz + 1, ix + 1]
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
        ) / dx[:, None]

        # Chain rule from (sigma, x) to physical (x, z).
        local_bed_slope = bed_slope(x, geometry)
        sigma_x = -local_bed_slope * (1.0 - sigma) / thickness
        sigma_z = 1.0 / thickness
        gradient = np.empty((len(points), 2, 2))
        gradient[:, :, 0] = (
            derivative_x_sigma + derivative_sigma * sigma_x[:, None]
        )
        gradient[:, :, 1] = derivative_sigma * sigma_z[:, None]
        return velocity_value, gradient

    def sample(name, x, z):
        x, z = np.broadcast_arrays(x, z)
        points = np.column_stack((x.ravel(), z.ravel()))
        component = 0 if name == "u" else 1
        result = interpolate(points)[0][:, component].reshape(x.shape)
        return float(result) if result.ndim == 0 else result

    def sample_vector(points):
        return interpolate(points)[0]

    def sample_gradient(points):
        return interpolate(points)[1]

    return sample, sample_vector, sample_gradient


def exit_events(geometry):
    """Events matching the conservative physical-domain mask in the map."""
    left_side, right_side = side_locations(geometry)

    def left_exit(_age, state):
        return state[0] - 0.994 * left_side

    def right_exit(_age, state):
        return 0.994 * right_side - state[0]

    def bed_exit(_age, state):
        bed = float(bed_height(state[0], geometry))
        sigma = (state[1] - bed) / (-bed)
        return sigma - 0.006

    def surface_exit(_age, state):
        bed = float(bed_height(state[0], geometry))
        sigma = (state[1] - bed) / (-bed)
        return 0.99994 - sigma

    events = (left_exit, right_exit, bed_exit, surface_exit)
    for event in events:
        event.terminal = True
        event.direction = -1.0
    return events, ("left side", "right side", "bed", "surface")


def integrate_tangent(
    geometry,
    sample_vector,
    sample_gradient,
    source_x,
    maximum_age=120_000.0,
    maximum_step=2.0,
    rtol=1.0e-9,
):
    """Adaptively integrate one trajectory and its material tangent."""
    initial = np.array([source_x, -0.05, 1.0, 0.0])

    def rhs(_age, state):
        point = state[None, :2]
        velocity = sample_vector(point)[0]
        tangent_rate = sample_gradient(point)[0] @ state[2:]
        return np.r_[velocity, tangent_rate]

    events, event_names = exit_events(geometry)

    def first_overturn(_age, state):
        return state[2]

    first_overturn.terminal = False
    first_overturn.direction = -1.0
    started = time.perf_counter()
    solution = solve_ivp(
        rhs,
        (0.0, maximum_age),
        initial,
        events=(*events, first_overturn),
        dense_output=True,
        max_step=maximum_step,
        rtol=rtol,
        atol=np.array([1.0e-9, 1.0e-9, 1.0e-10, 1.0e-10]),
    )
    runtime = time.perf_counter() - started
    if not solution.success:
        raise RuntimeError(f"tangent integration failed: {solution.message}")
    event_times = [
        times[0] if len(times) else np.inf
        for times in solution.t_events[: len(events)]
    ]
    event_index = int(np.argmin(event_times))
    if np.isfinite(event_times[event_index]):
        exit_age = float(event_times[event_index])
        exit_kind = event_names[event_index]
    else:
        exit_age = float(solution.t[-1])
        exit_kind = "safety cap"

    # Sample the dense solution only to summarize extrema.  The sign test is
    # separately refined below by a root of T_x.
    summary_ages = np.linspace(0.0, exit_age, 4001)
    summary_states = solution.sol(summary_ages)
    horizontal_velocity = sample_vector(summary_states[:2].T)[:, 0]
    tangent_x = summary_states[2]
    reversal_indices = np.flatnonzero(
        np.signbit(horizontal_velocity[:-1])
        != np.signbit(horizontal_velocity[1:])
    )
    reversal_ages = []
    reversal_x = []
    if len(reversal_indices):
        from scipy.optimize import brentq

        for lower in reversal_indices:
            upper = lower + 1
            crossing_age = brentq(
                lambda age: float(
                    sample_vector(solution.sol(age)[:2][None, :])[0, 0]
                ),
                summary_ages[lower],
                summary_ages[upper],
            )
            reversal_ages.append(float(crossing_age))
            reversal_x.append(float(solution.sol(crossing_age)[0]))
    first_zero_age = (
        float(solution.t_events[-1][0])
        if len(solution.t_events[-1])
        else np.nan
    )
    minimum_index = int(np.argmin(tangent_x))
    return {
        "solution": solution,
        "exit age (yr)": exit_age,
        "exit kind": exit_kind,
        "first T_x=0 age (yr)": first_zero_age,
        "minimum direct T_x": float(tangent_x[minimum_index]),
        "age of minimum direct T_x (yr)": float(summary_ages[minimum_index]),
        "minimum path u (m/yr)": float(np.min(horizontal_velocity)),
        "horizontal velocity reversal count": int(len(reversal_ages)),
        "first u=0 age (yr)": (
            reversal_ages[0] if reversal_ages else np.nan
        ),
        "first u=0 x (m)": reversal_x[0] if reversal_x else np.nan,
        "last u=0 age (yr)": (
            reversal_ages[-1] if reversal_ages else np.nan
        ),
        "last u=0 x (m)": reversal_x[-1] if reversal_x else np.nan,
        "adaptive steps": int(len(solution.t) - 1),
        "function evaluations": int(solution.nfev),
        "trajectory runtime (s)": runtime,
    }


def integrate_neighbor_pair(
    sample_vector, source_x, delta, age, maximum_step=2.0
):
    """Centered finite difference from two independently advected markers."""
    initial = np.array(
        [source_x - delta, -0.05, source_x + delta, -0.05]
    )

    def rhs(_age, state):
        points = state.reshape(2, 2)
        return sample_vector(points).ravel()

    solution = solve_ivp(
        rhs,
        (0.0, age),
        initial,
        max_step=maximum_step,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    if not solution.success:
        raise RuntimeError(
            f"neighbor-pair integration failed: {solution.message}"
        )
    x_minus, _, x_plus, _ = solution.y[:, -1]
    return (x_plus - x_minus) / (2.0 * delta)


def selected_source(snapshot):
    """Return the minimum-J source within the fixed central 80% band."""
    sources = snapshot["sources"]
    margin = int(np.ceil(0.10 * len(sources)))
    fixed = np.zeros(len(sources), dtype=bool)
    fixed[margin : len(sources) - margin] = True
    usable = fixed & snapshot["valid"] & np.isfinite(snapshot["jacobian"])
    if not np.any(usable):
        raise RuntimeError("no fixed-interior centered Jacobian at comparison age")
    candidates = np.flatnonzero(usable)
    index = int(candidates[np.argmin(snapshot["jacobian"][candidates])])
    spacing = abs(sources[index + 1] - sources[index])
    return index, float(sources[index]), float(spacing)


def run_validation(resolution=10.0, source_count=401):
    """Solve the four cases and return validation and convergence tables."""
    validation_rows = []
    convergence_rows = []
    total_started = time.perf_counter()
    for case in CASES:
        geometry = geometry_from_thicknesses(
            case["boundary thickness (m)"], case["center thickness (m)"]
        )
        solve_started = time.perf_counter()
        velocity, _ = solve_stokes(geometry, resolution)
        stokes_runtime = time.perf_counter() - solve_started
        sample, sample_vector, sample_gradient = make_variational_sampler(
            geometry, velocity
        )
        snapshot = material_curve_snapshots(
            geometry,
            sample,
            ages=(case["comparison age (yr)"],),
            time_step=1.0,
            source_count=source_count,
        )[0]
        index, source_x, source_spacing = selected_source(snapshot)
        trace = integrate_tangent(
            geometry,
            sample_vector,
            sample_gradient,
            source_x,
            maximum_step=2.0,
        )
        comparison_age = case["comparison age (yr)"]
        if comparison_age >= trace["exit age (yr)"]:
            raise RuntimeError(
                f"{case['case']} source exits before its comparison age"
            )
        direct_state = trace["solution"].sol(comparison_age)
        existing_neighbor_j = float(snapshot["jacobian"][index])
        row = case.copy()
        row.update(
            {
                "source x0 (m)": source_x,
                "source spacing (m)": source_spacing,
                "direct T_x at comparison age": float(direct_state[2]),
                "existing centered J at comparison age": existing_neighbor_j,
                "direct minus centered J": float(
                    direct_state[2] - existing_neighbor_j
                ),
                "direct x at comparison age (m)": float(direct_state[0]),
                "existing x at comparison age (m)": float(
                    snapshot["states"][index, 0]
                ),
                "direct minus existing x (m)": float(
                    direct_state[0] - snapshot["states"][index, 0]
                ),
                "Stokes solve runtime (s)": stokes_runtime,
            }
        )
        for key, value in trace.items():
            if key != "solution":
                row[key] = value
        validation_rows.append(row)

        for maximum_step in (8.0, 4.0, 2.0, 1.0):
            check = integrate_tangent(
                geometry,
                sample_vector,
                sample_gradient,
                source_x,
                maximum_age=max(comparison_age + 1.0, 1.1 * trace["exit age (yr)"]),
                maximum_step=maximum_step,
            )
            convergence_rows.append(
                {
                    "case": case["case"],
                    "check": "adaptive step",
                    "sample nx": 601,
                    "sample nz": 181,
                    "maximum step (yr)": maximum_step,
                    "source half-spacing delta (m)": np.nan,
                    "J at comparison age": float(
                        check["solution"].sol(comparison_age)[2]
                    ),
                    "exit age (yr)": check["exit age (yr)"],
                    "runtime (s)": check["trajectory runtime (s)"],
                }
            )
        for fraction in (2.0, 1.0, 0.5, 0.25):
            delta = fraction * source_spacing
            neighbor_j = integrate_neighbor_pair(
                sample_vector,
                source_x,
                delta,
                comparison_age,
                maximum_step=2.0,
            )
            convergence_rows.append(
                {
                    "case": case["case"],
                    "check": "centered source difference",
                    "sample nx": 601,
                    "sample nz": 181,
                    "maximum step (yr)": 2.0,
                    "source half-spacing delta (m)": delta,
                    "J at comparison age": neighbor_j,
                    "exit age (yr)": np.nan,
                    "runtime (s)": np.nan,
                }
            )
        if case["case"] == "overturned counterflow":
            for sample_nx, sample_nz in ((301, 91), (601, 181), (901, 271)):
                _, grid_vector, grid_gradient = make_variational_sampler(
                    geometry,
                    velocity,
                    nx=sample_nx,
                    nz=sample_nz,
                )
                check = integrate_tangent(
                    geometry,
                    grid_vector,
                    grid_gradient,
                    source_x,
                    maximum_age=max(
                        comparison_age + 1.0,
                        1.1 * trace["exit age (yr)"],
                    ),
                    maximum_step=2.0,
                )
                convergence_rows.append(
                    {
                        "case": case["case"],
                        "check": "velocity sampling grid",
                        "sample nx": sample_nx,
                        "sample nz": sample_nz,
                        "maximum step (yr)": 2.0,
                        "source half-spacing delta (m)": np.nan,
                        "J at comparison age": float(
                            check["solution"].sol(comparison_age)[2]
                        ),
                        "exit age (yr)": check["exit age (yr)"],
                        "runtime (s)": check["trajectory runtime (s)"],
                    }
                )
        print(
            f"{case['case']}: T_x={direct_state[2]:.6f}, "
            f"centered J={existing_neighbor_j:.6f}, "
            f"exit={trace['exit age (yr)']:.1f} yr ({trace['exit kind']})",
            flush=True,
        )
    return (
        pd.DataFrame(validation_rows),
        pd.DataFrame(convergence_rows),
        time.perf_counter() - total_started,
    )


def write_report(path, validation, convergence, total_runtime):
    """Write a compact, reproducible interpretation of the validation."""
    direct_error = np.abs(validation["direct minus centered J"])
    step_groups = convergence[convergence["check"] == "adaptive step"]
    step_spread = step_groups.groupby("case")["J at comparison age"].agg(
        lambda values: float(np.ptp(values))
    )
    finite_difference = convergence[
        convergence["check"] == "centered source difference"
    ].copy()
    direct_lookup = validation.set_index("case")["direct T_x at comparison age"]
    finite_difference["absolute error"] = finite_difference.apply(
        lambda row: abs(row["J at comparison age"] - direct_lookup[row["case"]]),
        axis=1,
    )
    smallest_delta_error = (
        finite_difference.sort_values("source half-spacing delta (m)")
        .groupby("case")
        .first()["absolute error"]
    )
    sampling_grid = convergence[
        convergence["check"] == "velocity sampling grid"
    ]
    sampling_grid_spread = float(
        np.ptp(sampling_grid["J at comparison age"])
    )
    lines = [
        "# Steady variational-tangent validation",
        "",
        "## Calculation",
        "",
        "The full-Stokes velocity is solved once for each fixed geometry.  "
        "Individual surface-origin characteristics then carry the material "
        "tangent by `dT/dt = grad(v) T`.  This is a steady-flow sensitivity "
        "calculation: particle age advances, but the glacier does not evolve.  "
        "Each characteristic ends at its first bed, surface, or side exit; "
        "120 kyr is only a numerical safety cap.",
        "",
        "The deposition line is horizontal at `z=-0.05 m`, so `T(0)=(1,0)`.  "
        "Velocity and its analytical bilinear derivative use the same "
        "601 by 181 bed-following sampling grid as the existing material map.  "
        "Comparisons use only the fixed central 80% source band.",
        "",
        "## Result",
        "",
        f"The direct tangent and existing centered-source Jacobian differ by "
        f"at most {direct_error.max():.3g} at the four comparison points.  "
        f"Changing the adaptive maximum step from 8 to 1 yr changes T_x by "
        f"at most {step_spread.max():.3g}.  At the smallest tested source "
        f"half-spacing, centered differences agree with T_x to at most "
        f"{smallest_delta_error.max():.3g}.  Changing the counterflow velocity "
        f"sampling grid from 301 by 91 to 901 by 271 changes its comparison "
        f"T_x over a range of {sampling_grid_spread:.3g}.",
        "",
    ]
    for _, row in validation.iterrows():
        crossing = (
            f"first T_x=0 at {row['first T_x=0 age (yr)']:.1f} yr"
            if np.isfinite(row["first T_x=0 age (yr)"])
            else "no T_x=0 crossing before exit"
        )
        reversal_detail = ""
        if row["horizontal velocity reversal count"]:
            reversal_detail = (
                f", at {row['first u=0 age (yr)']:.1f} and "
                f"{row['last u=0 age (yr)']:.1f} yr"
            )
        lines.append(
            f"- selected characteristic, {row['case']}: minimum "
            f"T_x={row['minimum direct T_x']:.4f}; "
            f"{crossing}; exit at {row['exit age (yr)']:.1f} yr "
            f"({row['exit kind']}); minimum path u="
            f"{row['minimum path u (m/yr)']:.4g} m/yr; horizontal velocity "
            f"reversals={int(row['horizontal velocity reversal count'])}"
            f"{reversal_detail}."
        )
    lines += [
        "",
        f"The calculation took {total_runtime:.1f} s on this machine, including "
        "four Stokes solves and all convergence integrations.",
        "",
        "## Interpretation and limitations",
        "",
        "A negative `T_x` is the same loss of horizontal source order detected "
        "by neighboring trajectories, but it does not require selecting a "
        "reporting age or differencing the last surviving markers.  The method "
        "still integrates cumulative deformation along steady characteristics; "
        "it removes transient glacier simulation, not material history.",
        "",
        "The exit surface is the conservative inset mask used by the existing "
        "elliptical calculation (0.994 of each side and sigma=0.006--0.99994), "
        "because the sampled velocity grid is inset from the curved finite-"
        "element boundary.  The validation checks transport and source-spacing "
        "convergence on the production 10 m Stokes mesh; it does not replace "
        "the existing Stokes mesh-convergence tests.  The companion stationary "
        "Eulerian solve treats open surface-fed streamtubes; closed "
        "recirculation and exact stagnation still require separate treatment.",
        "",
        "## Machine-readable outputs",
        "",
        "- `data/steady_tangent_elliptical_validation.csv`",
        "- `data/steady_tangent_elliptical_convergence.csv`",
        "- `data/steady_tangent_elliptical_validation_metadata.json`",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--source-count", type=int, default=401)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=NOTEBOOK_DIRECTORY.parent / "data",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    validation, convergence, total_runtime = run_validation(
        resolution=args.resolution, source_count=args.source_count
    )
    validation_path = (
        args.output_directory / "steady_tangent_elliptical_validation.csv"
    )
    convergence_path = (
        args.output_directory / "steady_tangent_elliptical_convergence.csv"
    )
    metadata_path = (
        args.output_directory
        / "steady_tangent_elliptical_validation_metadata.json"
    )
    report_path = NOTEBOOK_DIRECTORY / "steady_tangent_elliptical_validation.md"
    validation.to_csv(validation_path, index=False)
    convergence.to_csv(convergence_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "resolution (m)": args.resolution,
                "source count": args.source_count,
                "velocity sample grid": [601, 181],
                "total runtime (s)": total_runtime,
                "description": (
                    "Adaptive steady-characteristic integration of the exact "
                    "material-tangent variational equation."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    write_report(report_path, validation, convergence, total_runtime)
    print(f"Wrote {validation_path}")
    print(f"Wrote {convergence_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
