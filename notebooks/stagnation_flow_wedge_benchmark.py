"""Isochrone-order test for the Manos--Lipovsky analytical wedge flow.

The public ``bradlipovsky/wedge`` notebook prescribes

    psi = -r**lambda sin(lambda theta),  lambda = 2 pi / alpha,

in a wedge of angle alpha = 3 pi / 4.  The circular outer boundary is an
inflow (accumulation) boundary near each bed arm and an outflow (ablation)
boundary in the middle.  This script launches equal-age material curves
from each connected inflow arc and tests dx/dx0, not merely layer dip.

The field is an exact free-slip benchmark.  Its tangential traction on both
wedge arms is zero while tangential velocity is nonzero, so it does *not*
satisfy a finite-mobility linear sliding law.  The companion script
``stagnation_flow_linear_sliding.py`` performs that mechanically distinct
test.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALPHA = 3.0 * np.pi / 4.0
LAMBDA = 2.0 * np.pi / ALPHA
POWER = LAMBDA - 1.0
OUTER_RADIUS = 1.0
OUTPUT_DIRECTORY = Path("notebooks/stagnation_flow_results")


def velocity(points):
    """Dimensionless Cartesian velocity, scaled by U_R at r=R."""
    points = np.asarray(points)
    x, z = points[..., 0], points[..., 1]
    radius = np.hypot(x, z)
    theta = np.arctan2(z, x)
    amplitude = radius**POWER
    u = -amplitude * np.cos(POWER * theta)
    w = amplitude * np.sin(POWER * theta)
    return np.stack((u, w), axis=-1)


def rk4(points, step):
    k1 = velocity(points)
    k2 = velocity(points + 0.5 * step * k1)
    k3 = velocity(points + 0.5 * step * k2)
    k4 = velocity(points + step * k3)
    return points + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def source_theta(branch, count):
    """Outer-arc coordinates for one connected accumulation branch."""
    margin = 2.0e-3
    if branch == "shallow/right":
        return np.linspace(margin, ALPHA / 4.0 - margin, count)
    if branch == "steep/left":
        return np.linspace(3.0 * ALPHA / 4.0 + margin, ALPHA - margin, count)
    raise ValueError(branch)


def horizontal_order(points, theta, active):
    """Compute dx/dx0 on contiguous active source intervals."""
    jacobian = np.full(len(theta), np.nan)
    indices = np.flatnonzero(active)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    initial_derivative = -OUTER_RADIUS * np.sin(theta)
    for group in groups:
        if len(group) >= 7:
            dx_dtheta = np.gradient(points[group, 0], theta[group])
            jacobian[group] = dx_dtheta / initial_derivative[group]
            # Finite differences at a truncated material-curve endpoint are
            # not used to diagnose overturning.
            jacobian[group[:2]] = np.nan
            jacobian[group[-2:]] = np.nan
    return jacobian


def integrate_branch(branch, count=1201, step=1.0e-4, maximum_time=0.6):
    """Advect one source arc and retain markers only until first exit."""
    theta = source_theta(branch, count)
    points = OUTER_RADIUS * np.column_stack((np.cos(theta), np.sin(theta)))
    active = np.ones(count, dtype=bool)
    # Do not let the surface--bed corner or the accumulation--ablation
    # endpoint set the reported threshold.  The central 80% is a conservative
    # connected subset of the source arc.
    source_interior = np.zeros(count, dtype=bool)
    source_interior[int(0.1 * count) : int(0.9 * count)] = True
    times = [0.0]
    positions = [points.copy()]
    active_history = [active.copy()]
    jacobians = [np.ones(count)]
    minimum_jacobian = [1.0]
    minimum_index = [count // 2]

    steps = int(np.ceil(maximum_time / step))
    for step_index in range(steps):
        indices = np.flatnonzero(active)
        if len(indices) < 7:
            break
        points[indices] = rk4(points[indices], step)
        radius = np.linalg.norm(points, axis=1)
        current_theta = np.arctan2(points[:, 1], points[:, 0])
        active &= (
            (radius < OUTER_RADIUS - 1.0e-8)
            & (current_theta > 0.0)
            & (current_theta < ALPHA)
        )
        diagnostic_active = (
            active & source_interior & (radius < 0.995 * OUTER_RADIUS)
        )
        jacobian = horizontal_order(points, theta, diagnostic_active)
        if np.any(np.isfinite(jacobian)):
            index = int(np.nanargmin(jacobian))
            minimum = float(jacobian[index])
        else:
            index = 0
            minimum = np.nan
        time = (step_index + 1) * step
        times.append(time)
        recorded = points.copy()
        recorded[~active] = np.nan
        positions.append(recorded)
        active_history.append(active.copy())
        jacobians.append(jacobian)
        minimum_jacobian.append(minimum)
        minimum_index.append(index)

    times = np.asarray(times)
    positions = np.asarray(positions)
    active_history = np.asarray(active_history)
    jacobians = np.asarray(jacobians)
    minimum_jacobian = np.asarray(minimum_jacobian)
    minimum_index = np.asarray(minimum_index)
    negative = np.flatnonzero(minimum_jacobian < 0.0)
    if len(negative):
        crossing_index = negative[0]
        before = crossing_index - 1
        fraction = minimum_jacobian[before] / (
            minimum_jacobian[before] - minimum_jacobian[crossing_index]
        )
        crossing_time = times[before] + fraction * (
            times[crossing_index] - times[before]
        )
        marker = minimum_index[crossing_index]
        crossing_position = positions[crossing_index, marker]
        crossing_theta0 = theta[marker]
    else:
        crossing_time = np.nan
        crossing_position = np.array((np.nan, np.nan))
        crossing_theta0 = np.nan
    median_index = count // 2
    median_negative = np.flatnonzero(jacobians[:, median_index] < 0.0)
    median_crossing_time = (
        times[median_negative[0]] if len(median_negative) else np.nan
    )
    return {
        "branch": branch,
        "theta": theta,
        "times": times,
        "positions": positions,
        "active": active_history,
        "Jx": jacobians,
        "minimum_Jx": minimum_jacobian,
        "crossing_time": crossing_time,
        "crossing_position": crossing_position,
        "crossing_theta0": crossing_theta0,
        "median_source_vertical_time": median_crossing_time,
        "step": step,
        "count": count,
    }


def run_convergence():
    configurations = (
        (601, 2.0e-4),
        (1201, 1.0e-4),
        (2401, 5.0e-5),
    )
    rows = []
    finest = {}
    for count, step in configurations:
        for branch in ("shallow/right", "steep/left"):
            print(f"Tracing {branch}, {count} sources, dt={step:g}", flush=True)
            result = integrate_branch(branch, count, step)
            original_time_index = np.argmin(
                np.abs(
                    result["times"]
                    - 0.1 * LAMBDA * 2.0 ** (LAMBDA - 2.0)
                )
            )
            away_from_bed = (
                result["active"][original_time_index]
                & (result["theta"] >= np.radians(15.0))
                & np.isfinite(result["Jx"][original_time_index])
            )
            away_values = result["Jx"][original_time_index, away_from_bed]
            rows.append(
                {
                    "branch": branch,
                    "source_count": count,
                    "time_step": step,
                    "first_vertical_time": result["crossing_time"],
                    "first_vertical_time_R2_Aminus1": (
                        result["crossing_time"]
                        / (LAMBDA * 2.0 ** (LAMBDA - 2.0))
                    ),
                    "median_source_vertical_time": result[
                        "median_source_vertical_time"
                    ],
                    "minimum_Jx_to_t_0.6": np.nanmin(result["minimum_Jx"]),
                    "active_at_t_0.6": int(result["active"][-1].sum()),
                    "crossing_x": result["crossing_position"][0],
                    "crossing_z": result["crossing_position"][1],
                    "crossing_source_theta_deg": np.degrees(
                        result["crossing_theta0"]
                    ),
                    "Jmin_at_R2_t0.10_theta_ge15": (
                        np.min(away_values) if len(away_values) else np.nan
                    ),
                    "Jmax_at_R2_t0.10_theta_ge15": (
                        np.max(away_values) if len(away_values) else np.nan
                    ),
                    "active_count_R2_t0.10_theta_ge15": len(away_values),
                }
            )
            if count == configurations[-1][0]:
                finest[branch] = result
    return pd.DataFrame(rows), finest


def make_figure(results):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    x = np.linspace(-0.72, 1.0, 260)
    z = np.linspace(0.005, 1.0, 220)
    X, Z = np.meshgrid(x, z)
    points = np.column_stack((X.ravel(), Z.ravel()))
    values = velocity(points).reshape(X.shape + (2,))
    theta = np.arctan2(Z, X)
    radius = np.hypot(X, Z)
    outside = (theta <= 0.0) | (theta >= ALPHA) | (radius >= OUTER_RADIUS)
    U = values[..., 0]
    W = values[..., 1]
    U[outside] = np.nan
    W[outside] = np.nan
    speed = np.hypot(U, W)
    stream = axes[0].streamplot(
        x, z, U, W, color=speed, cmap="viridis", density=1.1, linewidth=0.8
    )
    arc_theta = np.linspace(0.0, ALPHA, 300)
    axes[0].plot(np.cos(arc_theta), np.sin(arc_theta), color="0.1")
    axes[0].plot([0, 1], [0, 0], color="0.1")
    axes[0].plot(
        [0, np.cos(ALPHA)], [0, np.sin(ALPHA)], color="0.1"
    )
    fig.colorbar(stream.lines, ax=axes[0], label="dimensionless speed")
    axes[0].set_title("(a) Analytical wedge flow")

    colors = {"shallow/right": "#4477AA", "steep/left": "#EE7733"}
    selected_times = (0.0, 0.05, 0.08, 0.10, 0.20, 0.40)
    for branch, result in results.items():
        for selected_time in selected_times:
            index = np.argmin(np.abs(result["times"] - selected_time))
            curve = result["positions"][index]
            axes[1].plot(
                curve[:, 0],
                curve[:, 1],
                color=colors[branch],
                alpha=0.25 + 0.7 * selected_time / max(selected_times),
                linewidth=1.0,
            )
        if np.isfinite(result["crossing_time"]):
            axes[1].scatter(
                *result["crossing_position"],
                facecolors="none",
                edgecolors="red",
                s=45,
                zorder=5,
            )
    axes[1].plot(np.cos(arc_theta), np.sin(arc_theta), color="0.1")
    axes[1].plot([0, 1], [0, 0], color="0.1")
    axes[1].plot(
        [0, np.cos(ALPHA)], [0, np.sin(ALPHA)], color="0.1"
    )
    axes[1].set_title("(b) Equal-age source curves")

    for branch, result in results.items():
        axes[2].plot(
            result["times"],
            result["minimum_Jx"],
            color=colors[branch],
            label=branch,
        )
    axes[2].axhline(0.0, color="0.4", linewidth=0.8)
    axes[2].set(
        xlabel=r"dimensionless age $tU_R/R$",
        ylabel=r"minimum $\partial x/\partial x_0$",
        title="(c) Vertical-tangent test",
        ylim=(-2.0, 2.0),
    )
    axes[2].legend(frameon=False)
    for axis in axes[:2]:
        axis.set(
            xlabel="$x/R$",
            ylabel="$z/R$",
            xlim=(-0.73, 1.02),
            ylim=(-0.02, 1.02),
        )
        axis.set_aspect("equal")
    path = OUTPUT_DIRECTORY / "stagnation_flow_wedge_benchmark.png"
    fig.savefig(path, dpi=220)
    return path


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table, results = run_convergence()
    table.to_csv(
        OUTPUT_DIRECTORY / "stagnation_flow_wedge_convergence.csv", index=False
    )
    figure_path = make_figure(results)
    print("\nAnalytical wedge material-map diagnostic")
    print(table.to_string(index=False))
    print("\nBasal check: tau_t = 0 while u_t != 0 on both wedge arms.")
    print("This is free slip (B=0, C=infinity), not finite linear sliding.")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
