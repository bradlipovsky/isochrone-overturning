"""Isochrone-order test for an analytical convergent-flow benchmark.

The steady streamfunction is

    psi = -r**lambda sin(lambda theta),  lambda = 2 pi / alpha,

in a sector of angle ``alpha = 3 pi / 4``.  The circular outer boundary
contains two accumulation (inflow) arcs separated by an ablation (outflow)
arc.  Equal-age material curves are therefore launched from two disconnected
source branches.  The diagnostic is ``J_x = dx/dx0`` on each branch, not
merely layer dip.

This is an exact free-slip benchmark: tangential traction on both straight
arms is zero while tangential velocity is nonzero.  It therefore does *not*
satisfy a finite-mobility linear sliding law.  The companion script
``stagnation_flow_linear_sliding.py`` performs that mechanically distinct
test.

The legacy output file names retain ``wedge`` so existing notebooks continue
to run, but all displayed names use "analytical convergent flow."
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ALPHA = 3.0 * np.pi / 4.0
LAMBDA = 2.0 * np.pi / ALPHA
POWER = LAMBDA - 1.0
OUTER_RADIUS = 1.0
SOURCE_MARGIN = 2.0e-3
DEFAULT_SOURCE_TRIM = 0.10
REFERENCE_TIME = 0.1 * LAMBDA * 2.0 ** (LAMBDA - 2.0)
OUTPUT_DIRECTORY = Path("notebooks/stagnation_flow_results")

BRANCH_COLORS = {
    "shallow/right": "#4477AA",
    "steep/left": "#EE7733",
}
BRANCH_LABELS = {
    "shallow/right": "right branch (shallow arm)",
    "steep/left": "left branch (steep arm)",
}
BRANCH_LINESTYLES = {
    "shallow/right": "-",
    "steep/left": (0, (4, 2)),
}


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
    """Advance material points one classical fourth-order Runge--Kutta step."""
    k1 = velocity(points)
    k2 = velocity(points + 0.5 * step * k1)
    k3 = velocity(points + 0.5 * step * k2)
    k4 = velocity(points + step * k3)
    return points + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def source_theta(branch, count):
    """Outer-arc coordinates for one connected accumulation branch."""
    if branch == "shallow/right":
        return np.linspace(SOURCE_MARGIN, ALPHA / 4.0 - SOURCE_MARGIN, count)
    if branch == "steep/left":
        return np.linspace(
            3.0 * ALPHA / 4.0 + SOURCE_MARGIN,
            ALPHA - SOURCE_MARGIN,
            count,
        )
    raise ValueError(branch)


def horizontal_order(points, theta, active):
    """Compute dx/dx0 on each contiguous active source interval."""
    jacobian = np.full(len(theta), np.nan)
    indices = np.flatnonzero(active)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    initial_derivative = -OUTER_RADIUS * np.sin(theta)
    for group in groups:
        if len(group) >= 7:
            dx_dtheta = np.gradient(points[group, 0], theta[group])
            jacobian[group] = dx_dtheta / initial_derivative[group]
            # Do not diagnose overturning from finite differences at a
            # truncated material-curve endpoint.
            jacobian[group[:2]] = np.nan
            jacobian[group[-2:]] = np.nan
    return jacobian


def integrate_branch(
    branch,
    count=1201,
    step=1.0e-4,
    maximum_time=0.6,
    source_trim=DEFAULT_SOURCE_TRIM,
):
    """Advect one source branch and retain markers only until first exit.

    Only the central ``1 - 2*source_trim`` fraction of the connected source
    arc contributes to the reported minimum.  This keeps both the arm corner
    and the accumulation--ablation transition away from the diagnostic.
    """
    if not 0.0 <= source_trim < 0.5:
        raise ValueError("source_trim must lie in [0, 0.5)")

    theta = source_theta(branch, count)
    points = OUTER_RADIUS * np.column_stack((np.cos(theta), np.sin(theta)))
    active = np.ones(count, dtype=bool)
    source_interior = np.zeros(count, dtype=bool)
    lower = int(source_trim * count)
    upper = int((1.0 - source_trim) * count)
    source_interior[lower:upper] = True

    times = [0.0]
    minimum_jacobian = [1.0]
    median_jacobian = [1.0]
    median_index = count // 2
    crossing_time = np.nan
    crossing_position = np.array((np.nan, np.nan))
    crossing_theta0 = np.nan
    median_crossing_time = np.nan
    reference_jacobian = None
    reference_active = None
    previous_minimum = 1.0
    previous_median = 1.0

    steps = int(np.ceil(maximum_time / step))
    reference_step = int(np.round(REFERENCE_TIME / step))
    for step_index in range(1, steps + 1):
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
        median = float(jacobian[median_index])
        time = step_index * step
        times.append(time)
        minimum_jacobian.append(minimum)
        median_jacobian.append(median)

        if (
            np.isnan(crossing_time)
            and np.isfinite(previous_minimum)
            and np.isfinite(minimum)
            and previous_minimum >= 0.0
            and minimum < 0.0
        ):
            fraction = previous_minimum / (previous_minimum - minimum)
            crossing_time = time - step + fraction * step
            crossing_position = points[index].copy()
            crossing_theta0 = theta[index]
        if (
            np.isnan(median_crossing_time)
            and np.isfinite(previous_median)
            and np.isfinite(median)
            and previous_median >= 0.0
            and median < 0.0
        ):
            fraction = previous_median / (previous_median - median)
            median_crossing_time = time - step + fraction * step
        if step_index == reference_step:
            reference_jacobian = jacobian.copy()
            reference_active = active.copy()

        previous_minimum = minimum
        previous_median = median

    if reference_jacobian is None:
        reference_jacobian = np.full(count, np.nan)
        reference_active = np.zeros(count, dtype=bool)

    return {
        "branch": branch,
        "theta": theta,
        "times": np.asarray(times),
        "minimum_Jx": np.asarray(minimum_jacobian),
        "median_Jx": np.asarray(median_jacobian),
        "crossing_time": crossing_time,
        "crossing_position": crossing_position,
        "crossing_theta0": crossing_theta0,
        "median_source_vertical_time": median_crossing_time,
        "reference_Jx": reference_jacobian,
        "reference_active": reference_active,
        "final_active_count": int(active.sum()),
        "step": step,
        "count": count,
        "source_trim": source_trim,
    }


def convergence_row(result, study):
    """Reduce one material-map integration to a convergence-table row."""
    away_from_arm = (
        result["reference_active"]
        & (result["theta"] >= np.radians(15.0))
        & np.isfinite(result["reference_Jx"])
    )
    away_values = result["reference_Jx"][away_from_arm]
    return {
        "study": study,
        "branch": result["branch"],
        "source_count": result["count"],
        "time_step": result["step"],
        "source_trim_fraction": result["source_trim"],
        "first_vertical_time": result["crossing_time"],
        "first_vertical_time_R2_Aminus1": (
            result["crossing_time"]
            / (LAMBDA * 2.0 ** (LAMBDA - 2.0))
        ),
        "median_source_vertical_time": result["median_source_vertical_time"],
        "minimum_Jx_to_t_0.6": np.nanmin(result["minimum_Jx"]),
        "active_at_t_0.6": result["final_active_count"],
        "crossing_x": result["crossing_position"][0],
        "crossing_z": result["crossing_position"][1],
        "crossing_source_theta_deg": np.degrees(result["crossing_theta0"]),
        "Jmin_at_R2_t0.10_theta_ge15": (
            np.min(away_values) if len(away_values) else np.nan
        ),
        "Jmax_at_R2_t0.10_theta_ge15": (
            np.max(away_values) if len(away_values) else np.nan
        ),
        "active_count_R2_t0.10_theta_ge15": len(away_values),
    }


def run_convergence():
    """Test marker/time resolution and the diagnostic source cutoff."""
    configurations = (
        (601, 2.0e-4),
        (1201, 1.0e-4),
        (2401, 5.0e-5),
    )
    rows = []
    finest = {}
    for count, step in configurations:
        for branch in ("shallow/right", "steep/left"):
            print(
                f"Tracing {branch}, {count} sources, dt={step:g}",
                flush=True,
            )
            result = integrate_branch(branch, count, step)
            rows.append(convergence_row(result, "resolution"))
            if count == configurations[-1][0]:
                finest[branch] = result

    # The first right-branch vertical tangent occurs close to the lower edge
    # of the retained source interval.  Vary that edge explicitly rather than
    # treating the central-80% convention as a physical boundary.
    for source_trim in (0.05, 0.20, 0.30):
        print(
            "Tracing shallow/right cutoff sensitivity, "
            f"trim={source_trim:.2f}",
            flush=True,
        )
        result = integrate_branch(
            "shallow/right",
            count=1201,
            step=1.0e-4,
            source_trim=source_trim,
        )
        rows.append(convergence_row(result, "source-cutoff"))
    return pd.DataFrame(rows), finest


def integrate_snapshots(branch, selected_times, count=2401, step=2.5e-4):
    """Return long-age material curves without retaining a full trajectory."""
    selected_times = tuple(float(value) for value in selected_times)
    target_steps = {int(np.round(value / step)): value for value in selected_times}
    for target_step, value in target_steps.items():
        if not np.isclose(target_step * step, value, atol=1.0e-12):
            raise ValueError("selected times must be integer multiples of step")

    theta = source_theta(branch, count)
    points = OUTER_RADIUS * np.column_stack((np.cos(theta), np.sin(theta)))
    active = np.ones(count, dtype=bool)
    snapshots = {}
    if 0 in target_steps:
        snapshots[target_steps[0]] = points.copy()

    maximum_step = max(target_steps)
    for step_index in range(1, maximum_step + 1):
        indices = np.flatnonzero(active)
        if len(indices) < 2:
            break
        points[indices] = rk4(points[indices], step)
        radius = np.linalg.norm(points, axis=1)
        current_theta = np.arctan2(points[:, 1], points[:, 0])
        active &= (
            (radius < OUTER_RADIUS - 1.0e-8)
            & (current_theta > 0.0)
            & (current_theta < ALPHA)
        )
        if step_index in target_steps:
            recorded = points.copy()
            recorded[~active] = np.nan
            snapshots[target_steps[step_index]] = recorded
    return snapshots


def snapshot_time_step_check(selected_times):
    """Return long-age curves and their RK4 time-step discrepancy."""
    fine = {}
    coarse = {}
    for branch in ("shallow/right", "steep/left"):
        fine[branch] = integrate_snapshots(
            branch, selected_times, count=2401, step=2.5e-4
        )
        coarse[branch] = integrate_snapshots(
            branch, selected_times, count=2401, step=5.0e-4
        )

    maximum_difference = 0.0
    compared_points = 0
    for branch in fine:
        for selected_time in selected_times:
            fine_curve = fine[branch][float(selected_time)]
            coarse_curve = coarse[branch][float(selected_time)]
            common = np.all(np.isfinite(fine_curve), axis=1) & np.all(
                np.isfinite(coarse_curve), axis=1
            )
            if np.any(common):
                difference = np.linalg.norm(
                    fine_curve[common] - coarse_curve[common], axis=1
                )
                maximum_difference = max(
                    maximum_difference, float(np.max(difference))
                )
                compared_points += int(common.sum())
    return fine, maximum_difference, compared_points


def draw_domain(axis, outer_arc_color="0.12"):
    """Draw the sector boundary on one axis."""
    arc_theta = np.linspace(0.0, ALPHA, 500)
    axis.plot(
        np.cos(arc_theta),
        np.sin(arc_theta),
        color=outer_arc_color,
        linewidth=1.2,
        zorder=4,
    )
    axis.plot([0, 1], [0, 0], color="0.12", linewidth=1.2, zorder=4)
    axis.plot(
        [0, np.cos(ALPHA)],
        [0, np.sin(ALPHA)],
        color="0.12",
        linewidth=1.2,
        zorder=4,
    )


def make_figure(results, snapshots, selected_times):
    """Plot the flow, long-age material curves, and overturn diagnostic."""
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.65), constrained_layout=True)

    # (a) Direction of the convergent flow and the source/outflow partition.
    x = np.linspace(-0.72, 1.0, 250)
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
    axes[0].streamplot(
        x,
        z,
        U,
        W,
        color="0.58",
        density=0.82,
        linewidth=0.75,
        arrowsize=0.75,
    )
    draw_domain(axes[0], outer_arc_color="0.65")
    source_arcs = {
        "shallow/right": np.linspace(0.0, ALPHA / 4.0, 100),
        "steep/left": np.linspace(3.0 * ALPHA / 4.0, ALPHA, 100),
    }
    for branch, angles in source_arcs.items():
        axes[0].plot(
            np.cos(angles),
            np.sin(angles),
            color=BRANCH_COLORS[branch],
            linewidth=4.0,
            solid_capstyle="butt",
            zorder=6,
            label=BRANCH_LABELS[branch] + " source",
        )
    outflow_angles = np.linspace(ALPHA / 4.0, 3.0 * ALPHA / 4.0, 180)
    axes[0].plot(
        np.cos(outflow_angles),
        np.sin(outflow_angles),
        color="0.25",
        linewidth=4.0,
        solid_capstyle="butt",
        zorder=6,
        label="ablation/outflow (no source)",
    )
    axes[0].legend(
        loc="lower left",
        fontsize=7.4,
        frameon=True,
        framealpha=0.94,
        handlelength=2.5,
    )
    axes[0].set_title("(a) Analytical convergent flow")

    # (b) Color means age; linestyle means source branch.  Ages through 3.0
    # bring both branches into the central interior.  The old plot stopped at
    # 0.4, which made that region look artificially empty.
    norm = mpl.colors.LogNorm(vmin=min(selected_times), vmax=max(selected_times))
    cmap = mpl.colormaps["viridis"]
    for selected_time in selected_times:
        color = cmap(norm(selected_time))
        for branch in ("shallow/right", "steep/left"):
            curve = snapshots[branch][float(selected_time)]
            axes[1].plot(
                curve[:, 0],
                curve[:, 1],
                color=color,
                linestyle=BRANCH_LINESTYLES[branch],
                linewidth=1.35,
                solid_capstyle="round",
            )
    draw_domain(axes[1], outer_arc_color="0.25")
    branch_handles = [
        Line2D(
            [0],
            [0],
            color="0.15",
            linestyle=BRANCH_LINESTYLES[branch],
            linewidth=1.5,
            label=BRANCH_LABELS[branch],
        )
        for branch in ("shallow/right", "steep/left")
    ]
    axes[1].legend(
        handles=branch_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.94,
        fontsize=7.8,
    )
    axes[1].annotate(
        "ages $>3$ near\nthe stagnation point",
        xy=(0.07, 0.07),
        xytext=(0.27, 0.23),
        arrowprops={"arrowstyle": "-", "color": "0.30", "lw": 0.7},
        fontsize=7.4,
        color="0.25",
        ha="center",
    )
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=axes[1],
        fraction=0.050,
        pad=0.025,
        ticks=selected_times,
    )
    colorbar.ax.set_yticklabels([f"{value:g}" for value in selected_times])
    colorbar.set_label(r"dimensionless age $tU_R/R$")
    axes[1].set_title("(b) Equal-age material curves")

    # (c) The open marker is now placed on the diagnostic itself; the old red
    # spatial marker was removed because its meaning was not self-evident.
    axes[2].axhspan(-1.35, 0.0, color="#FCE8E6", zorder=0)
    for branch, result in results.items():
        axes[2].plot(
            result["times"],
            result["minimum_Jx"],
            color=BRANCH_COLORS[branch],
            linewidth=1.8,
            label=BRANCH_LABELS[branch],
        )
    right_result = results["shallow/right"]
    crossing_time = right_result["crossing_time"]
    if np.isfinite(crossing_time):
        axes[2].scatter(
            crossing_time,
            0.0,
            s=38,
            facecolor="white",
            edgecolor=BRANCH_COLORS["shallow/right"],
            linewidth=1.3,
            zorder=5,
        )
        axes[2].annotate(
            rf"first $J_x=0$: {crossing_time:.3f}",
            xy=(crossing_time, 0.0),
            xytext=(0.40, 0.34),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "-", "color": "0.25", "lw": 0.8},
            fontsize=8,
        )
    axes[2].axhline(0.0, color="0.25", linewidth=0.9)
    axes[2].text(
        0.57,
        -1.13,
        r"$J_x<0$: overturned",
        color="#A33A2B",
        fontsize=8.5,
    )
    axes[2].text(
        0.02,
        0.02,
        "central 80% of each source arc;\ncurve endpoints excluded",
        transform=axes[2].transAxes,
        fontsize=7.4,
        color="0.28",
        va="bottom",
    )
    axes[2].set(
        xlabel=r"dimensionless age $tU_R/R$",
        ylabel=r"minimum $J_x=\partial x/\partial x_0$",
        title="(c) Overturning diagnostic",
        xlim=(0.0, 0.6),
        ylim=(-1.35, 1.65),
    )
    axes[2].legend(loc="upper left", frameon=False, fontsize=8.2)

    for axis in axes[:2]:
        axis.set(
            xlabel="$x/R$",
            ylabel="$z/R$",
            xlim=(-0.73, 1.02),
            ylim=(-0.02, 1.02),
        )
        axis.set_aspect("equal")

    png_path = OUTPUT_DIRECTORY / "analytical_convergent_flow_benchmark.png"
    pdf_path = OUTPUT_DIRECTORY / "analytical_convergent_flow_benchmark.pdf"
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    # Retain legacy aliases because the executed survey notebook currently
    # reads these paths.  New prose and links should use the canonical names.
    fig.savefig(OUTPUT_DIRECTORY / "stagnation_flow_wedge_benchmark.png", dpi=240)
    fig.savefig(OUTPUT_DIRECTORY / "stagnation_flow_wedge_benchmark.pdf")
    plt.close(fig)
    return png_path, pdf_path


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table, results = run_convergence()
    table.to_csv(
        OUTPUT_DIRECTORY / "analytical_convergent_flow_convergence.csv",
        index=False,
    )
    # Legacy alias retained for the executed survey notebook.
    table.to_csv(
        OUTPUT_DIRECTORY / "stagnation_flow_wedge_convergence.csv", index=False
    )

    selected_times = (0.05, 0.10, 0.20, 0.35, 0.55, 0.85, 1.30, 2.00, 3.00)
    snapshots, snapshot_difference, compared_points = snapshot_time_step_check(
        selected_times
    )
    figure_paths = make_figure(results, snapshots, selected_times)

    print("\nAnalytical convergent-flow material-map diagnostic")
    print(table.to_string(index=False))
    print(
        "\nLong-age curve RK4 check (dt 5e-4 versus 2.5e-4): "
        f"max position difference={snapshot_difference:.3e} R "
        f"over {compared_points} common active marker snapshots."
    )
    print(
        "The middle outer arc is ablation/outflow, not a forward-time "
        "material source; older curves through tU_R/R=3 fill the central "
        "interior."
    )
    print("Basal check: tau_t = 0 while u_t != 0 on both straight arms.")
    print("This is free slip (B=0, C=infinity), not finite linear sliding.")
    print("Saved " + " and ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
