"""Run the three-parameter steady sinusoidal-bed production sweep.

The geometry and characteristic calculation live in
``steady_characteristic_sinusoidal_parameter_map``.  This driver defines a
rectangular grid in the natural ratios ``A/H`` and ``lambda/H`` at five
dimensional mean thicknesses, writes restartable checkpoints, and keeps the
Firedrake workers isolated with the spawn start method.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from steady_characteristic_sinusoidal_parameter_map import (
    BASAL_DRAG,
    DIAGNOSTIC_METHOD_VERSION,
    NITSCHE_PENALTY,
    SURFACE_SLOPE,
    VISCOSITY,
    SinusoidalGeometry,
    solve_parameter_case,
)


MEAN_THICKNESSES = np.array((50.0, 100.0, 160.0, 240.0, 320.0))
AMPLITUDE_RATIOS = np.array(
    (0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45,
     0.55, 0.65, 0.75, 0.85)
)
WAVELENGTH_RATIOS = np.array(
    (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
     10.0, 12.0)
)

PRODUCTION_MESH_NZ = 24
PRODUCTION_SOURCE_COUNT = 241
PRODUCTION_RETAINED_FLUX_FRACTION = 0.80
PRODUCTION_TIME_STEP = 25.0
PRODUCTION_MAXIMUM_AGE = 300_000.0
PRODUCTION_SAMPLER_NZ = 121


def mesh_nx_for_ratio(wavelength_ratio):
    """Resolve one wave with 64--96 columns over the sampled ratio range."""
    requested = max(64.0, 8.0 * float(wavelength_ratio))
    return int(8 * np.ceil(requested / 8.0))


def sampler_nx_for_mesh(mesh_nx):
    """Keep at least four sampler intervals per Stokes column."""
    return int(max(321, 4 * int(mesh_nx) + 1))


def task_tuple(mean_thickness, amplitude_ratio, wavelength_ratio, maximum_age):
    mesh_nx = mesh_nx_for_ratio(wavelength_ratio)
    return (
        float(mean_thickness),
        float(amplitude_ratio),
        float(wavelength_ratio),
        mesh_nx,
        PRODUCTION_MESH_NZ,
        sampler_nx_for_mesh(mesh_nx),
        PRODUCTION_SAMPLER_NZ,
        PRODUCTION_SOURCE_COUNT,
        PRODUCTION_RETAINED_FLUX_FRACTION,
        PRODUCTION_TIME_STEP,
        float(maximum_age),
    )


def solve_task(task):
    (
        mean_thickness,
        amplitude_ratio,
        wavelength_ratio,
        mesh_nx,
        mesh_nz,
        sampler_nx,
        sampler_nz,
        source_count,
        retained_flux_fraction,
        time_step,
        maximum_age,
    ) = task
    geometry = SinusoidalGeometry(
        wavelength=wavelength_ratio * mean_thickness,
        amplitude=amplitude_ratio * mean_thickness,
        mean_thickness=mean_thickness,
    )
    row = solve_parameter_case(
        geometry,
        mesh_nx=mesh_nx,
        mesh_nz=mesh_nz,
        sampler_nx=sampler_nx,
        sampler_nz=sampler_nz,
        source_count=source_count,
        retained_flux_fraction=retained_flux_fraction,
        time_step=time_step,
        maximum_age=maximum_age,
    )
    row["sliding thickness ratio beta H / eta"] = mean_thickness / 2500.0
    return row


def task_key_from_row(row):
    """Identity of a completed case, excluding its unused safety ceiling.

    A successful trajectory is independent of how much unused integration
    time remained after its physical exit.  Excluding the ceiling lets a
    longer-cap restart reuse every completed row while retrying only censored
    cases.
    """
    return (
        round(float(row["mean ice thickness (m)"]), 12),
        round(float(row["amplitude / mean thickness"]), 12),
        round(float(row["wavelength / mean thickness"]), 12),
        int(row["mesh nx"]),
        int(row["mesh nz"]),
        int(row["sampler nx"]),
        int(row["sampler nz"]),
        int(row["candidate source count"]),
        round(float(row["retained accumulation-flux fraction"]), 12),
        round(float(row["characteristic time step (yr)"]), 12),
    )


def task_key_from_task(task):
    """Match :func:`task_key_from_row` for an unsolved task tuple."""
    key = []
    for value in task[:-1]:
        if isinstance(value, (float, np.floating)):
            key.append(round(float(value), 12))
        else:
            key.append(value)
    return tuple(key)


def sorted_table(rows):
    return pd.DataFrame(rows).sort_values(
        [
            "mean ice thickness (m)",
            "amplitude / mean thickness",
            "wavelength / mean thickness",
        ]
    ).reset_index(drop=True)


def write_metadata(table, output):
    """Write a compact machine-readable record of grid, method, and result."""
    numerical = table.loc[table["status"].eq("success")].copy()
    status_counts = {
        str(key): int(value)
        for key, value in table["status"].value_counts(dropna=False).items()
    }

    def selected_row(column):
        if numerical.empty or not numerical[column].notna().any():
            return None
        row = numerical.loc[numerical[column].idxmin()]
        return {
            "mean thickness (m)": float(row["mean ice thickness (m)"]),
            "amplitude / mean thickness": float(
                row["amplitude / mean thickness"]
            ),
            "wavelength / mean thickness": float(
                row["wavelength / mean thickness"]
            ),
            column: float(row[column]),
        }

    comparable = numerical.pivot_table(
        index=(
            "amplitude / mean thickness",
            "wavelength / mean thickness",
        ),
        columns="mean ice thickness (m)",
        values="minimum J",
        aggfunc="first",
    )
    thickness_spread = comparable.max(axis=1) - comparable.min(axis=1)
    metadata = {
        "diagnostic method version": DIAGNOSTIC_METHOD_VERSION,
        "geometry": "b(x) = -H + A cos(2 pi x / lambda), 0 <= x < lambda",
        "domain": "one exactly periodic wavelength with a flat surface",
        "dimensionless controls": [
            "A/H",
            "lambda/H",
            "beta H / eta",
        ],
        "physical constants": {
            "viscosity (Pa yr)": VISCOSITY,
            "basal drag (Pa yr/m)": BASAL_DRAG,
            "linear-slip length eta/beta (m)": VISCOSITY / BASAL_DRAG,
            "surface slope (rad)": SURFACE_SLOPE,
            "Nitsche penalty": NITSCHE_PENALTY,
        },
        "production grid": {
            "mean thicknesses (m)": MEAN_THICKNESSES.tolist(),
            "amplitude / mean thickness": AMPLITUDE_RATIOS.tolist(),
            "wavelength / mean thickness": WAVELENGTH_RATIOS.tolist(),
            "cell count": int(len(table)),
            "mesh nz": PRODUCTION_MESH_NZ,
            "mesh nx rule": "max(64, ceil(8 lambda/H)), rounded to 8",
            "candidate source count": PRODUCTION_SOURCE_COUNT,
            "retained accumulation-flux fraction": (
                PRODUCTION_RETAINED_FLUX_FRACTION
            ),
            "characteristic time step (yr)": PRODUCTION_TIME_STEP,
            "safety maximum ages present (yr)": sorted(
                float(value)
                for value in table["safety maximum age (yr)"].dropna().unique()
            ),
        },
        "source and metric convention": {
            "source depth (m)": 0.0,
            "sources": "all surface accumulation intervals, flux trimmed",
            "primary": (
                "minimum T_x, excluding only the upper 5% of local thickness "
                "beneath an ablating surface"
            ),
            "raw audit": "minimum T_x all the way to individual physical exit",
        },
        "result summary": {
            "status counts": status_counts,
            "lowest primary J": selected_row("minimum J"),
            "lowest raw J": selected_row("raw minimum J to exit"),
            "successful primary J < 0 count": int(
                (numerical["minimum J"] < 0.0).sum()
            ),
            "successful raw J < 0 count": int(
                (numerical["raw minimum J to exit"] < 0.0).sum()
            ),
            "Eulerian-domain counterflow case count": int(
                (numerical["minimum sampled u (m/yr)"] < 0.0).sum()
            ),
            "retained-path counterflow case count": int(
                (numerical["minimum path u (m/yr)"] < 0.0).sum()
            ),
            "retained-path velocity-reversal case count": int(
                (numerical["maximum velocity reversal count"] > 0).sum()
            ),
            "maximum fixed-geometry J spread across sampled H": (
                float(thickness_spread.max())
                if len(thickness_spread)
                else None
            ),
        },
    }
    metadata_output = output.with_name(output.stem + "_metadata.json")
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_output


def run_grid(output, jobs=1, maximum_age=PRODUCTION_MAXIMUM_AGE):
    reusable = {}
    if output.exists():
        existing = pd.read_csv(output)
        required = {
            "diagnostic method version",
            "minimum J",
            "classification",
            "ablating-surface buffer thickness fraction",
        }
        if required.issubset(existing.columns):
            for _, row in existing.iterrows():
                status = str(row["status"])
                current_method = (
                    row["diagnostic method version"]
                    == DIAGNOSTIC_METHOD_VERSION
                )
                reusable_status = status == "success" or status.startswith(
                    "analytic flat-bed limit"
                )
                if current_method and reusable_status:
                    reusable[task_key_from_row(row)] = row.to_dict()

    rows = []
    pending = []
    for mean_thickness in MEAN_THICKNESSES:
        for amplitude_ratio in AMPLITUDE_RATIOS:
            for wavelength_ratio in WAVELENGTH_RATIOS:
                task = task_tuple(
                    mean_thickness,
                    amplitude_ratio,
                    wavelength_ratio,
                    maximum_age,
                )
                key = task_key_from_task(task)
                if key in reusable:
                    rows.append(reusable[key])
                else:
                    pending.append(task)
    print(
        f"Sinusoidal grid: {len(rows) + len(pending)} cells, "
        f"{len(rows)} reused, {len(pending)} pending.",
        flush=True,
    )

    if jobs == 1:
        solved = map(solve_task, pending)
        pool = None
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            processes=jobs,
            maxtasksperchild=30,
        )
        solved = pool.imap(solve_task, pending, chunksize=1)
    try:
        for number, row in enumerate(solved, start=1):
            rows.append(row)
            print(
                json.dumps(
                    {
                        "completed": f"{number}/{len(pending)}",
                        "H": row["mean ice thickness (m)"],
                        "A/H": row["amplitude / mean thickness"],
                        "lambda/H": row["wavelength / mean thickness"],
                        "status": row["status"],
                        "minimum J": row.get("minimum J"),
                    }
                ),
                flush=True,
            )
            if number % 5 == 0:
                sorted_table(rows).to_csv(output, index=False, na_rep="NA")
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()
    result = sorted_table(rows)
    result.to_csv(output, index=False, na_rep="NA")
    return result


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--maximum-age", type=float, default=PRODUCTION_MAXIMUM_AGE
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_sinusoidal_parameter_map.csv",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = run_grid(args.output, jobs=args.jobs, maximum_age=args.maximum_age)
    metadata_output = write_metadata(table, args.output)
    print(table["status"].value_counts(dropna=False).to_string())
    print(f"Wrote {metadata_output}")


if __name__ == "__main__":
    main()
