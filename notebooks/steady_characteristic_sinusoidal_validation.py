"""Focused numerical audits for the steady sinusoidal-bed diagnostic.

This script is intentionally separate from the production sweep.  It first
selects the lowest-``J`` *successful* production geometry, then changes one
numerical choice at a time around that reference calculation.  A small regime
survey checks that the same production settings also behave sensibly near the
flat-bed, short-wave, long-wave, and thin-neck limits.

All characteristic sources lie exactly on the surface (``START_DEPTH=0``), as
in the production calculation.  The table is restartable: completed scenarios
with matching settings are reused from the output CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from steady_characteristic_sinusoidal_parameter_map import (
    DIAGNOSTIC_METHOD_VERSION,
    START_DEPTH,
    SinusoidalGeometry,
    solve_parameter_case,
)
from steady_characteristic_sinusoidal_parameter_sweep import (
    AMPLITUDE_RATIOS,
    MEAN_THICKNESSES,
    PRODUCTION_MAXIMUM_AGE,
    PRODUCTION_MESH_NZ,
    PRODUCTION_RETAINED_FLUX_FRACTION,
    PRODUCTION_SAMPLER_NZ,
    PRODUCTION_SOURCE_COUNT,
    PRODUCTION_TIME_STEP,
    WAVELENGTH_RATIOS,
    mesh_nx_for_ratio,
    sampler_nx_for_mesh,
)


def production_settings(wavelength_ratio):
    """Copy the production resolution, including wavelength-dependent nx."""
    mesh_nx = mesh_nx_for_ratio(wavelength_ratio)
    return {
        "mesh_nx": mesh_nx,
        "mesh_nz": PRODUCTION_MESH_NZ,
        "sampler_nx": sampler_nx_for_mesh(mesh_nx),
        "sampler_nz": PRODUCTION_SAMPLER_NZ,
        "source_count": PRODUCTION_SOURCE_COUNT,
        "retained_flux_fraction": PRODUCTION_RETAINED_FLUX_FRACTION,
        "time_step": PRODUCTION_TIME_STEP,
        "maximum_age": PRODUCTION_MAXIMUM_AGE,
    }

# These cases span qualitatively different limits but are not intended as a
# second parameter sweep.  H=160 m is the central production thickness.
REGIME_GEOMETRIES = {
    "formal flat-bed limit": (160.0, 0.00, 4.0),
    "small amplitude": (160.0, 0.05, 4.0),
    "short-wave map edge": (160.0, 0.65, 1.0),
    "short-wave extension": (160.0, 0.65, 0.5),
    "long-wave map edge": (160.0, 0.65, 12.0),
    "long-wave extension": (160.0, 0.65, 24.0),
    "thin-neck production edge": (160.0, 0.85, 3.0),
    "thin-neck extension A/H=0.90": (160.0, 0.90, 3.0),
    "thin-neck extension A/H=0.95": (160.0, 0.95, 3.0),
}


def _geometry(mean_thickness, amplitude_ratio, wavelength_ratio, phase=0.0):
    return SinusoidalGeometry(
        wavelength=float(wavelength_ratio * mean_thickness),
        amplitude=float(amplitude_ratio * mean_thickness),
        mean_thickness=float(mean_thickness),
        phase_origin=float(phase * wavelength_ratio * mean_thickness),
    )


def select_focus_geometry(production_table):
    """Return the lowest-J uncensored production geometry."""
    production = pd.read_csv(production_table)
    required = {
        "diagnostic method version",
        "status",
        "minimum J",
        "mean ice thickness (m)",
        "amplitude / mean thickness",
        "wavelength / mean thickness",
    }
    missing = required.difference(production.columns)
    if missing:
        raise ValueError(
            f"production table lacks required columns: {sorted(missing)}"
        )
    expected_count = (
        len(MEAN_THICKNESSES)
        * len(AMPLITUDE_RATIOS)
        * len(WAVELENGTH_RATIOS)
    )
    current = production.loc[
        production["diagnostic method version"].eq(
            DIAGNOSTIC_METHOD_VERSION
        )
    ]
    if len(current) != expected_count:
        raise ValueError(
            "production sweep is incomplete for diagnostic method "
            f"{DIAGNOSTIC_METHOD_VERSION!r}: found {len(current)} of "
            f"{expected_count} rows"
        )
    successful = current.loc[
        current["status"].eq("success")
        & current["minimum J"].notna()
        & current["amplitude / mean thickness"].gt(0.0)
    ]
    if successful.empty:
        raise ValueError("production table contains no successful numerical case")
    selected = successful.loc[successful["minimum J"].idxmin()]
    return (
        float(selected["mean ice thickness (m)"]),
        float(selected["amplitude / mean thickness"]),
        float(selected["wavelength / mean thickness"]),
        float(selected["minimum J"]),
        float(selected["safety maximum age (yr)"]),
    )


def scenario_table(focus):
    """Define one-at-a-time numerical variations and the regime survey."""
    (
        focus_h,
        focus_epsilon,
        focus_lambda,
        production_minimum,
        focus_maximum_age,
    ) = focus
    scenarios = []

    def add(name, family, h, epsilon, wavelength_ratio, **changes):
        phase_fraction = changes.pop("phase_fraction", 0.0)
        settings = production_settings(wavelength_ratio)
        if family == "focus convergence":
            settings["maximum_age"] = focus_maximum_age
        settings |= changes
        scenarios.append(
            {
                "scenario": name,
                "audit family": family,
                "H": h,
                "A/H": epsilon,
                "lambda/H": wavelength_ratio,
                "phase fraction": phase_fraction,
                "production focus minimum J": (
                    production_minimum
                    if family == "focus convergence"
                    else np.nan
                ),
                **settings,
            }
        )

    add(
        "reference",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
    )
    reference_mesh_nx = mesh_nx_for_ratio(focus_lambda)
    add(
        "coarser Stokes mesh",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        mesh_nx=max(48, int(8 * np.ceil(0.75 * reference_mesh_nx / 8))),
        mesh_nz=18,
    )
    add(
        "finer Stokes mesh",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        mesh_nx=int(8 * np.ceil(1.5 * reference_mesh_nx / 8)),
        mesh_nz=36,
    )
    add(
        "larger trajectory step",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        time_step=50.0,
    )
    add(
        "smaller trajectory step",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        time_step=12.5,
    )
    add(
        "coarser velocity sampler",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        sampler_nx=241,
        sampler_nz=81,
    )
    add(
        "finer velocity sampler",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        sampler_nx=481,
        sampler_nz=181,
    )
    add(
        "fewer source candidates",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        source_count=121,
    )
    add(
        "more source candidates",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        source_count=481,
    )
    add(
        "central 70 percent of accumulation flux",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        retained_flux_fraction=0.70,
    )
    add(
        "central 90 percent of accumulation flux",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        retained_flux_fraction=0.90,
    )
    add(
        "quarter-wave phase translation",
        "focus convergence",
        focus_h,
        focus_epsilon,
        focus_lambda,
        phase_fraction=0.25,
    )

    for label, (h, epsilon, wavelength_ratio) in REGIME_GEOMETRIES.items():
        add(label, "regime survey", h, epsilon, wavelength_ratio)
    add(
        "thin-neck extension A/H=0.95, finer mesh",
        "regime survey",
        160.0,
        0.95,
        3.0,
        mesh_nx=96,
        mesh_nz=36,
    )
    return pd.DataFrame(scenarios)


def scenario_key(row):
    columns = (
        "diagnostic method version",
        "scenario",
        "audit family",
        "mean ice thickness (m)",
        "amplitude / mean thickness",
        "wavelength / mean thickness",
        "bed phase origin (m; coordinate gauge)",
        "mesh nx",
        "mesh nz",
        "sampler nx",
        "sampler nz",
        "candidate source count",
        "retained accumulation-flux fraction",
        "characteristic time step (yr)",
        "safety maximum age (yr)",
    )
    values = []
    for column in columns:
        value = row[column]
        values.append(value if isinstance(value, str) else float(value))
    return tuple(values)


def annotate_result(result, specification, origin):
    """Attach audit labels to either a new solve or a production row."""
    result = dict(result)
    result.update(
        {
            "scenario": specification["scenario"],
            "audit family": specification["audit family"],
            "phase translation / wavelength": specification[
                "phase fraction"
            ],
            "production focus minimum J": specification[
                "production focus minimum J"
            ],
            "exact-surface sources": bool(START_DEPTH == 0.0),
            "audit result origin": origin,
        }
    )
    return result


def solve_scenario(specification):
    phase = specification["phase fraction"]
    geometry = _geometry(
        specification["H"],
        specification["A/H"],
        specification["lambda/H"],
        phase=phase,
    )
    result = solve_parameter_case(
        geometry,
        mesh_nx=int(specification["mesh_nx"]),
        mesh_nz=int(specification["mesh_nz"]),
        sampler_nx=int(specification["sampler_nx"]),
        sampler_nz=int(specification["sampler_nz"]),
        source_count=int(specification["source_count"]),
        retained_flux_fraction=float(
            specification["retained_flux_fraction"]
        ),
        time_step=float(specification["time_step"]),
        maximum_age=float(specification["maximum_age"]),
    )
    return annotate_result(result, specification, "independent audit solve")


def matching_production_result(production, specification):
    """Reuse exact production calculations for unmodified regime references."""
    geometry = _geometry(
        specification["H"],
        specification["A/H"],
        specification["lambda/H"],
        specification["phase fraction"],
    )
    requested = {
        "mean ice thickness (m)": geometry.mean_thickness,
        "amplitude / mean thickness": geometry.amplitude_ratio,
        "wavelength / mean thickness": geometry.wavelength_ratio,
        "bed phase origin (m; coordinate gauge)": geometry.phase_origin,
        "mesh nx": specification["mesh_nx"],
        "mesh nz": specification["mesh_nz"],
        "sampler nx": specification["sampler_nx"],
        "sampler nz": specification["sampler_nz"],
        "candidate source count": specification["source_count"],
        "retained accumulation-flux fraction": specification[
            "retained_flux_fraction"
        ],
        "characteristic time step (yr)": specification["time_step"],
        "safety maximum age (yr)": specification["maximum_age"],
    }
    mask = production["diagnostic method version"].eq(
        DIAGNOSTIC_METHOD_VERSION
    )
    for column, value in requested.items():
        mask &= np.isclose(production[column].astype(float), float(value))
    matches = production.loc[mask]
    if len(matches) != 1:
        return None
    return annotate_result(
        matches.iloc[0].to_dict(),
        specification,
        "reused exact production row",
    )


def add_reference_differences(table):
    """Add signed changes from the focus reference for concise interpretation."""
    focus = table.loc[
        table["audit family"].eq("focus convergence")
        & table["scenario"].eq("reference")
    ]
    if len(focus) != 1:
        return table
    reference = focus.iloc[0]
    if {
        "source phase at minimum J",
        "phase translation / wavelength",
    }.issubset(table.columns):
        table["source phase relative to bed at minimum J"] = np.mod(
            table["source phase at minimum J"]
            - table["phase translation / wavelength"],
            1.0,
        )
        reference_relative_phase = float(
            np.mod(
                reference["source phase at minimum J"]
                - reference["phase translation / wavelength"],
                1.0,
            )
        )
        table["wrapped change in bed-relative minimum-J source phase"] = (
            np.mod(
                table["source phase relative to bed at minimum J"]
                - reference_relative_phase
                + 0.5,
                1.0,
            )
            - 0.5
        )
    for column, output in (
        ("minimum J", "change in minimum J from reference"),
        (
            "raw minimum J to exit",
            "change in raw minimum J from reference",
        ),
    ):
        if column in table and pd.notna(reference.get(column)):
            table[output] = table[column] - float(reference[column])
    return table


def run(production_table, output):
    focus = select_focus_geometry(production_table)
    production = pd.read_csv(production_table)
    specifications = scenario_table(focus)
    existing = {}
    if output.exists():
        previous = pd.read_csv(output)
        for _, row in previous.iterrows():
            try:
                existing[scenario_key(row)] = row.to_dict()
            except (KeyError, TypeError, ValueError):
                pass

    rows = []
    for number, specification in specifications.iterrows():
        geometry = _geometry(
            specification["H"],
            specification["A/H"],
            specification["lambda/H"],
            specification["phase fraction"],
        )
        prospective = {
            "diagnostic method version": DIAGNOSTIC_METHOD_VERSION,
            "scenario": specification["scenario"],
            "audit family": specification["audit family"],
            "mean ice thickness (m)": geometry.mean_thickness,
            "amplitude / mean thickness": geometry.amplitude_ratio,
            "wavelength / mean thickness": geometry.wavelength_ratio,
            "bed phase origin (m; coordinate gauge)": geometry.phase_origin,
            "mesh nx": specification["mesh_nx"],
            "mesh nz": specification["mesh_nz"],
            "sampler nx": specification["sampler_nx"],
            "sampler nz": specification["sampler_nz"],
            "candidate source count": specification["source_count"],
            "retained accumulation-flux fraction": specification[
                "retained_flux_fraction"
            ],
            "characteristic time step (yr)": specification["time_step"],
            "safety maximum age (yr)": specification["maximum_age"],
        }
        key = scenario_key(prospective)
        if key in existing:
            result = existing[key]
            action = "reused"
        else:
            result = matching_production_result(production, specification)
            if result is not None:
                action = "reused exact production row"
            else:
                print(
                    f"[{number + 1}/{len(specifications)}] "
                    f"{specification['scenario']}",
                    flush=True,
                )
                result = solve_scenario(specification)
                action = "solved"
        rows.append(result)
        checkpoint = add_reference_differences(pd.DataFrame(rows))
        checkpoint.to_csv(output, index=False, na_rep="NA")
        print(
            f"  {action}: status={result.get('status')}, "
            f"minimum J={result.get('minimum J')}",
            flush=True,
        )
    table = add_reference_differences(pd.DataFrame(rows))
    table.to_csv(output, index=False, na_rep="NA")
    return table


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-table",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_sinusoidal_parameter_map.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "data"
        / "steady_characteristic_sinusoidal_validation.csv",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    table = run(arguments.production_table, arguments.output)
    print(
        table[
            [
                "scenario",
                "status",
                "minimum J",
                "raw minimum J to exit",
                "change in minimum J from reference",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
