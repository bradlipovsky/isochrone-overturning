"""Steady-characteristic sweep for a smooth localized periodic bed feature.

The bed combines one broad sinusoidal trough with an independently narrow,
rounded periodic riegel (or a rounded notch when ``polarity=-1``):

    b(x) = -H + A f(theta),
    f = [cos(theta) + p r (V_kappa(theta-phi) - mean(V_kappa))] / M,
    V_kappa(delta) = exp[kappa (cos(delta) - 1)].

Here ``M`` is the positive maximum of the bracketed expression.  Therefore
``H`` is the mean thickness, ``H-A`` is the minimum thickness, and the
unsubtracted localized feature has the requested physical full width at half
maximum ``ell``.  The broad period ``lambda`` and narrow width ``ell`` are
independent; unlike a two-harmonic bed, their ratio is not fixed.

The Stokes equations, flat surface, constant downslope body force, linear
sliding law, surface-source selection, and direct steady ``J`` and ``C``
diagnostics are imported unchanged from the one-mode calculation.  This is a
compact pilot rather than a complete four-dimensional tensor product:

``shape``
    Sweep width, feature phase, and polarity at fixed amplitude and period.

``selected``
    Sweep minimum gap (through ``A/H``) and broad period for one selected
    localized shape.

Both modes are restartable.  At most two worker processes are permitted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cached_property
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import i0e

import steady_characteristic_sinusoidal_parameter_map as core


METHOD_VERSION = "localized-von-mises-riegel-v1__" + core.DIAGNOSTIC_METHOD_VERSION

MEAN_THICKNESS = 160.0
SHAPE_AMPLITUDE_RATIO = 0.90
SHAPE_PERIOD_RATIO = 4.0
WIDTH_RATIOS = np.array((1.0 / 12.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0))
PHASE_FRACTIONS = np.array((0.0, 0.25, 0.5, 0.75))
POLARITIES = np.array((-1, 1), dtype=int)

AMPLITUDE_RATIOS = np.array((0.65, 0.80, 0.90, 0.95))
PERIOD_RATIOS = np.array((0.5, 1.0, 2.0, 3.0, 4.0, 6.0))

DEFAULT_SELECTED_WIDTH_RATIO = 1.0 / 12.0
DEFAULT_SELECTED_PHASE_FRACTION = 0.25
DEFAULT_SELECTED_POLARITY = 1
DEFAULT_FEATURE_STRENGTH = 1.0

RETAINED_FLUX_FRACTION = 0.80
SOURCE_COUNT = 241
MAXIMUM_AGE = 500_000.0
MINIMUM_SURFACE_SIGNAL_TO_BASAL_LEAKAGE = 10.0


def round_up(value, multiple):
    return int(multiple * np.ceil(float(value) / multiple))


@dataclass(frozen=True)
class LocalizedPacketGeometry:
    """One smooth, exactly periodic broad trough plus rounded local feature."""

    wavelength: float
    amplitude: float
    mean_thickness: float
    feature_width: float
    feature_phase: float
    feature_polarity: int
    feature_strength: float = DEFAULT_FEATURE_STRENGTH
    phase_origin: float = 0.0

    def __post_init__(self):
        values = (
            self.wavelength,
            self.amplitude,
            self.mean_thickness,
            self.feature_width,
            self.feature_phase,
            self.feature_strength,
            self.phase_origin,
        )
        if not all(np.isfinite(values)):
            raise ValueError("geometry values must be finite")
        if self.wavelength <= 0.0 or self.feature_width <= 0.0:
            raise ValueError("period and localized-feature width must be positive")
        if self.feature_width >= self.wavelength:
            raise ValueError("localized-feature FWHM must be less than one period")
        if self.amplitude < 0.0 or self.mean_thickness <= self.amplitude:
            raise ValueError("require 0 <= amplitude < mean thickness")
        if int(self.feature_polarity) not in (-1, 1):
            raise ValueError("feature polarity must be -1 or +1")
        if self.feature_strength < 0.0:
            raise ValueError("feature strength must be nonnegative")

    @property
    def amplitude_ratio(self):
        return self.amplitude / self.mean_thickness

    @property
    def wavelength_ratio(self):
        return self.wavelength / self.mean_thickness

    @property
    def feature_width_ratio(self):
        return self.feature_width / self.mean_thickness

    @property
    def feature_width_per_period(self):
        return self.feature_width / self.wavelength

    @cached_property
    def kappa(self):
        # V(delta)=1/2 at |delta|=pi*ell/lambda.
        half_width_phase = np.pi * self.feature_width / self.wavelength
        return float(np.log(2.0) / (1.0 - np.cos(half_width_phase)))

    @cached_property
    def von_mises_mean(self):
        # exp(-kappa) I_0(kappa), evaluated without overflow.
        return float(i0e(self.kappa))

    def phase(self, x):
        return (
            2.0
            * np.pi
            * (np.asarray(x, dtype=float) - self.phase_origin)
            / self.wavelength
        )

    def localized_feature(self, phase):
        delta = np.asarray(phase, dtype=float) - self.feature_phase
        return np.exp(self.kappa * (np.cos(delta) - 1.0))

    def raw_profile_from_phase(self, phase):
        phase = np.asarray(phase, dtype=float)
        local = self.localized_feature(phase) - self.von_mises_mean
        return np.cos(phase) + (
            self.feature_polarity * self.feature_strength * local
        )

    @cached_property
    def _profile_audit(self):
        # Even the narrowest production feature has >300 samples across its
        # FWHM.  The exact maximum is then polished by a local parabolic fit.
        phase = np.linspace(0.0, 2.0 * np.pi, 32768, endpoint=False)
        values = self.raw_profile_from_phase(phase)
        maximum_index = int(np.argmax(values))
        minimum_index = int(np.argmin(values))

        def polished(index):
            left = values[(index - 1) % len(values)]
            center = values[index]
            right = values[(index + 1) % len(values)]
            denominator = left - 2.0 * center + right
            if denominator == 0.0:
                return float(center)
            offset = 0.5 * (left - right) / denominator
            return float(center - 0.25 * (left - right) * offset)

        return polished(maximum_index), polished(minimum_index)

    @property
    def normalization_maximum(self):
        return self._profile_audit[0]

    @property
    def normalized_minimum(self):
        return self._profile_audit[1] / self.normalization_maximum

    def normalized_bed_profile(self, x):
        return self.raw_profile_from_phase(self.phase(x)) / self.normalization_maximum

    def thickness(self, x):
        return self.mean_thickness - self.amplitude * self.normalized_bed_profile(x)

    def thickness_slope(self, x):
        phase = self.phase(x)
        delta = phase - self.feature_phase
        local = self.localized_feature(phase)
        raw_phase_derivative = -np.sin(phase) - (
            self.feature_polarity
            * self.feature_strength
            * self.kappa
            * np.sin(delta)
            * local
        )
        profile_slope = (
            2.0 * np.pi / self.wavelength
        ) * raw_phase_derivative / self.normalization_maximum
        return -self.amplitude * profile_slope

    def bed_height(self, x):
        return -self.thickness(x)

    @property
    def minimum_thickness(self):
        return self.mean_thickness - self.amplitude

    @property
    def maximum_thickness(self):
        return self.mean_thickness - self.amplitude * self.normalized_minimum

    @property
    def relief(self):
        return self.amplitude * (1.0 - self.normalized_minimum)

    @cached_property
    def peak_bed_slope(self):
        phase = np.linspace(0.0, 2.0 * np.pi, 32768, endpoint=False)
        delta = phase - self.feature_phase
        local = self.localized_feature(phase)
        raw_phase_derivative = -np.sin(phase) - (
            self.feature_polarity
            * self.feature_strength
            * self.kappa
            * np.sin(delta)
            * local
        )
        return float(
            self.amplitude
            * 2.0
            * np.pi
            / self.wavelength
            * np.max(np.abs(raw_phase_derivative))
            / self.normalization_maximum
        )


def geometry_diagnostics(geometry):
    return {
        "localized feature FWHM (m)": geometry.feature_width,
        "localized feature FWHM / mean thickness": geometry.feature_width_ratio,
        "localized feature FWHM / broad period": geometry.feature_width_per_period,
        "localized feature phase (rad)": geometry.feature_phase,
        "localized feature phase / (2 pi)": float(
            np.mod(geometry.feature_phase, 2.0 * np.pi) / (2.0 * np.pi)
        ),
        "localized feature polarity": int(geometry.feature_polarity),
        "localized / broad raw amplitude": geometry.feature_strength,
        "localized von Mises kappa": geometry.kappa,
        "localized von Mises mean": geometry.von_mises_mean,
        "localized normalization maximum": geometry.normalization_maximum,
        "normalized bed-profile minimum": geometry.normalized_minimum,
        "bed peak-to-trough relief / mean thickness": (
            geometry.relief / geometry.mean_thickness
        ),
        "narrow-feature screening chi at minimum gap": (
            2.0 * np.pi * geometry.minimum_thickness / geometry.feature_width
        ),
    }


def mesh_nx_for_geometry(geometry):
    # Eight Q2 horizontal elements across the feature FWHM is the pilot
    # resolution; focused minima are repeated below at twelve.
    return round_up(max(96.0, 8.0 * geometry.wavelength / geometry.feature_width), 8)


def mesh_nz_for_geometry(geometry):
    vertical_scale_ratio = min(geometry.feature_width_ratio, geometry.wavelength_ratio)
    requested = max(48.0, 6.0 * np.sqrt(8.0 * np.pi / vertical_scale_ratio))
    if geometry.amplitude_ratio >= 0.90:
        requested = max(requested, 96.0)
    return round_up(requested, 8)


def vertical_grading_power_for_geometry(geometry):
    return 2.0 if geometry.feature_width_ratio <= 0.25 + 1.0e-12 else 1.0


def task_for_geometry(geometry, experiment, maximum_age=MAXIMUM_AGE, refined=False):
    mesh_nx = mesh_nx_for_geometry(geometry)
    mesh_nz = mesh_nz_for_geometry(geometry)
    grading = vertical_grading_power_for_geometry(geometry)
    thin = geometry.amplitude_ratio >= 0.90
    task = {
        "experiment": experiment,
        "wavelength": float(geometry.wavelength),
        "amplitude": float(geometry.amplitude),
        "mean_thickness": float(geometry.mean_thickness),
        "feature_width": float(geometry.feature_width),
        "feature_phase": float(geometry.feature_phase),
        "feature_polarity": int(geometry.feature_polarity),
        "feature_strength": float(geometry.feature_strength),
        "phase_origin": float(geometry.phase_origin),
        "mesh_nx": mesh_nx,
        "mesh_nz": mesh_nz,
        "mesh_vertical_grading_power": grading,
        "sampler_nx": max(513, 4 * mesh_nx + 1),
        "sampler_nz": max(257, 2 * mesh_nz + 1),
        "sampler_vertical_grading_power": grading,
        "source_count": 321 if thin else SOURCE_COUNT,
        "retained_flux_fraction": RETAINED_FLUX_FRACTION,
        "time_step": 12.5 if thin else 25.0,
        "maximum_age": float(max(maximum_age, 1_500_000.0) if thin else maximum_age),
        "resolution": "production",
    }
    if refined:
        task.update(
            mesh_nx=round_up(max(192.0, 12.0 * geometry.wavelength / geometry.feature_width), 8),
            mesh_nz=max(144, mesh_nz),
            sampler_nx=max(769, 6 * mesh_nx + 1),
            sampler_nz=max(481, 3 * mesh_nz + 1),
            source_count=481,
            time_step=6.25,
            maximum_age=max(1_500_000.0, float(task["maximum_age"])),
            resolution="refined",
        )
    return task


def geometry_from_task(task):
    return LocalizedPacketGeometry(
        wavelength=task["wavelength"],
        amplitude=task["amplitude"],
        mean_thickness=task["mean_thickness"],
        feature_width=task["feature_width"],
        feature_phase=task["feature_phase"],
        feature_polarity=task["feature_polarity"],
        feature_strength=task["feature_strength"],
        phase_origin=task["phase_origin"],
    )


def solve_task(task):
    geometry = geometry_from_task(task)
    result = core.solve_parameter_case(
        geometry,
        mesh_nx=task["mesh_nx"],
        mesh_nz=task["mesh_nz"],
        vertical_grading_power=task["mesh_vertical_grading_power"],
        sampler_nx=task["sampler_nx"],
        sampler_nz=task["sampler_nz"],
        sampler_vertical_grading_power=task["sampler_vertical_grading_power"],
        source_count=task["source_count"],
        retained_flux_fraction=task["retained_flux_fraction"],
        time_step=task["time_step"],
        maximum_age=task["maximum_age"],
        minimum_surface_signal_to_basal_leakage=MINIMUM_SURFACE_SIGNAL_TO_BASAL_LEAKAGE,
    )
    result.update(geometry_diagnostics(geometry))
    result["localized-packet method version"] = METHOD_VERSION
    result["localized-packet experiment"] = task["experiment"]
    result["localized-packet resolution"] = task["resolution"]
    return result


def shape_tasks(args):
    tasks = []
    for width_ratio in parse_float_list(args.width_ratios):
        for phase_fraction in parse_float_list(args.phase_fractions):
            for polarity in parse_int_list(args.polarities):
                geometry = LocalizedPacketGeometry(
                    wavelength=args.shape_period_ratio * args.mean_thickness,
                    amplitude=args.shape_amplitude_ratio * args.mean_thickness,
                    mean_thickness=args.mean_thickness,
                    feature_width=width_ratio * args.mean_thickness,
                    feature_phase=2.0 * np.pi * phase_fraction,
                    feature_polarity=polarity,
                    feature_strength=args.feature_strength,
                )
                tasks.append(task_for_geometry(geometry, "shape"))
    return tasks


def selected_tasks(args):
    tasks = []
    for amplitude_ratio in parse_float_list(args.amplitude_ratios):
        for period_ratio in parse_float_list(args.period_ratios):
            if args.selected_width_ratio >= period_ratio:
                continue
            geometry = LocalizedPacketGeometry(
                wavelength=period_ratio * args.mean_thickness,
                amplitude=amplitude_ratio * args.mean_thickness,
                mean_thickness=args.mean_thickness,
                feature_width=args.selected_width_ratio * args.mean_thickness,
                feature_phase=2.0 * np.pi * args.selected_phase_fraction,
                feature_polarity=args.selected_polarity,
                feature_strength=args.feature_strength,
            )
            tasks.append(task_for_geometry(geometry, "selected"))
    return tasks


def focused_tasks(args):
    geometry = LocalizedPacketGeometry(
        wavelength=args.focused_period_ratio * args.mean_thickness,
        amplitude=args.focused_amplitude_ratio * args.mean_thickness,
        mean_thickness=args.mean_thickness,
        feature_width=args.focused_width_ratio * args.mean_thickness,
        feature_phase=2.0 * np.pi * args.focused_phase_fraction,
        feature_polarity=args.focused_polarity,
        feature_strength=args.feature_strength,
    )
    tasks = [
        task_for_geometry(geometry, "focused", maximum_age=args.maximum_age, refined=False),
        task_for_geometry(geometry, "focused", maximum_age=args.maximum_age, refined=True),
    ]
    if args.focused_grading_power is not None:
        for task in tasks:
            task["mesh_vertical_grading_power"] = args.focused_grading_power
            task["sampler_vertical_grading_power"] = args.focused_grading_power
            task["resolution"] += f" p={args.focused_grading_power:g}"
    return tasks


def task_key(task):
    fields = (
        "experiment", "wavelength", "amplitude", "mean_thickness",
        "feature_width", "feature_phase", "feature_polarity", "feature_strength",
        "phase_origin", "mesh_nx", "mesh_nz", "mesh_vertical_grading_power",
        "sampler_nx", "sampler_nz", "sampler_vertical_grading_power",
        "source_count", "retained_flux_fraction", "time_step", "resolution",
    )
    values = []
    for field in fields:
        value = task[field]
        values.append(round(float(value), 12) if isinstance(value, (float, np.floating)) else value)
    return tuple(values)


def row_key(row):
    task = {
        "experiment": row["localized-packet experiment"],
        "wavelength": row["wavelength (m)"],
        "amplitude": row["bed amplitude (m)"],
        "mean_thickness": row["mean ice thickness (m)"],
        "feature_width": row["localized feature FWHM (m)"],
        "feature_phase": row["localized feature phase (rad)"],
        "feature_polarity": int(row["localized feature polarity"]),
        "feature_strength": row["localized / broad raw amplitude"],
        "phase_origin": row["bed phase origin (m; coordinate gauge)"],
        "mesh_nx": int(row["mesh nx"]),
        "mesh_nz": int(row["mesh nz"]),
        "mesh_vertical_grading_power": row["mesh vertical grading power"],
        "sampler_nx": int(row["sampler nx"]),
        "sampler_nz": int(row["sampler nz"]),
        "sampler_vertical_grading_power": row["sampler vertical grading power"],
        "source_count": int(row["candidate source count"]),
        "retained_flux_fraction": row["retained accumulation-flux fraction"],
        "time_step": row["characteristic time step (yr)"],
        "resolution": row["localized-packet resolution"],
    }
    return task_key(task)


def reusable_rows(output):
    if not output.exists():
        return {}
    table = pd.read_csv(output)
    required = {
        "localized-packet method version", "localized-packet experiment",
        "localized feature FWHM (m)", "localized feature polarity", "status",
    }
    if not required.issubset(table.columns):
        return {}
    reusable = {}
    for _, row in table.iterrows():
        if row["localized-packet method version"] != METHOD_VERSION:
            continue
        status = str(row["status"])
        if status in ("success", "screened below numerical surface signal"):
            reusable[row_key(row)] = row.to_dict()
    return reusable


def sorted_table(rows):
    return pd.DataFrame(rows).sort_values(
        [
            "localized-packet experiment", "localized-packet resolution",
            "amplitude / mean thickness", "wavelength / mean thickness",
            "localized feature FWHM / mean thickness",
            "localized feature phase / (2 pi)", "localized feature polarity",
        ]
    ).reset_index(drop=True)


def run_tasks(tasks, output, jobs):
    reusable = reusable_rows(output)
    rows, pending = [], []
    for task in tasks:
        if task_key(task) in reusable:
            rows.append(reusable[task_key(task)])
        else:
            pending.append(task)
    print(f"Localized grid: {len(tasks)} cells, {len(rows)} reused, {len(pending)} pending.", flush=True)
    if jobs == 1:
        solved = map(solve_task, pending)
        pool = None
    else:
        pool = multiprocessing.get_context("spawn").Pool(processes=jobs, maxtasksperchild=10)
        solved = pool.imap(solve_task, pending, chunksize=1)
    try:
        for number, row in enumerate(solved, start=1):
            rows.append(row)
            print(json.dumps({
                "completed": f"{number}/{len(pending)}",
                "A/H": row["amplitude / mean thickness"],
                "L/H": row["wavelength / mean thickness"],
                "ell/H": row["localized feature FWHM / mean thickness"],
                "phase": row["localized feature phase / (2 pi)"],
                "polarity": row["localized feature polarity"],
                "status": row["status"],
                "minimum J": row.get("minimum J"),
            }), flush=True)
            if number % 2 == 0:
                sorted_table(rows).to_csv(output, index=False, na_rep="NA")
    except BaseException:
        if pool is not None:
            pool.terminate(); pool.join()
        raise
    else:
        if pool is not None:
            pool.close(); pool.join()
    table = sorted_table(rows)
    table.to_csv(output, index=False, na_rep="NA")
    return table


def write_metadata(table, output, args):
    success = table[table["status"].eq("success")]
    def minimum_record(rows):
        if not len(rows):
            return None
        row = rows.loc[rows["minimum J"].idxmin()]
        return {
            "minimum J": float(row["minimum J"]),
            "raw minimum J": float(row["raw minimum J to exit"]),
            "A/H": float(row["amplitude / mean thickness"]),
            "L/H": float(row["wavelength / mean thickness"]),
            "ell/H": float(row["localized feature FWHM / mean thickness"]),
            "phase/(2pi)": float(row["localized feature phase / (2 pi)"]),
            "polarity": int(row["localized feature polarity"]),
            "maximum identity residual": float(
                row["maximum relative streamfunction identity residual"]
            ),
        }
    credible = success[
        success["maximum relative streamfunction identity residual"] <= 0.10
    ]
    metadata = {
        "method version": METHOD_VERSION,
        "mode": args.mode,
        "bed": "b=-H+A[cos(theta)+p*r*(V_kappa(theta-phi)-mean(V_kappa))]/M",
        "invariants": ["mean thickness H", "minimum thickness H-A", "exact periodicity"],
        "cell count": int(len(table)),
        "status counts": {str(k): int(v) for k, v in table["status"].value_counts(dropna=False).items()},
        "lowest numerical primary J (unfiltered)": minimum_record(success),
        "lowest primary J with identity residual <= 10%": minimum_record(credible),
    }
    path = output.with_name(output.stem + "_metadata.json")
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def parse_float_list(text):
    return np.array([float(value) for value in text.split(",")])


def parse_int_list(text):
    return np.array([int(value) for value in text.split(",")], dtype=int)


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shape", "selected", "focused"), default="shape")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mean-thickness", type=float, default=MEAN_THICKNESS)
    parser.add_argument("--feature-strength", type=float, default=DEFAULT_FEATURE_STRENGTH)
    parser.add_argument("--maximum-age", type=float, default=MAXIMUM_AGE)

    parser.add_argument("--shape-amplitude-ratio", type=float, default=SHAPE_AMPLITUDE_RATIO)
    parser.add_argument("--shape-period-ratio", type=float, default=SHAPE_PERIOD_RATIO)
    parser.add_argument("--width-ratios", default=",".join(str(v) for v in WIDTH_RATIOS))
    parser.add_argument("--phase-fractions", default=",".join(str(v) for v in PHASE_FRACTIONS))
    parser.add_argument("--polarities", default=",".join(str(v) for v in POLARITIES))

    parser.add_argument("--amplitude-ratios", default=",".join(str(v) for v in AMPLITUDE_RATIOS))
    parser.add_argument("--period-ratios", default=",".join(str(v) for v in PERIOD_RATIOS))
    parser.add_argument("--selected-width-ratio", type=float, default=DEFAULT_SELECTED_WIDTH_RATIO)
    parser.add_argument("--selected-phase-fraction", type=float, default=DEFAULT_SELECTED_PHASE_FRACTION)
    parser.add_argument("--selected-polarity", type=int, choices=(-1, 1), default=DEFAULT_SELECTED_POLARITY)

    parser.add_argument("--focused-amplitude-ratio", type=float, default=0.90)
    parser.add_argument("--focused-period-ratio", type=float, default=4.0)
    parser.add_argument("--focused-width-ratio", type=float, default=1.0 / 12.0)
    parser.add_argument("--focused-phase-fraction", type=float, default=0.25)
    parser.add_argument("--focused-polarity", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--focused-grading-power", type=float)
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 2:
        parser.error("--jobs must be 1 or 2")
    if args.output is None:
        args.output = repository / "data" / f"steady_characteristic_localized_packet_{args.mode}.csv"
    return args


def main():
    args = parse_arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    builders = {"shape": shape_tasks, "selected": selected_tasks, "focused": focused_tasks}
    table = run_tasks(builders[args.mode](args), args.output, args.jobs)
    metadata = write_metadata(table, args.output, args)
    print(table["status"].value_counts(dropna=False).to_string())
    print(f"Wrote {metadata}")


if __name__ == "__main__":
    main()
