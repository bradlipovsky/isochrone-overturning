"""Steady-characteristic sweep for a smooth two-harmonic periodic bed.

The dimensional bed is

    b(x) = -H + A f(theta; q, phi),
    f = [cos(theta) + q cos(2 theta + phi)] / M(q, phi),
    M = max_theta [cos(theta) + q cos(2 theta + phi)],
    theta = 2 pi (x - x0) / lambda.

This positive-maximum normalization keeps both the mean ice thickness ``H``
and the minimum ice thickness ``H - A`` fixed while ``q`` and ``phi`` change
the shape.  It also recovers the one-harmonic sinusoid exactly at ``q = 0``.
The peak-to-trough relief is deliberately an output rather than a third
quantity hidden in the normalization.

Two restartable grids are available:

``shape``
    Scan ``q``, relative phase, and wavelength at fixed ``H`` and ``A/H``.
    This is the inexpensive shape-selection experiment.

``selected``
    After choosing a two-harmonic shape, scan the original mean-thickness,
    amplitude, and wavelength axes for that fixed ``q`` and ``phi``.

The shortest requested wavelength is ``lambda/H = 1/12``.  Consequently the
Stokes mesh and sampler use symmetric endpoint clustering for
``lambda/H <= 1/4``.  Longer waves use a uniform terrain-following coordinate
so that thin-neck refinement remains quasi-uniform rather than creating
increasingly anisotropic endpoint cells.  Every case also records its largest
characteristic step as a fraction of one wavelength.
The solver refuses to run until the shared characteristic module supports
geometry methods; this prevents an older sinusoid-only core from silently
discarding the second harmonic.
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

import steady_characteristic_sinusoidal_parameter_map as core


WAVELENGTH_RATIOS = np.array(
    (1.0 / 12.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0,
     1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0)
)
MEAN_THICKNESSES = np.array((50.0, 100.0, 160.0, 240.0, 320.0))
AMPLITUDE_RATIOS = np.array(
    (0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45,
     0.55, 0.65, 0.75, 0.85, 0.90, 0.95)
)

# The quick scan brackets the q=1/4 topological transition and includes both
# symmetric and strongly asymmetric shapes with eight evenly spaced phases.
QUICK_Q_RATIOS = np.array((0.0, 0.125, 0.24, 0.26, 0.5, 1.0))
FULL_Q_RATIOS = np.array(
    (0.0, 0.0625, 0.125, 0.1875, 0.24, 0.25, 0.26,
     0.375, 0.5, 0.75, 1.0)
)
QUICK_PHASE_COUNT = 8
FULL_PHASE_COUNT = 12

DEFAULT_SHAPE_THICKNESS = 160.0
DEFAULT_SHAPE_AMPLITUDE_RATIO = 0.85
DEFAULT_SELECTED_Q = 0.125
DEFAULT_SELECTED_PHASE_FRACTION = 0.75

SHORT_WAVE_VERTICAL_GRADING_POWER = 2.0
RETAINED_FLUX_FRACTION = 0.80
SOURCE_COUNT = 241
MAXIMUM_AGE = 500_000.0
TIME_STEP = 25.0
MINIMUM_SURFACE_SIGNAL_TO_BASAL_LEAKAGE = 10.0
METHOD_VERSION = (
    "two-harmonic-positive-maximum-v2-adaptive-grading__"
    + core.DIAGNOSTIC_METHOD_VERSION
)


def _unique_circular(phases, tolerance=5.0e-4):
    """Deduplicate phases on a circle, including split repeated roots."""
    phases = np.sort(np.mod(np.asarray(phases, dtype=float), 2.0 * np.pi))
    unique = []
    for phase in phases:
        if not unique or phase - unique[-1] > tolerance:
            unique.append(float(phase))
    if len(unique) > 1 and 2.0 * np.pi - unique[-1] + unique[0] <= tolerance:
        unique[0] = 0.5 * (unique[0] + unique[-1] - 2.0 * np.pi)
        unique = unique[:-1]
    return np.mod(np.asarray(unique), 2.0 * np.pi)


def stationary_phases(second_harmonic_ratio, relative_phase):
    """Return every stationary phase of ``cos(t)+q cos(2t+phi)``.

    With ``z = exp(i t)``, stationarity gives a quartic polynomial.  Repeated
    roots at the shape transition ``q=1/4`` are slightly displaced from the
    unit circle by roundoff, so the unit-circle tolerance is intentionally
    looser than machine precision and circular duplicates are merged.
    """
    q = float(second_harmonic_ratio)
    phi = float(relative_phase)
    if q == 0.0:
        return np.array((0.0, np.pi))
    coefficients = np.array(
        (
            2.0 * q * np.exp(1j * phi),
            1.0,
            0.0,
            -1.0,
            -2.0 * q * np.exp(-1j * phi),
        ),
        dtype=complex,
    )
    roots = np.roots(coefficients)
    on_circle = roots[np.abs(np.abs(roots) - 1.0) < 2.0e-4]
    phases = np.angle(on_circle)

    # A fine-grid maximum and minimum are harmless redundant candidates in
    # ordinary cases and guarantee robust normalization at repeated roots.
    audit_phase = np.linspace(0.0, 2.0 * np.pi, 8192, endpoint=False)
    audit_values = np.cos(audit_phase) + q * np.cos(
        2.0 * audit_phase + phi
    )
    phases = np.r_[
        phases,
        audit_phase[int(np.argmax(audit_values))],
        audit_phase[int(np.argmin(audit_values))],
    ]
    return _unique_circular(phases)


@dataclass(frozen=True)
class TwoHarmonicGeometry:
    """One smooth, exactly periodic two-harmonic bed cell."""

    wavelength: float
    amplitude: float
    mean_thickness: float
    second_harmonic_ratio: float
    relative_phase: float
    phase_origin: float = 0.0

    def __post_init__(self):
        values = (
            self.wavelength,
            self.amplitude,
            self.mean_thickness,
            self.second_harmonic_ratio,
            self.relative_phase,
            self.phase_origin,
        )
        if not all(np.isfinite(values)):
            raise ValueError("geometry values must be finite")
        if self.wavelength <= 0.0:
            raise ValueError("wavelength must be positive")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be nonnegative")
        if self.mean_thickness <= self.amplitude:
            raise ValueError("mean thickness must exceed amplitude")
        if self.second_harmonic_ratio < 0.0:
            raise ValueError("second / first harmonic amplitude must be >= 0")

    @cached_property
    def _stationary_phases(self):
        return stationary_phases(
            self.second_harmonic_ratio, self.relative_phase
        )

    def raw_profile_from_phase(self, phase):
        phase = np.asarray(phase, dtype=float)
        return np.cos(phase) + self.second_harmonic_ratio * np.cos(
            2.0 * phase + self.relative_phase
        )

    @cached_property
    def normalization_maximum(self):
        return float(
            np.max(self.raw_profile_from_phase(self._stationary_phases))
        )

    @cached_property
    def normalized_minimum(self):
        return float(
            np.min(self.raw_profile_from_phase(self._stationary_phases))
            / self.normalization_maximum
        )

    def phase(self, x):
        return (
            2.0
            * np.pi
            * (np.asarray(x, dtype=float) - self.phase_origin)
            / self.wavelength
        )

    def normalized_bed_profile(self, x):
        return (
            self.raw_profile_from_phase(self.phase(x))
            / self.normalization_maximum
        )

    def thickness(self, x):
        return self.mean_thickness - self.amplitude * (
            self.normalized_bed_profile(x)
        )

    def thickness_slope(self, x):
        phase = self.phase(x)
        return (
            self.amplitude
            * 2.0
            * np.pi
            / self.wavelength
            * (
                np.sin(phase)
                + 2.0
                * self.second_harmonic_ratio
                * np.sin(2.0 * phase + self.relative_phase)
            )
            / self.normalization_maximum
        )

    def bed_height(self, x):
        return -self.thickness(x)

    @property
    def minimum_thickness(self):
        return self.mean_thickness - self.amplitude

    @property
    def maximum_thickness(self):
        return (
            self.mean_thickness
            - self.amplitude * self.normalized_minimum
        )

    @property
    def amplitude_ratio(self):
        return self.amplitude / self.mean_thickness

    @property
    def wavelength_ratio(self):
        return self.wavelength / self.mean_thickness

    @property
    def relief(self):
        return self.amplitude * (1.0 - self.normalized_minimum)

    @cached_property
    def peak_bed_slope(self):
        phase = np.linspace(0.0, 2.0 * np.pi, 16384, endpoint=False)
        derivative = -np.sin(phase) - 2.0 * self.second_harmonic_ratio * (
            np.sin(2.0 * phase + self.relative_phase)
        )
        return float(
            self.amplitude
            * 2.0
            * np.pi
            / self.wavelength
            * np.max(np.abs(derivative))
            / self.normalization_maximum
        )

    @cached_property
    def local_extremum_counts(self):
        """Count maxima and minima, retaining flat repeated extrema."""
        delta = 1.0e-3
        phases = self._stationary_phases
        center = self.raw_profile_from_phase(phases)
        left = self.raw_profile_from_phase(phases - delta)
        right = self.raw_profile_from_phase(phases + delta)
        maxima = (center >= left) & (center >= right)
        minima = (center <= left) & (center <= right)
        return int(np.count_nonzero(maxima)), int(np.count_nonzero(minima))

    @property
    def screening_chi(self):
        """Fundamental shielding based on the minimum local ice gap."""
        return 2.0 * np.pi * self.minimum_thickness / self.wavelength


def verify_generic_geometry_core():
    """Fail rather than silently solve this bed as a one-mode sinusoid."""
    probe = TwoHarmonicGeometry(480.0, 80.0, 160.0, 0.5, 0.5 * np.pi)
    x = np.array((0.0, 0.13, 0.41, 0.78)) * probe.wavelength
    represented = np.asarray(core.thickness(x, probe), dtype=float)
    expected = np.asarray(probe.thickness(x), dtype=float)
    if not np.allclose(represented, expected, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError(
            "The shared steady-characteristic core does not yet delegate "
            "thickness and slope evaluation to geometry methods. Refusing "
            "to discard the second harmonic silently."
        )


def round_up(value, multiple):
    return int(multiple * np.ceil(float(value) / multiple))


def mesh_nx_for_geometry(_geometry):
    """Give the second harmonic at least 48 horizontal cells per cycle."""
    return 128 if _geometry.amplitude_ratio >= 0.90 else 96


def mesh_nz_for_geometry(geometry):
    """Resolve the H-scaled second-harmonic decay layer."""
    ratio = geometry.wavelength_ratio
    requested = max(32.0, 6.0 * np.sqrt(8.0 * np.pi / ratio))
    if geometry.amplitude_ratio >= 0.90:
        requested = max(requested, 96.0)
    return round_up(requested, 8)


def vertical_grading_power_for_geometry(geometry):
    """Cluster endpoints only when the fundamental is genuinely short.

    Applying quadratic endpoint clustering to a long-wave thin neck creates
    extremely anisotropic cells even though the vertical boundary layer is
    already well resolved by the thin-neck mesh.  Direct uniform-versus-graded
    audits show that the quasi-uniform coordinate is better converged at
    ``lambda/H=1/2``; clustering is retained only for ``lambda/H <= 1/4``.
    """
    if geometry.wavelength_ratio <= 0.25 + 1.0e-12:
        return SHORT_WAVE_VERTICAL_GRADING_POWER
    return 1.0


def sampler_nx_for_mesh(mesh_nx):
    return int(max(385, 4 * int(mesh_nx) + 1))


def sampler_nz_for_mesh(mesh_nz):
    return int(max(193, 2 * int(mesh_nz) + 1))


def geometry_diagnostics(geometry):
    maxima, minima = geometry.local_extremum_counts
    chi = geometry.screening_chi
    q = geometry.second_harmonic_ratio
    phi = geometry.relative_phase
    return {
        "second / first harmonic amplitude": q,
        "relative phase (rad)": phi,
        "relative phase / (2 pi)": float(np.mod(phi, 2.0 * np.pi) / (2.0 * np.pi)),
        "normalization maximum": geometry.normalization_maximum,
        "normalized bed-profile minimum": geometry.normalized_minimum,
        "normalized relief": geometry.relief / geometry.mean_thickness,
        "bed peak-to-trough relief / mean thickness": (
            geometry.relief / geometry.mean_thickness
        ),
        "bed peak-to-trough relief (m)": geometry.relief,
        "maximum ice thickness (m)": geometry.maximum_thickness,
        "peak absolute bed slope": geometry.peak_bed_slope,
        "fore-aft asymmetry q sin(phi)": float(q * np.sin(phi)),
        "peakedness q cos(phi)": float(q * np.cos(phi)),
        "bed stationary point count": int(len(geometry._stationary_phases)),
        "bed local maximum count": maxima,
        "bed local minimum count": minima,
        "screening chi": chi,
        "fundamental screening proxy exp(-chi)": float(np.exp(-chi)),
        "second-harmonic screening chi": (2.0 * chi if q > 0.0 else np.nan),
        "second-harmonic screening proxy exp(-2 chi)": (
            float(np.exp(-2.0 * chi)) if q > 0.0 else np.nan
        ),
        "shortest harmonic decay length / mean thickness": (
            geometry.wavelength_ratio / (4.0 * np.pi)
            if q > 0.0
            else geometry.wavelength_ratio / (2.0 * np.pi)
        ),
    }


def task_for_geometry(geometry, maximum_age):
    mesh_nx = mesh_nx_for_geometry(geometry)
    mesh_nz = mesh_nz_for_geometry(geometry)
    vertical_grading_power = vertical_grading_power_for_geometry(geometry)
    thin_neck = geometry.amplitude_ratio >= 0.90
    return {
        "wavelength": float(geometry.wavelength),
        "amplitude": float(geometry.amplitude),
        "mean_thickness": float(geometry.mean_thickness),
        "second_harmonic_ratio": float(geometry.second_harmonic_ratio),
        "relative_phase": float(geometry.relative_phase),
        "phase_origin": float(geometry.phase_origin),
        "mesh_nx": mesh_nx,
        "mesh_nz": mesh_nz,
        "mesh_vertical_grading_power": vertical_grading_power,
        "sampler_nx": 641 if thin_neck else sampler_nx_for_mesh(mesh_nx),
        "sampler_nz": 321 if thin_neck else sampler_nz_for_mesh(mesh_nz),
        "sampler_vertical_grading_power": vertical_grading_power,
        "source_count": 321 if thin_neck else SOURCE_COUNT,
        "retained_flux_fraction": RETAINED_FLUX_FRACTION,
        "time_step": 12.5 if thin_neck else TIME_STEP,
        "maximum_age": float(
            max(maximum_age, 1_500_000.0) if thin_neck else maximum_age
        ),
    }


def geometry_from_task(task):
    return TwoHarmonicGeometry(
        wavelength=task["wavelength"],
        amplitude=task["amplitude"],
        mean_thickness=task["mean_thickness"],
        second_harmonic_ratio=task["second_harmonic_ratio"],
        relative_phase=task["relative_phase"],
        phase_origin=task["phase_origin"],
    )


def solve_task(task):
    verify_generic_geometry_core()
    geometry = geometry_from_task(task)
    result = core.solve_parameter_case(
        geometry,
        mesh_nx=task["mesh_nx"],
        mesh_nz=task["mesh_nz"],
        vertical_grading_power=task["mesh_vertical_grading_power"],
        sampler_nx=task["sampler_nx"],
        sampler_nz=task["sampler_nz"],
        sampler_vertical_grading_power=task[
            "sampler_vertical_grading_power"
        ],
        source_count=task["source_count"],
        retained_flux_fraction=task["retained_flux_fraction"],
        time_step=task["time_step"],
        maximum_age=task["maximum_age"],
        minimum_surface_signal_to_basal_leakage=(
            MINIMUM_SURFACE_SIGNAL_TO_BASAL_LEAKAGE
        ),
    )
    result.update(geometry_diagnostics(geometry))
    result["two-harmonic method version"] = METHOD_VERSION
    surface_signal = float(
        result.get("maximum absolute surface mass balance (m/yr)", np.nan)
    )
    basal_leakage = float(
        result.get("rms basal normal speed (m/yr)", np.nan)
    )
    if basal_leakage > 0.0:
        signal_ratio = surface_signal / basal_leakage
    elif surface_signal > 0.0:
        signal_ratio = np.inf
    else:
        signal_ratio = 0.0
    result["surface signal / basal normal leakage"] = signal_ratio
    result["minimum credible surface signal / basal leakage"] = (
        MINIMUM_SURFACE_SIGNAL_TO_BASAL_LEAKAGE
    )
    result["computed minimum J before signal screening"] = result.get(
        "minimum J", np.nan
    )
    closure = float(
        result.get("sampled relative surface mass balance closure", np.inf)
    )
    status = str(result.get("status", ""))
    eligible_status = status == "success" or status.startswith(
        "no retained surface-deposition source"
    )
    if eligible_status and (
        signal_ratio < MINIMUM_SURFACE_SIGNAL_TO_BASAL_LEAKAGE
        or closure > 1.0e-3
    ):
        result["status"] = "screened below numerical surface signal"
        result["classification"] = (
            "not evaluated: surface signal below numerical leakage"
        )
    return result


def shape_tasks(args):
    q_values = (
        parse_float_list(args.q_values)
        if args.q_values
        else (QUICK_Q_RATIOS if args.grid == "quick" else FULL_Q_RATIOS)
    )
    phase_count = args.phase_count or (
        QUICK_PHASE_COUNT if args.grid == "quick" else FULL_PHASE_COUNT
    )
    phases = 2.0 * np.pi * np.arange(phase_count) / phase_count
    wavelength_ratios = (
        parse_float_list(args.wavelength_ratios)
        if args.wavelength_ratios
        else WAVELENGTH_RATIOS
    )
    tasks = []
    for wavelength_ratio in wavelength_ratios:
        for q in q_values:
            # Relative phase has no meaning when the second harmonic is absent.
            current_phases = np.array((0.0,)) if q == 0.0 else phases
            for phase in current_phases:
                geometry = TwoHarmonicGeometry(
                    wavelength=wavelength_ratio * args.mean_thickness,
                    amplitude=args.amplitude_ratio * args.mean_thickness,
                    mean_thickness=args.mean_thickness,
                    second_harmonic_ratio=q,
                    relative_phase=phase,
                )
                tasks.append(task_for_geometry(geometry, args.maximum_age))
    return tasks


def selected_shape_tasks(args):
    wavelength_ratios = (
        parse_float_list(args.wavelength_ratios)
        if args.wavelength_ratios
        else WAVELENGTH_RATIOS
    )
    tasks = []
    phase = 2.0 * np.pi * args.selected_phase_fraction
    for mean_thickness in MEAN_THICKNESSES:
        for amplitude_ratio in AMPLITUDE_RATIOS:
            for wavelength_ratio in wavelength_ratios:
                geometry = TwoHarmonicGeometry(
                    wavelength=wavelength_ratio * mean_thickness,
                    amplitude=amplitude_ratio * mean_thickness,
                    mean_thickness=mean_thickness,
                    second_harmonic_ratio=args.selected_q,
                    relative_phase=phase,
                )
                tasks.append(task_for_geometry(geometry, args.maximum_age))
    return tasks


def task_key(task):
    fields = (
        "mean_thickness",
        "amplitude",
        "wavelength",
        "second_harmonic_ratio",
        "relative_phase",
        "phase_origin",
        "mesh_nx",
        "mesh_nz",
        "mesh_vertical_grading_power",
        "sampler_nx",
        "sampler_nz",
        "sampler_vertical_grading_power",
        "source_count",
        "retained_flux_fraction",
        "time_step",
    )
    return tuple(
        round(float(task[field]), 12)
        if isinstance(task[field], (float, np.floating))
        else int(task[field])
        for field in fields
    )


def row_key(row):
    task = {
        "mean_thickness": row["mean ice thickness (m)"],
        "amplitude": row["bed amplitude (m)"],
        "wavelength": row["wavelength (m)"],
        "second_harmonic_ratio": row["second / first harmonic amplitude"],
        "relative_phase": row["relative phase (rad)"],
        "phase_origin": row["bed phase origin (m; coordinate gauge)"],
        "mesh_nx": row["mesh nx"],
        "mesh_nz": row["mesh nz"],
        "mesh_vertical_grading_power": row[
            "mesh vertical grading power"
        ],
        "sampler_nx": row["sampler nx"],
        "sampler_nz": row["sampler nz"],
        "sampler_vertical_grading_power": row[
            "sampler vertical grading power"
        ],
        "source_count": row["candidate source count"],
        "retained_flux_fraction": row[
            "retained accumulation-flux fraction"
        ],
        "time_step": row["characteristic time step (yr)"],
    }
    return task_key(task)


def reusable_rows(output):
    if not output.exists():
        return {}
    table = pd.read_csv(output)
    required = {
        "two-harmonic method version",
        "second / first harmonic amplitude",
        "relative phase (rad)",
        "mesh vertical grading power",
        "sampler vertical grading power",
        "minimum J",
        "status",
    }
    if not required.issubset(table.columns):
        return {}
    reusable = {}
    successful_diagnostic_columns = (
        "surface signal / basal normal leakage",
        "maximum horizontal step (wavelengths)",
        "mechanism C at minimum J",
        "u / source u at minimum J",
        "maximum relative streamfunction identity residual",
    )
    for _, row in table.iterrows():
        status = str(row["status"])
        successful = (
            status == "success"
            or status.startswith("analytic flat-bed limit")
            or status.startswith("screened below numerical surface signal")
        )
        complete_success = status != "success" or all(
            column in row.index and pd.notna(row[column])
            for column in successful_diagnostic_columns
        )
        if (
            successful
            and complete_success
            and row["two-harmonic method version"] == METHOD_VERSION
        ):
            reusable[row_key(row)] = row.to_dict()
    return reusable


def sorted_table(rows):
    return pd.DataFrame(rows).sort_values(
        [
            "mean ice thickness (m)",
            "amplitude / mean thickness",
            "wavelength / mean thickness",
            "second / first harmonic amplitude",
            "relative phase (rad)",
        ]
    ).reset_index(drop=True)


def run_tasks(tasks, output, jobs):
    reusable = reusable_rows(output)
    rows = []
    pending = []
    for task in tasks:
        key = task_key(task)
        if key in reusable:
            rows.append(reusable[key])
        else:
            pending.append(task)
    print(
        f"Two-harmonic grid: {len(tasks)} cells, {len(rows)} reused, "
        f"{len(pending)} pending.",
        flush=True,
    )
    if jobs == 1:
        solved = map(solve_task, pending)
        pool = None
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=jobs, maxtasksperchild=20)
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
                        "q": row["second / first harmonic amplitude"],
                        "phi/(2pi)": row["relative phase / (2 pi)"],
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
    table = sorted_table(rows)
    table.to_csv(output, index=False, na_rep="NA")
    return table


def write_metadata(table, output, args):
    numerical = table[table["status"].eq("success")]
    minimum = None
    if len(numerical) and numerical["minimum J"].notna().any():
        row = numerical.loc[numerical["minimum J"].idxmin()]
        minimum = {
            "minimum J": float(row["minimum J"]),
            "H (m)": float(row["mean ice thickness (m)"]),
            "A/H": float(row["amplitude / mean thickness"]),
            "lambda/H": float(row["wavelength / mean thickness"]),
            "q": float(row["second / first harmonic amplitude"]),
            "phi/(2pi)": float(row["relative phase / (2 pi)"]),
        }
    metadata = {
        "method version": METHOD_VERSION,
        "mode": args.mode,
        "normalization": (
            "b=-H+A[cos(theta)+q cos(2 theta+phi)]/M; "
            "M=max_theta[cos(theta)+q cos(2 theta+phi)]"
        ),
        "normalization invariants": [
            "mean thickness H",
            "minimum thickness H-A",
            "q=0 exactly recovers the one-harmonic bed",
        ],
        "wavelength / mean thickness": sorted(
            {
                round(float(value), 12)
                for value in table["wavelength / mean thickness"].unique()
            }
        ),
        "vertical grading powers present": sorted(
            float(value)
            for value in table["mesh vertical grading power"].dropna().unique()
        ),
        "characteristic time steps present (yr)": sorted(
            float(value)
            for value in table["characteristic time step (yr)"].dropna().unique()
        ),
        "candidate source counts present": sorted(
            int(value)
            for value in table["candidate source count"].dropna().unique()
        ),
        "safety maximum ages present (yr)": sorted(
            float(value)
            for value in table["safety maximum age (yr)"].dropna().unique()
        ),
        "resolution policy": {
            "horizontal mesh": "128 columns for A/H >= 0.90; otherwise 96",
            "vertical mesh": (
                "at least 96 rows for A/H >= 0.90; quadratic endpoint "
                "grading only for lambda/H <= 0.25"
            ),
            "thin-neck A/H >= 0.90": (
                "sampler 641 x 321, 321 sources, dt=12.5 yr, "
                "maximum age at least 1.5 Myr"
            ),
        },
        "minimum surface signal / basal leakage": (
            MINIMUM_SURFACE_SIGNAL_TO_BASAL_LEAKAGE
        ),
        "cell count": int(len(table)),
        "status counts": {
            str(key): int(value)
            for key, value in table["status"].value_counts(dropna=False).items()
        },
        "lowest primary J": minimum,
    }
    path = output.with_name(output.stem + "_metadata.json")
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def parse_float_list(text):
    return np.array(tuple(float(value) for value in text.split(",")))


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shape", "selected"), default="shape")
    parser.add_argument("--grid", choices=("quick", "full"), default="quick")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--maximum-age", type=float, default=MAXIMUM_AGE)
    parser.add_argument("--mean-thickness", type=float, default=DEFAULT_SHAPE_THICKNESS)
    parser.add_argument(
        "--amplitude-ratio", type=float, default=DEFAULT_SHAPE_AMPLITUDE_RATIO
    )
    parser.add_argument(
        "--q-values",
        help="comma-separated q values overriding the quick/full shape grid",
    )
    parser.add_argument("--phase-count", type=int)
    parser.add_argument(
        "--wavelength-ratios",
        help="comma-separated lambda/H values overriding the standard grid",
    )
    parser.add_argument("--selected-q", type=float, default=DEFAULT_SELECTED_Q)
    parser.add_argument(
        "--selected-phase-fraction",
        type=float,
        default=DEFAULT_SELECTED_PHASE_FRACTION,
        help="relative phase divided by 2 pi",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase_count is not None and args.phase_count < 1:
        parser.error("--phase-count must be positive")
    if args.output is None:
        stem = (
            "steady_characteristic_two_harmonic_shape_scan.csv"
            if args.mode == "shape"
            else "steady_characteristic_two_harmonic_parameter_map.csv"
        )
        args.output = repository / "data" / stem
    return args


def main():
    args = parse_arguments()
    verify_generic_geometry_core()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks = shape_tasks(args) if args.mode == "shape" else selected_shape_tasks(args)
    table = run_tasks(tasks, args.output, args.jobs)
    metadata = write_metadata(table, args.output, args)
    print(table["status"].value_counts(dropna=False).to_string())
    print(f"Wrote {metadata}")


if __name__ == "__main__":
    main()
