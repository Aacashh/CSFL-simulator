"""Hardware-energy measurement helpers for the MAML-Select experiment suite.

CodeCarbon measures the compute block when its hardware backends are available.
Carbon emissions are still estimates: they are derived from measured electricity
consumption and a declared regional grid intensity.
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


@dataclass
class EnergyMeasurement:
    status: str
    tracker: str = "codecarbon.OfflineEmissionsTracker"
    tracking_scope: str = "simulator.run only; dataset loading and model setup excluded"
    country_iso_code: str = ""
    declared_grid_intensity_g_per_kwh: float = 0.0
    measured_energy_kwh: float = 0.0
    estimated_emissions_kg_from_tracker: float = 0.0
    estimated_emissions_g_declared_intensity: float = 0.0
    cpu_energy_kwh: float = 0.0
    gpu_energy_kwh: float = 0.0
    ram_energy_kwh: float = 0.0
    duration_seconds: float = 0.0
    csv_path: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CodeCarbonMeter:
    """Measure one simulation run without making CodeCarbon a hard dependency."""

    def __init__(
        self,
        output_dir: Path,
        run_label: str,
        country_iso_code: str,
        grid_intensity_g_per_kwh: float,
        enabled: bool = True,
        measure_power_secs: int = 1,
        verified_hardware_telemetry: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.run_label = run_label
        self.country_iso_code = str(country_iso_code).upper()
        self.grid_intensity = float(grid_intensity_g_per_kwh)
        self.enabled = bool(enabled)
        self.measure_power_secs = max(1, int(measure_power_secs))
        self.verified_hardware_telemetry = bool(verified_hardware_telemetry)
        self.output_file = f"codecarbon_{run_label}.csv"
        self.csv_path = self.output_dir / self.output_file
        self.tracker: Optional[Any] = None
        self._start_note = ""

    def _empty(self, status: str, note: str) -> Dict[str, Any]:
        return EnergyMeasurement(
            status=status,
            country_iso_code=self.country_iso_code,
            declared_grid_intensity_g_per_kwh=self.grid_intensity,
            csv_path=str(self.csv_path),
            note=note,
        ).to_dict()

    def start(self) -> None:
        if not self.enabled:
            self._start_note = "Hardware meter disabled by command-line flag."
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from codecarbon import OfflineEmissionsTracker
        except ImportError:
            self._start_note = (
                "CodeCarbon is not installed. Install the experiment requirements "
                "before collecting hardware-energy evidence."
            )
            return

        kwargs = {
            "project_name": self.run_label,
            "country_iso_code": self.country_iso_code,
            "measure_power_secs": self.measure_power_secs,
            "output_dir": str(self.output_dir),
            "output_file": self.output_file,
            "save_to_file": True,
            "log_level": "warning",
            "tracking_mode": "machine",
            "allow_multiple_runs": True,
        }
        try:
            self.tracker = OfflineEmissionsTracker(**kwargs)
        except TypeError:
            # Older CodeCarbon releases do not accept all recent tuning knobs.
            kwargs.pop("allow_multiple_runs", None)
            kwargs.pop("tracking_mode", None)
            try:
                self.tracker = OfflineEmissionsTracker(**kwargs)
            except Exception as exc:  # pragma: no cover - dependency dependent
                self._start_note = f"CodeCarbon could not initialize: {exc}"
                return
        except Exception as exc:  # pragma: no cover - dependency dependent
            self._start_note = f"CodeCarbon could not initialize: {exc}"
            return
        try:
            self.tracker.start()
        except Exception as exc:  # pragma: no cover - hardware dependent
            self._start_note = f"CodeCarbon could not start: {exc}"
            self.tracker = None

    def _latest_row(self) -> Dict[str, str]:
        if not self.csv_path.exists():
            return {}
        with self.csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1] if rows else {}

    def stop(self) -> Dict[str, Any]:
        if self.tracker is None:
            status = "disabled" if not self.enabled else "unavailable"
            return self._empty(status, self._start_note)

        try:
            emissions_kg = _float(self.tracker.stop())
        except Exception as exc:  # pragma: no cover - hardware dependent
            return self._empty("unavailable", f"CodeCarbon could not stop cleanly: {exc}")

        row = self._latest_row()
        energy_kwh = _float(row.get("energy_consumed"))
        if energy_kwh <= 0.0:
            note = (
                "The tracker did not report positive energy. Use a longer run and verify "
                "that RAPL or NVIDIA power telemetry is exposed on the experiment host."
            )
            status = "unavailable"
        elif self.verified_hardware_telemetry:
            note = (
                "The operator confirmed hardware telemetry on this host. Electricity is "
                "hardware-measured where CodeCarbon exposes telemetry; emissions remain estimates."
            )
            status = "measured"
        else:
            note = (
                "CodeCarbon reported electricity consumption, but hardware telemetry was not "
                "operator-verified. Treat this as an estimate until telemetry is confirmed."
            )
            status = "tracked_unverified"

        measurement = EnergyMeasurement(
            status=status,
            country_iso_code=self.country_iso_code,
            declared_grid_intensity_g_per_kwh=self.grid_intensity,
            measured_energy_kwh=energy_kwh,
            estimated_emissions_kg_from_tracker=_float(row.get("emissions"), emissions_kg),
            estimated_emissions_g_declared_intensity=energy_kwh * self.grid_intensity,
            cpu_energy_kwh=_float(row.get("cpu_energy")),
            gpu_energy_kwh=_float(row.get("gpu_energy")),
            ram_energy_kwh=_float(row.get("ram_energy")),
            duration_seconds=_float(row.get("duration")),
            csv_path=str(self.csv_path),
            note=note,
        )
        return measurement.to_dict()
