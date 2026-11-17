"""Clinical data service shared by the MCP server and the REST API.

Data sources (CARESIGNAL_SOURCE):
  synthetic  generated demo cohort (default)
  uci        de-identified real encounters (UCI diabetes dataset)
  fhir       live hospital record system over FHIR R4, refreshed every CARESIGNAL_FHIR_TTL seconds
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from .model import ARTIFACT_DIR, RiskModel

log = logging.getLogger("caresignal.service")

# Illustrative, well-known interaction pairs. NOT a complete drug database -
# a real deployment must use a licensed, maintained interaction source.
INTERACTIONS: dict[frozenset[str], tuple[str, str]] = {
    frozenset({"warfarin", "aspirin"}): ("major", "Additive bleeding risk."),
    frozenset({"warfarin", "ibuprofen"}): ("major", "NSAID increases bleeding risk on anticoagulation."),
    frozenset({"warfarin", "amiodarone"}): ("major", "Amiodarone potentiates warfarin; INR can rise markedly."),
    frozenset({"lisinopril", "spironolactone"}): ("moderate", "Hyperkalemia risk; monitor potassium and renal function."),
    frozenset({"lisinopril", "ibuprofen"}): ("moderate", "NSAID blunts ACE-inhibitor effect and raises acute kidney injury risk."),
    frozenset({"simvastatin", "amiodarone"}): ("major", "Raised simvastatin exposure; myopathy/rhabdomyolysis risk."),
    frozenset({"simvastatin", "amlodipine"}): ("moderate", "Raised simvastatin exposure; dose limits apply."),
    frozenset({"digoxin", "amiodarone"}): ("major", "Amiodarone raises digoxin levels; toxicity risk."),
    frozenset({"digoxin", "furosemide"}): ("moderate", "Diuretic-induced hypokalemia increases digoxin toxicity risk."),
    frozenset({"sertraline", "tramadol"}): ("major", "Serotonin syndrome and seizure risk."),
    frozenset({"clopidogrel", "omeprazole"}): ("moderate", "Omeprazole reduces clopidogrel activation."),
    frozenset({"ibuprofen", "furosemide"}): ("moderate", "NSAID reduces diuretic effect; fluid retention in heart failure."),
}

RISK_BANDS = [(0.30, "high"), (0.15, "medium"), (0.0, "low")]
CONDITIONS = ("chf", "copd", "diabetes", "ckd")
GROUPABLE = ("risk_band", "discharge_disposition", "sex", "chf", "copd", "diabetes", "ckd", "lives_alone")


def risk_band(p: float) -> str:
    return next(name for cutoff, name in RISK_BANDS if p >= cutoff)


def _clean(obj):
    """Make values JSON-safe: NaN becomes None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


class NotFound(LookupError):
    pass


class ClinicalService:
    def __init__(
        self,
        cohort: pd.DataFrame,
        model: RiskModel,
        *,
        metrics: dict | None = None,
        source: str = "synthetic",
        data_label: str | None = None,
    ):
        self.model = model
        self.source = source
        self.metrics = metrics or {}
        self.data_label = data_label or self.metrics.get("data_label", "")
        self.built_at = datetime.now(timezone.utc)
        self.df = cohort.assign(patient_id=cohort["patient_id"].astype(str).str.upper())
        self.df = self.df.set_index("patient_id", drop=False)
        self.df["predicted_risk"] = model.predict(self.df).round(4)
        self.df["risk_band"] = self.df["predicted_risk"].map(risk_band)

    # -- source information -------------------------------------------------
    def info(self) -> dict:
        observed = self.df["readmitted_30d"].mean() if len(self.df) else np.nan
        return _clean({
            "source": self.source,
            "data_label": self.data_label,
            "n_patients": int(len(self.df)),
            "observed_readmission_rate": round(float(observed), 4) if pd.notna(observed) else None,
            "test_auroc": self.metrics.get("test_auroc"),
            "features": self.metrics.get("features"),
            "updated_at": self.built_at.isoformat(),
        })

    # -- patients ---------------------------------------------------------
    def _row(self, patient_id: str) -> pd.Series:
        pid = patient_id.strip().upper()
        if pid not in self.df.index:
            raise NotFound(f"No patient with id {patient_id!r}")
        return self.df.loc[pid]

    @staticmethod
    def _public(row: pd.Series) -> dict:
        d = row.drop("readmitted_30d").to_dict()  # never expose the outcome label
        d["medications"] = [m for m in str(d["medications"] or "").split(";") if m]
        return _clean({k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()})

    def get_patient(self, patient_id: str) -> dict:
        return self._public(self._row(patient_id))

    def list_patients(
        self,
        risk_band: str | None = None,
        condition: str | None = None,
        min_age: int | None = None,
        limit: int = 20,
    ) -> list[dict]:
        df = self.df
        if risk_band:
            df = df[df["risk_band"] == risk_band]
        if condition:
            if condition not in CONDITIONS:
                raise ValueError(f"condition must be one of {', '.join(CONDITIONS)}")
            df = df[df[condition] == 1]
        if min_age is not None:
            df = df[df["age"] >= min_age]
        cols = ["patient_id", "age", "sex", "charlson_index", "prior_admissions_12m",
                "discharge_disposition", "predicted_risk", "risk_band"]
        top = df.sort_values("predicted_risk", ascending=False).head(max(1, min(limit, 200)))
        return _clean(top[cols].to_dict("records"))

    # -- risk ---------------------------------------------------------------
    def readmission_risk(self, patient_id: str) -> dict:
        row = self._row(patient_id)
        p = float(row["predicted_risk"])
        # Capped so the top patients don't round up to "higher than 100%".
        percentile = min(round(float((self.df["predicted_risk"] < p).mean() * 100), 1), 99.9)
        return {
            "patient_id": row["patient_id"],
            "predicted_30d_readmission_risk": p,
            "risk_band": risk_band(p),
            "cohort_percentile": percentile,
            "top_drivers": self.model.explain(row.to_frame().T),
            "note": "Model output; decision support only, not a diagnosis.",
        }

    # -- medications --------------------------------------------------------
    def check_interactions(self, medications: list[str]) -> list[dict]:
        meds = sorted({m.strip().lower() for m in medications if m.strip()})
        found = []
        for i, a in enumerate(meds):
            for b in meds[i + 1 :]:
                hit = INTERACTIONS.get(frozenset({a, b}))
                if hit:
                    found.append({"drugs": [a, b], "severity": hit[0], "effect": hit[1]})
        return sorted(found, key=lambda f: f["severity"] != "major")

    def patient_interactions(self, patient_id: str) -> dict:
        meds = self.get_patient(patient_id)["medications"]
        return {"patient_id": patient_id.strip().upper(), "medications": meds,
                "interactions": self.check_interactions(meds)}

    # -- cohort -------------------------------------------------------------
    def cohort_summary(self, group_by: str | None = None) -> dict:
        if group_by and group_by not in GROUPABLE:
            raise ValueError(f"group_by must be one of {', '.join(GROUPABLE)}")
        if group_by and self.df[group_by].isna().all():
            raise ValueError(f"{group_by} is not recorded in this data source")
        agg = dict(
            patients=("patient_id", "count"),
            mean_predicted_risk=("predicted_risk", "mean"),
            observed_readmission_rate=("readmitted_30d", "mean"),
            mean_age=("age", "mean"),
            mean_length_of_stay=("length_of_stay", "mean"),
        )
        if group_by:
            table = self.df.groupby(group_by).agg(**agg).round(3).reset_index().to_dict("records")
        else:
            table = [self.df.assign(_all="all").groupby("_all").agg(**agg).round(3).iloc[0].to_dict()]
        return _clean({
            "group_by": group_by,
            "rows": table,
            "risk_band_counts": self.df["risk_band"].value_counts().to_dict(),
        })


class RefreshingService:
    """Rebuilds a live cohort every `ttl` seconds. If a rebuild fails, it keeps serving the
    last good snapshot rather than failing the request."""

    def __init__(self, build, ttl: float):
        self._build = build
        self._ttl = ttl
        self._svc: ClinicalService | None = None
        self._built = 0.0
        self._lock = threading.Lock()

    def _current(self) -> ClinicalService:
        with self._lock:
            stale = time.time() - self._built > self._ttl
            if self._svc is None or stale:
                try:
                    self._svc = self._build()
                except Exception:
                    if self._svc is None:
                        raise
                    log.exception("Live refresh failed; serving the previous snapshot")
                self._built = time.time()
            return self._svc

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._current(), name)


def _artifact_dir(source: str) -> Path:
    return ARTIFACT_DIR if source == "synthetic" else ARTIFACT_DIR / source


def _load_static(source: str) -> ClinicalService:
    d = _artifact_dir(source)
    cohort_path, model_path, metrics_path = d / "cohort.csv", d / "model.pt", d / "metrics.json"
    if not (cohort_path.exists() and model_path.exists()):
        raise RuntimeError(
            f"Model artifacts missing in {d}. Run `python -m caresignal.model --dataset "
            f"{'uci' if source == 'uci' else 'synthetic'}` first."
        )
    metrics = json.loads(metrics_path.read_text())
    return ClinicalService(
        pd.read_csv(cohort_path, low_memory=False),
        RiskModel.load(model_path),
        metrics=metrics,
        source=source,
    )


def _build_fhir() -> ClinicalService:
    from .fhir import FhirClient, discharge_cohort

    base = os.environ.get("CARESIGNAL_FHIR_BASE")
    if not base:
        raise RuntimeError("Set CARESIGNAL_FHIR_BASE to the hospital's FHIR R4 base URL.")
    client = FhirClient(base, os.environ.get("CARESIGNAL_FHIR_TOKEN") or None)
    cohort = discharge_cohort(client, max_patients=int(os.environ.get("CARESIGNAL_FHIR_MAX_PATIENTS", "50")))
    d = _artifact_dir("uci")  # the model is trained on the UCI cohort, whose features match
    return ClinicalService(
        cohort,
        RiskModel.load(d / "model.pt"),
        metrics=json.loads((d / "metrics.json").read_text()),
        source="fhir",
        data_label=f"Live FHIR feed ({urlsplit(base).netloc})",
    )


@lru_cache(maxsize=1)
def get_service():
    source = os.environ.get("CARESIGNAL_SOURCE", "synthetic").lower()
    if source == "fhir":
        return RefreshingService(_build_fhir, float(os.environ.get("CARESIGNAL_FHIR_TTL", "900")))
    if source not in ("synthetic", "uci"):
        raise ValueError(f"Unknown CARESIGNAL_SOURCE {source!r}; use synthetic, uci or fhir.")
    return _load_static(source)
