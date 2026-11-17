"""Live FHIR R4 connector: builds the discharge cohort from a hospital's record system.

Reads standard FHIR R4 resources (Encounter, Patient, Condition, MedicationRequest,
Observation) from any compliant server. Hospitals on Epic, Oracle Health and similar
systems expose this through SMART on FHIR; the access token belongs in Secret Manager,
never in code or the repository.

Each inpatient encounter becomes one row with the same columns as the UCI cohort, so the
model trained on that data scores live patients directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import complete_schema

HBA1C_LOINC = "4548-4"
SNF_CODES = {"03"}  # UB-04 patient discharge status codes
HOME_HEALTH_CODES = {"06"}
HOME_CODES = {"01"}
READMIT_WINDOW_DAYS = 30
PRIOR_WINDOW_DAYS = 365
ICD10 = {
    "chf": ("I50",),
    "copd": ("J43", "J44"),
    "diabetes": ("E08", "E09", "E10", "E11", "E13"),
    "ckd": ("N18",),
}
COLUMNS = [
    "patient_id", "sex", "age", "length_of_stay", "prior_admissions_12m", "num_medications",
    "num_diagnoses", "hba1c", "chf", "copd", "diabetes", "ckd", "discharge_disposition",
    "medications", "readmitted_30d",
]


class FhirClient:
    """Minimal FHIR R4 client. It follows paging links only on the same server, so the
    access token is never sent to another host."""

    def __init__(self, base_url: str, token: str | None = None, *, transport=None, timeout: float = 30.0):
        import httpx

        headers = {"Accept": "application/fhir+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._root = base_url.rstrip("/") + "/"
        self._http = httpx.Client(base_url=self._root, headers=headers, timeout=timeout, transport=transport)

    def get(self, path: str, **params) -> dict:
        resp = self._http.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def search(self, resource: str, *, max_pages: int = 20, **params) -> list[dict]:
        bundle = self.get(resource, **params)
        out: list[dict] = []
        for _ in range(max_pages):
            out += [e["resource"] for e in bundle.get("entry", []) if "resource" in e]
            nxt = next((l["url"] for l in bundle.get("link", []) if l.get("relation") == "next"), None)
            if not nxt or not nxt.startswith(self._root):
                break
            resp = self._http.get(nxt)
            resp.raise_for_status()
            bundle = resp.json()
        return out


def _ref_id(ref: str | None) -> str | None:
    return ref.split("/")[-1] if ref else None


def _ts(text: str | None) -> pd.Timestamp | None:
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def _age(birth: str | None, at: pd.Timestamp) -> float:
    b = pd.to_datetime(birth, errors="coerce") if birth else pd.NaT
    if pd.isna(b):
        return np.nan
    return float(np.floor((at.tz_localize(None) - b).days / 365.25))


def _codes(conditions: list[dict]) -> list[str]:
    return [
        coding.get("code", "").upper()
        for cond in conditions
        for coding in cond.get("code", {}).get("coding", [])
        if coding.get("code")
    ]


def _has(codes: list[str], prefixes: tuple[str, ...]) -> int:
    return int(any(c.startswith(p) for c in codes for p in prefixes))


def _disposition(enc: dict) -> str:
    coding = (enc.get("hospitalization", {}).get("dischargeDisposition") or {}).get("coding", [])
    code = coding[0].get("code", "") if coding else ""
    if code in SNF_CODES:
        return "snf"
    if code in HOME_HEALTH_CODES:
        return "home_health"
    if code in HOME_CODES:
        return "home"
    return "other"


def _latest_hba1c(client: FhirClient, pid: str) -> float:
    for obs in client.search("Observation", max_pages=1, patient=pid, code=HBA1C_LOINC, _sort="-date", _count=1):
        value = obs.get("valueQuantity", {}).get("value")
        if value is not None:
            return float(value)
    return np.nan


def discharge_cohort(client: FhirClient, max_patients: int = 50) -> pd.DataFrame:
    """One row per finished inpatient encounter that has a discharge time.

    readmitted_30d is 1 if the patient's next admission started within 30 days of this
    discharge, 0 if it started later, and unknown (NaN) if no later admission is on record.
    """
    encounters = client.search("Encounter", **{"class": "IMP", "status": "finished", "_count": 100})
    by_patient: dict[str, list[dict]] = {}
    for enc in encounters:
        pid = _ref_id(enc.get("subject", {}).get("reference"))
        if pid and _ts(enc.get("period", {}).get("start")) is not None:
            by_patient.setdefault(pid, []).append(enc)

    rows = []
    for pid, encs in list(by_patient.items())[:max_patients]:
        encs.sort(key=lambda e: _ts(e["period"]["start"]))
        starts = [_ts(e["period"]["start"]) for e in encs]
        ends = [_ts(e.get("period", {}).get("end")) for e in encs]
        patient = client.get(f"Patient/{pid}")
        codes = _codes(client.search("Condition", patient=pid))
        n_meds = len(client.search("MedicationRequest", patient=pid, status="active"))
        hba1c = _latest_hba1c(client, pid)

        for i, enc in enumerate(encs):
            start, end = starts[i], ends[i]
            if end is None:
                continue  # not discharged yet, so there is no outcome to learn from
            prior = sum(
                1 for j, e_end in enumerate(ends)
                if j != i and e_end is not None and e_end < start
                and (start - e_end).days <= PRIOR_WINDOW_DAYS
            )
            if i + 1 < len(encs):
                gap_days = (starts[i + 1] - end).total_seconds() / 86400
                label = float(0 <= gap_days <= READMIT_WINDOW_DAYS)
            else:
                label = np.nan
            rows.append({
                "patient_id": f"F{enc['id']}".upper(),
                "sex": {"male": "M", "female": "F"}.get(patient.get("gender", ""), None),
                "age": _age(patient.get("birthDate"), start),
                "length_of_stay": round((end - start).total_seconds() / 86400, 2),
                "prior_admissions_12m": float(prior),
                "num_medications": float(n_meds),
                "num_diagnoses": float(len(codes)),
                "hba1c": hba1c,
                "chf": _has(codes, ICD10["chf"]),
                "copd": _has(codes, ICD10["copd"]),
                "diabetes": _has(codes, ICD10["diabetes"]),
                "ckd": _has(codes, ICD10["ckd"]),
                "discharge_disposition": _disposition(enc),
                "medications": "",
                "readmitted_30d": label,
            })

    if not rows:
        return complete_schema(pd.DataFrame(columns=COLUMNS))
    return complete_schema(pd.DataFrame(rows))
