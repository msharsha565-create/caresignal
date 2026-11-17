"""Real-data paths: UCI loader, FHIR connector, live refresh, audit log, Gemini guard."""
import json
import logging
import os

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from caresignal import fhir, uci
from caresignal.model import ARTIFACT_DIR, REAL_FEATURES, prepare
from caresignal.service import RefreshingService, _load_static, get_service

UCI_CSV = ARTIFACT_DIR / "data" / "diabetic_data.csv"
UCI_MODEL = ARTIFACT_DIR / "uci" / "model.pt"
needs_uci = pytest.mark.skipif(not UCI_CSV.exists(), reason="run: python -m caresignal.model --dataset uci")


@needs_uci
def test_uci_loader_maps_real_encounters():
    df = uci.load_uci(UCI_CSV)
    assert len(df) > 90_000
    assert set(REAL_FEATURES) <= set(prepare(df).columns)  # derived columns added by prepare()
    assert 0.09 < df["readmitted_30d"].mean() < 0.13
    assert df["sex"].isin(["M", "F"]).all()
    assert df["hba1c"].isna().mean() > 0.5  # most encounters have no HbA1c measured
    assert df["discharge_disposition"].isin(["snf", "home_health", "home", "other"]).all()


@needs_uci
def test_uci_model_generalises_to_held_out_patients():
    svc = _load_static("uci")
    assert svc.metrics["test_auroc"] > 0.60
    assert svc.info()["n_patients"] > 90_000
    assert set(svc.metrics["features"]) == set(REAL_FEATURES)


def _fake_fhir(request: httpx.Request) -> httpx.Response:
    path, q = request.url.path, dict(request.url.params)

    def bundle(*resources):
        return httpx.Response(200, json={"resourceType": "Bundle",
                                         "entry": [{"resource": r} for r in resources]})

    if path.endswith("/Encounter"):
        return bundle(
            {"resourceType": "Encounter", "id": "e1", "subject": {"reference": "Patient/p1"},
             "period": {"start": "2024-01-01T08:00:00Z", "end": "2024-01-06T12:00:00Z"},
             "hospitalization": {"dischargeDisposition": {"coding": [{"code": "03"}]}}},
            {"resourceType": "Encounter", "id": "e2", "subject": {"reference": "Patient/p1"},
             "period": {"start": "2024-01-20T08:00:00Z", "end": "2024-01-22T12:00:00Z"}},
            {"resourceType": "Encounter", "id": "e3", "subject": {"reference": "Patient/p2"},
             "period": {"start": "2024-02-01T08:00:00Z", "end": "2024-02-03T10:00:00Z"}},
        )
    if path.endswith("/Patient/p1"):
        return httpx.Response(200, json={"resourceType": "Patient", "id": "p1", "gender": "female", "birthDate": "1940-03-02"})
    if path.endswith("/Patient/p2"):
        return httpx.Response(200, json={"resourceType": "Patient", "id": "p2", "gender": "male", "birthDate": "1970-01-01"})
    if path.endswith("/Condition"):
        if q["patient"] == "p1":
            return bundle({"resourceType": "Condition", "code": {"coding": [{"code": "I50.9"}]}},
                          {"resourceType": "Condition", "code": {"coding": [{"code": "E11.9"}]}})
        return bundle({"resourceType": "Condition", "code": {"coding": [{"code": "J44.9"}]}})
    if path.endswith("/MedicationRequest"):
        meds = [{"resourceType": "MedicationRequest", "medicationCodeableConcept": {"text": "furosemide"}}] * 3
        return bundle(*(meds if q["patient"] == "p1" else []))
    if path.endswith("/Observation"):
        if q["patient"] == "p1":
            return bundle({"resourceType": "Observation", "valueQuantity": {"value": 8.2}})
        return bundle()
    return httpx.Response(404)


def test_fhir_cohort_builds_discharges_and_outcomes():
    client = fhir.FhirClient("https://fhir.test/fhir", transport=httpx.MockTransport(_fake_fhir))
    df = fhir.discharge_cohort(client, max_patients=10).set_index("patient_id")
    assert len(df) == 3
    assert df.loc["FE1", "readmitted_30d"] == 1      # back in 14 days
    assert np.isnan(df.loc["FE2", "readmitted_30d"])  # no later admission on record
    assert df.loc["FE2", "prior_admissions_12m"] == 1
    assert df.loc["FE1", "discharge_disposition"] == "snf"
    assert df.loc["FE1", "chf"] == 1 and df.loc["FE1", "diabetes"] == 1 and df.loc["FE1", "copd"] == 0
    assert df.loc["FE1", "hba1c"] == 8.2
    assert df.loc["FE1", "age"] == 83 and df.loc["FE1", "num_medications"] == 3
    assert df.loc["FE1", "medications"] == ""  # names come from the drug list, not counts


def test_fhir_client_never_follows_links_to_other_hosts():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"resourceType": "Bundle", "entry": [],
                                         "link": [{"relation": "next", "url": "https://evil.example/x"}]})

    client = fhir.FhirClient("https://fhir.test/fhir", token="secret", transport=httpx.MockTransport(handler))
    assert client.search("Encounter") == []
    assert len(seen) == 1  # the next-page link to another host was not requested


@pytest.mark.skipif(os.environ.get("CARESIGNAL_LIVE_TESTS") != "1", reason="set CARESIGNAL_LIVE_TESTS=1")
def test_live_public_fhir_server_responds():
    client = fhir.FhirClient("https://hapi.fhir.org/baseR4")
    assert client.get("metadata")["resourceType"] == "CapabilityStatement"


def test_refreshing_service_keeps_last_good_snapshot():
    class Snapshot:
        value = "good"

    state = {"n": 0}

    def build():
        state["n"] += 1
        if state["n"] == 1:
            return Snapshot()
        raise RuntimeError("hospital feed down")

    svc = RefreshingService(build, ttl=0)  # always stale, so every access tries a rebuild
    assert svc.value == "good"
    assert svc.value == "good"  # rebuild failed; previous snapshot still served
    assert state["n"] == 2


def test_patient_views_are_audited(caplog):
    from caresignal.api import app

    pid = get_service().list_patients(limit=1)[0]["patient_id"]
    with caplog.at_level(logging.INFO, logger="caresignal.audit"):
        with TestClient(app) as client:
            assert client.get(f"/api/patients/{pid}").status_code == 200
    records = [r for r in caplog.records if r.name == "caresignal.audit"]
    event = json.loads(records[-1].getMessage())
    assert event["event"] == "patient_record_view" and event["status"] == 200


def test_gemini_is_refused_for_live_patient_data(monkeypatch):
    from caresignal.api import app

    monkeypatch.setenv("CARESIGNAL_SOURCE", "fhir")
    monkeypatch.setenv("CARESIGNAL_PROVIDER", "gemini")
    monkeypatch.delenv("CARESIGNAL_ALLOW_FREE_TIER_LLM", raising=False)
    with pytest.raises(RuntimeError, match="Gemini free tier"):
        with TestClient(app):
            pass
