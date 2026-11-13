"""Clinical data service shared by the MCP server and the REST API."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .model import ARTIFACT_DIR, RiskModel

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


def risk_band(p: float) -> str:
    return next(name for cutoff, name in RISK_BANDS if p >= cutoff)


class NotFound(LookupError):
    pass


class ClinicalService:
    def __init__(self, cohort: pd.DataFrame, model: RiskModel):
        self.model = model
        self.df = cohort.set_index("patient_id", drop=False)
        self.df["predicted_risk"] = model.predict(self.df).round(4)
        self.df["risk_band"] = self.df["predicted_risk"].map(risk_band)

    # -- patients ---------------------------------------------------------
    def _row(self, patient_id: str) -> pd.Series:
        pid = patient_id.strip().upper()
        if pid not in self.df.index:
            raise NotFound(f"No patient with id {patient_id!r}")
        return self.df.loc[pid]

    @staticmethod
    def _public(row: pd.Series) -> dict:
        d = row.drop("readmitted_30d").to_dict()  # never leak the label
        d["medications"] = d["medications"].split(";")
        return {k: (None if isinstance(v, float) and pd.isna(v) else getattr(v, "item", lambda: v)()) for k, v in d.items()}

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
            if condition not in ("chf", "copd", "diabetes", "ckd"):
                raise ValueError("condition must be one of chf, copd, diabetes, ckd")
            df = df[df[condition] == 1]
        if min_age is not None:
            df = df[df["age"] >= min_age]
        cols = ["patient_id", "age", "sex", "charlson_index", "prior_admissions_12m",
                "discharge_disposition", "predicted_risk", "risk_band"]
        top = df.sort_values("predicted_risk", ascending=False).head(max(1, min(limit, 200)))
        return top[cols].to_dict("records")

    # -- risk ---------------------------------------------------------------
    def readmission_risk(self, patient_id: str) -> dict:
        row = self._row(patient_id)
        p = float(row["predicted_risk"])
        return {
            "patient_id": row["patient_id"],
            "predicted_30d_readmission_risk": p,
            "risk_band": risk_band(p),
            # Capped so the top few patients don't round up to "higher than 100%".
            "cohort_percentile": min(round(float((self.df["predicted_risk"] < p).mean() * 100), 1), 99.9),
            "top_drivers": self.model.explain(row.to_frame().T),
            "note": "Model output on synthetic data; decision support only, not a diagnosis.",
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
        allowed = ("risk_band", "discharge_disposition", "sex", "chf", "copd", "diabetes", "ckd", "lives_alone")
        if group_by and group_by not in allowed:
            raise ValueError(f"group_by must be one of {', '.join(allowed)}")
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
        return {"group_by": group_by, "rows": table,
                "risk_band_counts": self.df["risk_band"].value_counts().to_dict()}


@lru_cache(maxsize=1)
def get_service() -> ClinicalService:
    cohort_path, model_path = ARTIFACT_DIR / "cohort.csv", ARTIFACT_DIR / "model.pt"
    if not (cohort_path.exists() and model_path.exists()):
        raise RuntimeError("Model artifacts missing. Run `python -m caresignal.model` first.")
    return ClinicalService(pd.read_csv(cohort_path), RiskModel.load(model_path))
