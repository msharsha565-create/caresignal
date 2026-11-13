"""Synthetic inpatient discharge cohort.

Everything here is generated from a seeded random process. No real patient
data is used anywhere in this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DISPOSITIONS = ["home", "home_health", "snf"]

# Medications a synthetic patient is likely to be on, keyed by condition.
CONDITION_MEDS = {
    "chf": ["furosemide", "lisinopril", "metoprolol", "spironolactone", "digoxin"],
    "copd": ["tiotropium", "albuterol", "prednisone"],
    "diabetes": ["metformin", "insulin glargine", "empagliflozin"],
    "ckd": ["sevelamer", "furosemide"],
}
BACKGROUND_MEDS = [
    "atorvastatin", "simvastatin", "aspirin", "warfarin", "ibuprofen",
    "omeprazole", "clopidogrel", "sertraline", "tramadol", "amiodarone",
    "acetaminophen", "amlodipine",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_cohort(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Generate `n` synthetic discharges with a 30-day readmission label."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(66, 14, n), 18, 98).round()
    sex = rng.choice(["F", "M"], n)
    chf = rng.random(n) < _sigmoid((age - 75) / 9)
    copd = rng.random(n) < 0.18
    diabetes = rng.random(n) < 0.30
    ckd = rng.random(n) < _sigmoid((age - 80) / 10) * 0.6 + diabetes * 0.08
    charlson = np.clip(
        rng.poisson(1.2, n) + chf + copd + diabetes + 2 * ckd + (age > 70), 0, 12
    )
    prior = rng.poisson(0.5 + 0.25 * charlson)
    los = np.clip(rng.gamma(2.0, 2.2, n) + 0.4 * charlson, 1, 45).round()
    hba1c = np.where(diabetes, rng.normal(8.1, 1.5, n), rng.normal(5.5, 0.4, n)).round(1)
    creat = np.where(ckd, rng.normal(2.3, 0.7, n), rng.normal(0.95, 0.2, n)).clip(0.4).round(2)
    sbp = rng.normal(132, 18, n).round()
    bmi = rng.normal(28.5, 6, n).clip(15, 60).round(1)
    lives_alone = rng.random(n) < 0.27
    disposition = rng.choice(DISPOSITIONS, n, p=[0.68, 0.19, 0.13])

    meds: list[str] = []
    for i in range(n):
        m: set[str] = set()
        for cond, flag in (("chf", chf[i]), ("copd", copd[i]), ("diabetes", diabetes[i]), ("ckd", ckd[i])):
            if flag:
                pool = CONDITION_MEDS[cond]
                m.update(rng.choice(pool, size=rng.integers(1, len(pool) + 1), replace=False))
        m.update(rng.choice(BACKGROUND_MEDS, size=rng.integers(1, 6), replace=False))
        meds.append(";".join(sorted(m)))
    num_meds = np.array([len(m.split(";")) for m in meds])

    # Ground-truth risk process (the model never sees these coefficients).
    z = (
        -3.55
        + 0.018 * (age - 65)
        + 0.42 * prior
        + 0.035 * los
        + 0.16 * charlson
        + 0.55 * chf
        + 0.30 * copd
        + 0.22 * np.clip(hba1c - 7.0, 0, None)
        + 0.30 * np.clip(creat - 1.3, 0, None)
        + 0.045 * np.clip(num_meds - 5, 0, None)
        + 0.40 * lives_alone
        + 0.35 * (disposition == "snf")
        - 0.25 * (disposition == "home_health")
        + 0.5 * (chf & lives_alone)  # interaction a linear model will miss
    )
    readmitted = rng.random(n) < _sigmoid(z)

    df = pd.DataFrame(
        {
            "patient_id": [f"P{100000 + i}" for i in range(n)],
            "age": age.astype(int),
            "sex": sex,
            "length_of_stay": los.astype(int),
            "prior_admissions_12m": prior,
            "num_medications": num_meds,
            "charlson_index": charlson.astype(int),
            "hba1c": hba1c,
            "creatinine": creat,
            "systolic_bp": sbp.astype(int),
            "bmi": bmi,
            "chf": chf.astype(int),
            "copd": copd.astype(int),
            "diabetes": diabetes.astype(int),
            "ckd": ckd.astype(int),
            "lives_alone": lives_alone.astype(int),
            "discharge_disposition": disposition,
            "medications": meds,
            "readmitted_30d": readmitted.astype(int),
        }
    )
    # Labs are not always drawn; leave realistic gaps for the pipeline to impute.
    df.loc[rng.random(n) < 0.12, "hba1c"] = np.nan
    df.loc[rng.random(n) < 0.05, "creatinine"] = np.nan
    return df
