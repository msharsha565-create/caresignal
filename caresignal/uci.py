"""Real de-identified hospital encounters: the UCI "Diabetes 130-US hospitals" dataset.

101,766 inpatient diabetes encounters from 130 US hospitals, 1999-2008, de-identified by
its authors and published for research (Strack et al., 2014). Each row is one encounter;
readmitted_30d is 1 when the patient came back within 30 days.

The dataset is coarser than a live record: ages are decade brackets (we use the
midpoint), HbA1c is a three-level category, and creatinine, blood pressure, BMI and
living situation are not recorded. See README "Real data" for the full mapping.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .model import ARTIFACT_DIR, REAL_FEATURES, complete_schema, train_frame

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
DATA_DIR = ARTIFACT_DIR / "data"
UCI_DIR = ARTIFACT_DIR / "uci"

AGE_MIDPOINT = {f"[{lo}-{lo + 10})": lo + 5 for lo in range(0, 100, 10)}
A1C_MIDPOINT = {"Norm": 5.5, ">7": 7.5, ">8": 9.0}  # category -> approximate HbA1c %; "None" = not measured
# Discharged to hospice or died: these patients cannot be readmitted, so they are excluded.
EXCLUDED_DISPOSITIONS = [11, 13, 14, 19, 20, 21]
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]


def fetch(dest: Path = DATA_DIR) -> Path:
    """Download the dataset once and return the path of diabetic_data.csv."""
    csv_path = dest / "diabetic_data.csv"
    if csv_path.exists():
        return csv_path
    dest.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(UCI_URL, follow_redirects=True, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extract("diabetic_data.csv", dest)  # named member only
    return csv_path


# ICD-9 diagnosis codes (this dataset's coding system).
def _is_diabetes(code: str) -> bool:
    return code.startswith("250")


def _is_chf(code: str) -> bool:
    return code.startswith("428")


def _is_ckd(code: str) -> bool:
    return code.startswith("585")


def _is_copd(code: str) -> bool:
    try:
        return 490 <= float(code) < 497
    except ValueError:
        return False


def _any_diag(raw: pd.DataFrame, pred) -> np.ndarray:
    hits = [raw[c].fillna("").astype(str).map(pred).to_numpy() for c in DIAG_COLUMNS]
    return np.logical_or.reduce(hits)


def load_uci(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        csv_path, na_values=["?"], dtype={c: str for c in DIAG_COLUMNS}, low_memory=False
    )
    keep = raw["gender"].isin(["Male", "Female"]) & ~raw["discharge_disposition_id"].isin(EXCLUDED_DISPOSITIONS)
    raw = raw[keep].reset_index(drop=True)

    disp = raw["discharge_disposition_id"]
    taken = raw[DRUG_COLUMNS].isin(["Steady", "Up", "Down"]).to_numpy()
    drug_names = np.array(DRUG_COLUMNS)

    df = pd.DataFrame(
        {
            "patient_id": "E" + raw["encounter_id"].astype(str),
            "member": raw["patient_nbr"].astype(str),  # used only to split by patient, then dropped
            "age": raw["age"].map(AGE_MIDPOINT).astype(float),
            "sex": raw["gender"].map({"Male": "M", "Female": "F"}),
            "length_of_stay": raw["time_in_hospital"].astype(float),
            "prior_admissions_12m": raw["number_inpatient"].astype(float),
            "num_medications": raw["num_medications"].astype(float),
            "num_diagnoses": raw["number_diagnoses"].astype(float),
            "hba1c": raw["A1Cresult"].map(A1C_MIDPOINT).astype(float),
            "chf": _any_diag(raw, _is_chf).astype(int),
            "copd": _any_diag(raw, _is_copd).astype(int),
            "diabetes": _any_diag(raw, _is_diabetes).astype(int),
            "ckd": _any_diag(raw, _is_ckd).astype(int),
            "discharge_disposition": np.select(
                [disp.eq(3), disp.eq(6), disp.eq(1)], ["snf", "home_health", "home"], "other"
            ),
            "medications": [";".join(drug_names[row]) for row in taken],
            "readmitted_30d": (raw["readmitted"] == "<30").astype(int),
        }
    )
    return complete_schema(df)


def split_by_patient(members: np.ndarray, seed: int = 7) -> np.ndarray:
    """Assign whole patients to train/val/test, so no patient appears in two splits."""
    rng = np.random.default_rng(seed)
    unique = pd.unique(members)
    labels = rng.choice(["train", "val", "test"], len(unique), p=[0.7, 0.15, 0.15])
    return pd.Series(members).map(dict(zip(unique, labels))).to_numpy()


def build_uci_artifacts(out_dir: Path = UCI_DIR, seed: int = 7, epochs: int = 40) -> dict:
    df = load_uci(fetch())
    members = df.pop("member").to_numpy()
    split = split_by_patient(members, seed)
    return train_frame(
        df, split, REAL_FEATURES, out_dir, epochs=epochs, seed=seed,
        metadata={
            "dataset": "uci",
            "data_label": "De-identified inpatient encounters: UCI Diabetes 130-US hospitals, 1999-2008",
            "n_patients": int(len(df)),
            "readmission_rate": round(float(df["readmitted_30d"].mean()), 4),
        },
    )
