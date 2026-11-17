"""PyTorch 30-day readmission risk model: features, training, inference.

One network, several cohorts. The synthetic demo records different things from the
real-data cohorts (UCI research data and a live FHIR feed), so every model artifact
stores the feature list it was trained on.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import generate_cohort

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("CARESIGNAL_ARTIFACTS", ROOT / "artifacts"))

# Synthetic demo cohort (see data.py).
NUMERIC = [
    "age", "length_of_stay", "prior_admissions_12m", "num_medications",
    "charlson_index", "hba1c", "creatinine", "systolic_bp", "bmi",
]
BINARY = ["chf", "copd", "diabetes", "ckd", "lives_alone"]
FEATURES = NUMERIC + BINARY + ["sex_male", "discharged_snf", "discharged_home_health"]

# What the real-data cohorts (uci.py, fhir.py) can measure.
REAL_FEATURES = [
    "age", "length_of_stay", "prior_admissions_12m", "num_medications", "num_diagnoses",
    "hba1c", "chf", "copd", "diabetes", "ckd", "sex_male", "discharged_snf", "discharged_home_health",
]
# Display columns the real-data cohorts do not record. Kept (empty) so the API shape is stable.
NOT_RECORDED = ["creatinine", "systolic_bp", "bmi", "lives_alone", "charlson_index"]

FEATURE_LABELS = {
    "age": "Age",
    "length_of_stay": "Length of stay",
    "prior_admissions_12m": "Admissions in prior 12 months",
    "num_medications": "Number of medications",
    "num_diagnoses": "Number of diagnoses",
    "charlson_index": "Charlson comorbidity index",
    "hba1c": "HbA1c",
    "creatinine": "Creatinine",
    "systolic_bp": "Systolic blood pressure",
    "bmi": "BMI",
    "chf": "Heart failure",
    "copd": "COPD",
    "diabetes": "Diabetes",
    "ckd": "Chronic kidney disease",
    "lives_alone": "Lives alone",
    "sex_male": "Male sex",
    "discharged_snf": "Discharged to skilled nursing",
    "discharged_home_health": "Discharged with home health",
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns every cohort shares."""
    x = df.copy()
    x["sex_male"] = (x["sex"] == "M").astype(int)
    x["discharged_snf"] = (x["discharge_disposition"] == "snf").astype(int)
    x["discharged_home_health"] = (x["discharge_disposition"] == "home_health").astype(int)
    return x


def complete_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in display columns this data source does not record."""
    x = df.copy()
    for col in NOT_RECORDED:
        if col not in x:
            x[col] = np.nan
    return x


def compute_medians(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    x = prepare(df)
    return {f: float(x[f].median()) for f in features}


def build_features(df: pd.DataFrame, medians: dict[str, float], features: list[str]) -> pd.DataFrame:
    """Derive, impute with training medians, and select the model's inputs."""
    x = prepare(df).fillna(value=medians)
    return x[list(features)].astype("float32")


class RiskNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties averaged."""
    ranks = pd.Series(p).rank().to_numpy()
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class RiskModel:
    """Trained network plus the preprocessing needed to score raw rows."""

    net: RiskNet
    features: list[str]
    medians: dict[str, float]
    mean: np.ndarray
    std: np.ndarray

    def _x(self, df: pd.DataFrame) -> torch.Tensor:
        feats = build_features(df, self.medians, self.features).to_numpy()
        return torch.from_numpy((feats - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        return torch.sigmoid(self.net(self._x(df))).numpy()

    @torch.no_grad()
    def explain(self, row: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """Occlusion attribution: how much does risk move when one feature is
        replaced by the cohort average? Positive = feature raises this patient's risk."""
        self.net.eval()
        x = self._x(row)
        base = torch.sigmoid(self.net(x)).item()
        n = len(self.features)
        occluded = x.repeat(n, 1)
        occluded[torch.arange(n), torch.arange(n)] = 0.0
        delta = base - torch.sigmoid(self.net(occluded)).numpy()
        raw = build_features(row, self.medians, self.features).iloc[0]
        order = np.argsort(-np.abs(delta))[:top_k]
        return [
            {
                "feature": self.features[i],
                "label": FEATURE_LABELS.get(self.features[i], self.features[i]),
                "value": round(float(raw.iloc[i]), 2),
                "risk_contribution": round(float(delta[i]), 4),
            }
            for i in order
        ]

    def save(self, path: Path) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "features": self.features,
                "medians": self.medians,
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "RiskModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        features = list(ckpt.get("features", FEATURES))
        net = RiskNet(len(features))
        net.load_state_dict(ckpt["state_dict"])
        return cls(
            net,
            features,
            ckpt["medians"],
            np.array(ckpt["mean"], dtype="float32"),
            np.array(ckpt["std"], dtype="float32"),
        )


def train_frame(
    df: pd.DataFrame,
    split: np.ndarray,
    features: list[str],
    out_dir: Path,
    *,
    epochs: int = 60,
    seed: int = 7,
    metadata: dict | None = None,
) -> dict:
    """Train on the 'train' rows, early-stop on 'val' AUROC, report 'test' metrics,
    and write cohort.csv, model.pt, predictions.csv and metrics.json to `out_dir`."""
    torch.manual_seed(seed)
    df = df.reset_index(drop=True)
    split = np.asarray(split)
    tr, val, test = df[split == "train"], df[split == "val"], df[split == "test"]

    medians = compute_medians(tr, features)
    f_tr = build_features(tr, medians, features).to_numpy()
    mean, std = f_tr.mean(0), f_tr.std(0) + 1e-6
    model = RiskModel(RiskNet(len(features)), list(features), medians, mean, std)

    x_tr = model._x(tr)
    y_tr = torch.tensor(tr["readmitted_30d"].to_numpy(), dtype=torch.float32)
    y_val = val["readmitted_30d"].to_numpy()

    opt = torch.optim.AdamW(model.net.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0.0, None
    for _ in range(epochs):
        model.net.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(perm), 128):
            idx = perm[i : i + 128]
            opt.zero_grad()
            loss_fn(model.net(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
        val_auc = auroc(y_val, model.predict(val))
        if val_auc > best_auc:  # early stopping on validation AUROC
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.net.state_dict().items()}
    model.net.load_state_dict(best_state)

    p_test = model.predict(test)
    y_test = test["readmitted_30d"].to_numpy()
    metrics = {
        **(metadata or {}),
        "features": list(features),
        "n_test": int(len(test)),
        "val_auroc": round(best_auc, 4),
        "test_auroc": round(auroc(y_test, p_test), 4),
        "test_brier": round(float(np.mean((p_test - y_test) ** 2)), 4),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cohort.csv", index=False)
    model.save(out_dir / "model.pt")
    pd.DataFrame(
        {"patient_id": df["patient_id"], "split": split, "predicted_risk": model.predict(df).round(5)}
    ).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train(n: int = 6000, epochs: int = 60, seed: int = 7, out_dir: Path = ARTIFACT_DIR) -> dict:
    """Synthetic demo cohort (see data.py)."""
    df = generate_cohort(n, seed)
    rng = np.random.default_rng(seed)
    split = rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15])
    return train_frame(
        df, split, FEATURES, out_dir, epochs=epochs, seed=seed,
        metadata={
            "dataset": "synthetic",
            "data_label": "Synthetic demo data",
            "n_patients": n,
            "readmission_rate": round(float(df["readmitted_30d"].mean()), 4),
        },
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the CareSignal readmission model.")
    ap.add_argument("--dataset", choices=["synthetic", "uci"], default="synthetic")
    args = ap.parse_args(argv)
    if args.dataset == "uci":
        from .uci import build_uci_artifacts

        metrics = build_uci_artifacts()
    else:
        metrics = train()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
