"""PyTorch 30-day readmission risk model: features, training, inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import generate_cohort

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

NUMERIC = [
    "age", "length_of_stay", "prior_admissions_12m", "num_medications",
    "charlson_index", "hba1c", "creatinine", "systolic_bp", "bmi",
]
BINARY = ["chf", "copd", "diabetes", "ckd", "lives_alone"]
FEATURES = NUMERIC + BINARY + ["sex_male", "discharged_snf", "discharged_home_health"]

FEATURE_LABELS = {
    "age": "Age",
    "length_of_stay": "Length of stay",
    "prior_admissions_12m": "Admissions in prior 12 months",
    "num_medications": "Number of medications",
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


def build_features(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Turn raw cohort rows into the model's feature frame (imputing labs)."""
    out = df[NUMERIC + BINARY].copy()
    for col, med in medians.items():
        out[col] = out[col].fillna(med)
    out["sex_male"] = (df["sex"] == "M").astype(int)
    out["discharged_snf"] = (df["discharge_disposition"] == "snf").astype(int)
    out["discharged_home_health"] = (df["discharge_disposition"] == "home_health").astype(int)
    return out[FEATURES].astype("float32")


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
    medians: dict[str, float]
    mean: np.ndarray
    std: np.ndarray

    def _x(self, df: pd.DataFrame) -> torch.Tensor:
        feats = build_features(df, self.medians).to_numpy()
        return torch.from_numpy((feats - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        return torch.sigmoid(self.net(self._x(df))).numpy()

    @torch.no_grad()
    def explain(self, row: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """Occlusion attribution: how much does risk move when one feature is
        replaced by the cohort average? Positive = feature raises this
        patient's risk."""
        self.net.eval()
        x = self._x(row)
        base = torch.sigmoid(self.net(x)).item()
        occluded = x.repeat(len(FEATURES), 1)
        occluded[torch.arange(len(FEATURES)), torch.arange(len(FEATURES))] = 0.0
        delta = base - torch.sigmoid(self.net(occluded)).numpy()
        raw = build_features(row, self.medians).iloc[0]
        order = np.argsort(-np.abs(delta))[:top_k]
        return [
            {
                "feature": FEATURES[i],
                "label": FEATURE_LABELS[FEATURES[i]],
                "value": round(float(raw.iloc[i]), 2),
                "risk_contribution": round(float(delta[i]), 4),
            }
            for i in order
        ]

    def save(self, path: Path) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "medians": self.medians,
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "features": FEATURES,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "RiskModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt["features"] != FEATURES:
            raise ValueError("Model artifact was trained with a different feature set; retrain.")
        net = RiskNet(len(FEATURES))
        net.load_state_dict(ckpt["state_dict"])
        return cls(
            net,
            ckpt["medians"],
            np.array(ckpt["mean"], dtype="float32"),
            np.array(ckpt["std"], dtype="float32"),
        )


def train(n: int = 6000, epochs: int = 60, seed: int = 7, out_dir: Path = ARTIFACT_DIR) -> dict:
    """Generate the cohort, train, and write cohort.csv / model.pt /
    predictions.csv / metrics.json to `out_dir`."""
    torch.manual_seed(seed)
    df = generate_cohort(n, seed)
    rng = np.random.default_rng(seed)
    split = rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15])
    tr = df[split == "train"]

    medians = {c: float(tr[c].median()) for c in ("hba1c", "creatinine")}
    f_tr = build_features(tr, medians).to_numpy()
    mean, std = f_tr.mean(0), f_tr.std(0) + 1e-6
    model = RiskModel(RiskNet(len(FEATURES)), medians, mean, std)

    x_tr = model._x(tr)
    y_tr = torch.tensor(tr["readmitted_30d"].to_numpy(), dtype=torch.float32)
    val = df[split == "val"]
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

    test = df[split == "test"]
    p_test = model.predict(test)
    y_test = test["readmitted_30d"].to_numpy()
    metrics = {
        "n_patients": n,
        "n_test": int(len(test)),
        "readmission_rate": round(float(df["readmitted_30d"].mean()), 4),
        "val_auroc": round(best_auc, 4),
        "test_auroc": round(auroc(y_test, p_test), 4),
        "test_brier": round(float(np.mean((p_test - y_test) ** 2)), 4),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cohort.csv", index=False)
    model.save(out_dir / "model.pt")
    pd.DataFrame(
        {"patient_id": df["patient_id"], "split": split, "predicted_risk": model.predict(df).round(5)}
    ).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
