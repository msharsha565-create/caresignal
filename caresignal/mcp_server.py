"""MCP server exposing the clinical tools.

Run standalone (stdio) with `python -m caresignal.mcp_server`; any MCP client
(Claude Desktop, Claude Code, the bundled agent) can connect to it.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .service import get_service

mcp = MCPServer(
    "caresignal",
    instructions=(
        "Tools over a SYNTHETIC inpatient discharge cohort: patient lookup, a PyTorch "
        "30-day readmission risk model, medication interaction checks and cohort statistics."
    ),
)


@mcp.tool()
def get_patient(patient_id: str) -> dict:
    """Fetch one patient's discharge record (demographics, comorbidities, labs,
    medications, predicted risk). Patient ids look like P100042."""
    return get_service().get_patient(patient_id)


@mcp.tool()
def list_patients(
    risk_band: str | None = None,
    condition: str | None = None,
    min_age: int | None = None,
    limit: int = 20,
) -> list[dict]:
    """List patients ordered by predicted readmission risk (highest first).

    Args:
        risk_band: Optional filter - "high" (>=30%), "medium" (15-30%) or "low".
        condition: Optional filter - one of chf, copd, diabetes, ckd.
        min_age: Optional minimum age.
        limit: Max rows to return (1-200).
    """
    return get_service().list_patients(risk_band, condition, min_age, limit)


@mcp.tool()
def predict_readmission_risk(patient_id: str) -> dict:
    """Score a patient's 30-day readmission risk with the PyTorch model and
    return the features driving it (occlusion attribution; positive
    risk_contribution means the feature raises this patient's risk)."""
    return get_service().readmission_risk(patient_id)


@mcp.tool()
def check_patient_medications(patient_id: str) -> dict:
    """Check a patient's current medication list for known drug-drug interactions."""
    return get_service().patient_interactions(patient_id)


@mcp.tool()
def check_drug_interactions(medications: list[str]) -> list[dict]:
    """Check an arbitrary list of generic drug names for known interactions
    (e.g. before proposing a medication change)."""
    return get_service().check_interactions(medications)


@mcp.tool()
def cohort_summary(group_by: str | None = None) -> dict:
    """Cohort statistics: patient counts, mean predicted risk and observed
    30-day readmission rate, optionally grouped.

    Args:
        group_by: One of risk_band, discharge_disposition, sex, chf, copd,
            diabetes, ckd, lives_alone. Omit for whole-cohort totals.
    """
    return get_service().cohort_summary(group_by)


if __name__ == "__main__":
    mcp.run()
