"""End-to-end tests: model artifacts, service layer, MCP server over stdio, REST API."""
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from caresignal.model import ARTIFACT_DIR, FEATURES
from caresignal.service import NotFound, get_service


@pytest.fixture(scope="module")
def svc():
    return get_service()


def test_model_beats_chance():
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    assert metrics["test_auroc"] > 0.70


def test_risk_and_drivers(svc):
    top = svc.list_patients(risk_band="high", limit=1)[0]
    risk = svc.readmission_risk(top["patient_id"])
    assert 0.30 <= risk["predicted_30d_readmission_risk"] <= 1
    assert risk["risk_band"] == "high"
    assert len(risk["top_drivers"]) == 5
    assert {d["feature"] for d in risk["top_drivers"]} <= set(FEATURES)


def test_label_never_exposed(svc):
    pid = svc.list_patients(limit=1)[0]["patient_id"]
    assert "readmitted_30d" not in svc.get_patient(pid)


def test_unknown_patient(svc):
    with pytest.raises(NotFound):
        svc.get_patient("P999")


def test_interactions(svc):
    hits = svc.check_interactions(["Warfarin", "ibuprofen ", "metformin"])
    assert [h["drugs"] for h in hits] == [["ibuprofen", "warfarin"]]
    assert hits[0]["severity"] == "major"
    assert svc.check_interactions(["metformin", "acetaminophen"]) == []


def test_cohort_summary(svc):
    rows = svc.cohort_summary("risk_band")["rows"]
    by_band = {r["risk_band"]: r for r in rows}
    # Higher predicted band should actually be readmitted more often.
    assert by_band["high"]["observed_readmission_rate"] > by_band["low"]["observed_readmission_rate"]
    with pytest.raises(ValueError):
        svc.cohort_summary("medications")


@pytest.mark.anyio
async def test_mcp_server_over_stdio(svc):
    pid = svc.list_patients(limit=1)[0]["patient_id"]
    params = StdioServerParameters(command=sys.executable, args=["-m", "caresignal.mcp_server"], env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert names == {
                "data_source_info", "get_patient", "list_patients", "predict_readmission_risk",
                "check_patient_medications", "check_drug_interactions", "cohort_summary",
            }
            result = await session.call_tool("predict_readmission_risk", {"patient_id": pid})
            assert not result.is_error
            payload = json.loads(result.content[0].text)
            assert payload["patient_id"] == pid
            bad = await session.call_tool("get_patient", {"patient_id": "nope"})
            assert bad.is_error


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_rest_api(svc):
    from caresignal.api import app

    with TestClient(app) as client:
        assert len(client.get("/api/health").json()["tools"]) == 7
        rows = client.get("/api/patients", params={"risk_band": "high", "limit": 5}).json()
        assert len(rows) == 5 and rows[0]["predicted_risk"] >= rows[-1]["predicted_risk"]
        detail = client.get(f"/api/patients/{rows[0]['patient_id']}").json()
        assert detail["risk"]["top_drivers"] and "interactions" in detail
        assert client.get("/api/patients/P1").status_code == 404
        assert client.get("/api/patients", params={"limit": 9999}).status_code == 422
        assert client.post("/api/chat", json={"messages": [{"role": "assistant", "content": "hi"}]}).status_code == 422


@pytest.mark.anyio
async def test_agent_loop_with_stubbed_llm(svc):
    """Drive the real agent + real MCP server against a stubbed Claude API:
    turn 1 asks for a tool, turn 2 must contain the MCP tool's result."""
    import anthropic

    from caresignal.agent import CareSignalAgent

    httpx = sys.modules[anthropic.DefaultAsyncHttpxClient.__mro__[1].__module__.split(".")[0]]
    pid = svc.list_patients(limit=1)[0]["patient_id"]
    seen: list[dict] = []

    def msg(content, stop):
        return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                "content": content, "stop_reason": stop, "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1}}

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json=msg(
                [{"type": "tool_use", "id": "toolu_1", "name": "predict_readmission_risk", "input": {"patient_id": pid}}],
                "tool_use"))
        return httpx.Response(200, json=msg([{"type": "text", "text": "High risk."}], "end_turn"))

    client = anthropic.AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with CareSignalAgent(client) as agent:
        reply = await agent.chat([{"role": "user", "content": f"Risk for {pid}?"}])

    assert reply.text == "High risk."
    assert reply.tool_calls == [{"name": "predict_readmission_risk", "input": {"patient_id": pid}}]
    assert len(seen[0]["tools"]) == 7 and seen[0]["model"] == "claude-opus-5-5"
    tool_result = seen[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and pid in json.dumps(tool_result["content"])


@pytest.mark.anyio
async def test_gemini_agent_loop_with_stubbed_llm(svc):
    """Gemini provider: real MCP server, stubbed Gemini HTTP endpoint."""
    import httpx

    from caresignal.agent import GEMINI_BASE_URL, CareSignalAgent

    pid = svc.list_patients(limit=1)[0]["patient_id"]
    seen: list[dict] = []

    def handler(request):
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            message = {"role": "assistant", "tool_calls": [{
                "id": "c1", "type": "function", "extra_content": {"google": {"thought_signature": "sig"}},
                "function": {"name": "check_patient_medications", "arguments": json.dumps({"patient_id": pid})}}]}
        else:
            message = {"role": "assistant", "content": "No major interactions."}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    async with httpx.AsyncClient(base_url=GEMINI_BASE_URL, transport=httpx.MockTransport(handler)) as stub:
        async with CareSignalAgent(stub, provider="gemini") as agent:
            reply = await agent.chat([{"role": "user", "content": f"Meds for {pid}?"}])

    assert reply.text == "No major interactions."
    assert reply.tool_calls == [{"name": "check_patient_medications", "input": {"patient_id": pid}}]
    assert len(seen[0]["tools"]) == 7 and seen[0]["messages"][0]["role"] == "system"
    echoed, result = seen[1]["messages"][-2:]
    assert echoed["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig"
    assert result == {"role": "tool", "tool_call_id": "c1", "content": result["content"]} and pid in result["content"]


@pytest.mark.anyio
async def test_gemini_bad_key_is_reported_as_credentials(svc):
    """Google answers an invalid key with HTTP 400; the user should be told it's the key."""
    import httpx

    from caresignal.agent import GEMINI_BASE_URL, AgentUnavailable, CareSignalAgent

    def handler(request):
        return httpx.Response(400, json=[{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}}])

    async with httpx.AsyncClient(base_url=GEMINI_BASE_URL, transport=httpx.MockTransport(handler)) as stub:
        async with CareSignalAgent(stub, provider="gemini") as agent:
            with pytest.raises(AgentUnavailable, match="credentials"):
                await agent.chat([{"role": "user", "content": "hi"}])
