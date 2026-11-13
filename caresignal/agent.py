"""Claude agent that works through the CareSignal MCP server.

The agent owns no clinical logic: every fact it states comes from an MCP tool
call, so the same tools serve Claude Desktop / Claude Code unchanged.

Providers:
  CARESIGNAL_PROVIDER=anthropic (default)  -> Claude API (ANTHROPIC_API_KEY)
  CARESIGNAL_PROVIDER=bedrock              -> Amazon Bedrock (AWS credentials/role)
  CARESIGNAL_PROVIDER=gemini               -> Google Gemini API (GEMINI_API_KEY; has a free tier)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

import anthropic
import httpx
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROVIDER = os.environ.get("CARESIGNAL_PROVIDER", "anthropic").lower()
MODEL = os.environ.get("CARESIGNAL_MODEL", "claude-opus-5-5")
GEMINI_MODEL = os.environ.get("CARESIGNAL_GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_TOOL_ROUNDS = 12
MAX_HISTORY_MESSAGES = 40

SYSTEM_PROMPT = """You are CareSignal, a discharge-planning assistant for hospital care teams \
(nurses, case managers, pharmacists, hospitalists). You help them find patients at high risk of \
30-day readmission, understand why, and plan follow-up.

How to work:
- Ground every patient-specific statement in a tool result from this conversation. If a tool \
has not told you something, look it up or say you don't have it; never guess a value.
- When asked about a patient's risk, call the risk tool and the medication check, then explain \
the main drivers in plain clinical language and suggest concrete, proportionate follow-up \
(e.g. early clinic visit, medication reconciliation, home health referral).
- The risk score comes from a statistical model. Present it as an estimate that supports, and \
never replaces, clinical judgment. Flag medication interactions for pharmacist or prescriber \
review rather than telling anyone to start, stop or change a drug.
- The cohort is synthetic demonstration data. Say so if the user appears to treat it as real.
- Be concise: lead with the answer, then the supporting numbers."""


class AgentUnavailable(RuntimeError):
    """The LLM could not be reached (missing credentials, outage, rate limit)."""


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    model: str = MODEL
    refused: bool = False


def make_client() -> tuple[anthropic.AsyncAnthropic, str, bool]:
    """Return (client, model id, is_first_party)."""
    if PROVIDER == "bedrock":
        from anthropic import AsyncAnthropicBedrockMantle

        region = os.environ.get("AWS_REGION", "us-east-1")
        return AsyncAnthropicBedrockMantle(aws_region=region), f"anthropic.{MODEL}", False
    return anthropic.AsyncAnthropic(), MODEL, True


class CareSignalAgent:
    """Holds one MCP stdio session for the life of the process."""

    def __init__(self, client=None, provider: str = PROVIDER) -> None:
        """`client` lets tests inject a stubbed SDK client for `provider`."""
        if provider not in ("anthropic", "bedrock", "gemini"):
            raise ValueError(f"Unknown CARESIGNAL_PROVIDER {provider!r}")
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._mcp_tools: list = []
        self._tools: list = []
        self._gemini = provider == "gemini"
        if self._gemini:
            self._client, self._model, self._first_party = client, GEMINI_MODEL, False
        elif client is not None:
            self._client, self._model, self._first_party = client, MODEL, True
        else:
            self._client, self._model, self._first_party = make_client()

    async def __aenter__(self) -> "CareSignalAgent":
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "caresignal.mcp_server"], env=dict(os.environ)
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = session
        self._mcp_tools = (await session.list_tools()).tools
        self._tools = [async_mcp_tool(t, session) for t in self._mcp_tools]
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._mcp_tools]

    async def chat(self, messages: list[dict]) -> AgentReply:
        """Run one user turn. `messages` is the text-only conversation so far,
        ending with the user's newest message."""
        if self._gemini:
            return await self._chat_gemini(messages[-MAX_HISTORY_MESSAGES:])
        extra: dict = {}
        if self._first_party:
            # If a safety classifier declines, let the API retry on a fallback
            # model instead of failing the turn (not available on Bedrock).
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

        runner = self._client.beta.messages.tool_runner(
            model=self._model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            tools=self._tools,
            messages=messages[-MAX_HISTORY_MESSAGES:],
            max_iterations=MAX_TOOL_ROUNDS,
            **extra,
        )

        tool_calls: list[dict] = []
        last = None
        try:
            async for message in runner:
                last = message
                tool_calls += [
                    {"name": b.name, "input": b.input} for b in message.content if b.type == "tool_use"
                ]
        except anthropic.AuthenticationError as e:
            raise AgentUnavailable("LLM credentials are missing or invalid.") from e
        except anthropic.RateLimitError as e:
            raise AgentUnavailable("The LLM is rate limited; try again shortly.") from e
        except anthropic.APIStatusError as e:
            raise AgentUnavailable(f"LLM request failed ({e.status_code}).") from e
        except anthropic.APIConnectionError as e:
            raise AgentUnavailable("Could not reach the LLM service.") from e
        except TypeError as e:  # SDK raises this when no credentials are configured
            raise AgentUnavailable("LLM credentials are not configured.") from e

        if last is None:
            raise AgentUnavailable("The LLM returned no response.")
        if last.stop_reason == "refusal":
            return AgentReply("I can't help with that request.", tool_calls, last.model, refused=True)
        text = "\n".join(b.text for b in last.content if b.type == "text").strip()
        if last.stop_reason == "max_tokens":
            text += "\n\n(Response was cut off at the length limit.)"
        return AgentReply(text or "(No answer produced.)", tool_calls, last.model)


    async def _chat_gemini(self, messages: list[dict]) -> AgentReply:
        """Same agent on Google Gemini (OpenAI-compatible endpoint): a manual
        tool loop over the MCP session."""
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise AgentUnavailable("LLM credentials are not configured (set GEMINI_API_KEY).")
            self._client = await self._stack.enter_async_context(
                httpx.AsyncClient(base_url=GEMINI_BASE_URL, headers={"Authorization": f"Bearer {key}"}, timeout=180)
            )

        tools = [
            {"type": "function",
             "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema}}
            for t in self._mcp_tools
        ]
        convo: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        tool_calls: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            for attempt in range(4):  # ride out brief "high demand" 5xx responses
                try:
                    resp = await self._client.post(
                        "chat/completions", json={"model": self._model, "messages": convo, "tools": tools}
                    )
                except httpx.HTTPError as e:
                    raise AgentUnavailable("Could not reach the LLM service.") from e
                if resp.status_code < 500 or attempt == 3:
                    break
                await asyncio.sleep(2 * 2**attempt)
            if resp.status_code >= 500:
                raise AgentUnavailable("The LLM service is overloaded right now; try again in a minute.")
            if resp.status_code == 429:
                raise AgentUnavailable("The LLM is rate limited (free-tier quota); try again shortly.")
            if resp.status_code in (401, 403):
                raise AgentUnavailable("LLM credentials are missing or invalid.")
            if resp.status_code >= 400:
                raise AgentUnavailable(f"LLM request failed ({resp.status_code}).")

            choices = resp.json().get("choices") or []
            message = choices[0].get("message") if choices else None
            if not message or not (message.get("content") or message.get("tool_calls")):
                # Blocked by a safety filter, or an empty completion.
                return AgentReply("I can't help with that request.", tool_calls, self._model, refused=True)

            calls = message.get("tool_calls") or []
            if not calls:
                return AgentReply(message["content"].strip(), tool_calls, self._model)

            # Echo the model turn back unchanged (it carries thought signatures).
            convo.append(message)
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                    result = await self._session.call_tool(name, args)
                    text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
                    if result.is_error:
                        text = f"ERROR: {text}"
                except json.JSONDecodeError:
                    args, text = {}, "ERROR: tool arguments were not valid JSON."
                tool_calls.append({"name": name, "input": args})
                convo.append({"role": "tool", "tool_call_id": call["id"], "content": text})

        return AgentReply(
            "I ran out of tool steps before finishing; try a narrower question.", tool_calls, self._model
        )


async def _repl() -> None:
    async with CareSignalAgent() as agent:
        print(f"CareSignal agent ready. Tools: {', '.join(agent.tool_names)}\n(ctrl-d to quit)")
        history: list[dict] = []
        while True:
            try:
                q = input("\nyou> ").strip()
            except EOFError:
                return
            if not q:
                continue
            history.append({"role": "user", "content": q})
            reply = await agent.chat(history)
            for call in reply.tool_calls:
                print(f"  [tool] {call['name']} {call['input']}")
            print(f"\ncaresignal> {reply.text}")
            history.append({"role": "assistant", "content": reply.text})


if __name__ == "__main__":
    asyncio.run(_repl())
