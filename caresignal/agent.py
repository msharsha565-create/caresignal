"""Claude agent that works through the CareSignal MCP server.

The agent owns no clinical logic: every fact it states comes from an MCP tool
call, so the same tools serve Claude Desktop / Claude Code unchanged.

Providers:
  CARESIGNAL_PROVIDER=anthropic (default)  -> Claude API (ANTHROPIC_API_KEY)
  CARESIGNAL_PROVIDER=bedrock              -> Amazon Bedrock (AWS credentials/role)
"""
from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

import anthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

MODEL = os.environ.get("CARESIGNAL_MODEL", "claude-opus-5-5")
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
    if os.environ.get("CARESIGNAL_PROVIDER", "anthropic").lower() == "bedrock":
        from anthropic import AsyncAnthropicBedrockMantle

        region = os.environ.get("AWS_REGION", "us-east-1")
        return AsyncAnthropicBedrockMantle(aws_region=region), f"anthropic.{MODEL}", False
    return anthropic.AsyncAnthropic(), MODEL, True


class CareSignalAgent:
    """Holds one MCP stdio session for the life of the process."""

    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._stack = AsyncExitStack()
        self._tools: list = []
        if client is not None:  # injected (tests)
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
        self._tools = [async_mcp_tool(t, session) for t in (await session.list_tools()).tools]
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    async def chat(self, messages: list[dict]) -> AgentReply:
        """Run one user turn. `messages` is the text-only conversation so far,
        ending with the user's newest message."""
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
            max_iterations=12,
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
    import asyncio

    asyncio.run(_repl())
