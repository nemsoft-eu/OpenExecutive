from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from openexecutive.audit.usage import log_model_usage
from openexecutive.config import get_settings
from openexecutive.providers import get_provider, model_supports_deep_reasoning

_SPECIALIST_TIMEOUT = 180.0

logger = logging.getLogger(__name__)


def _unwrap_tool_outcome(outcome: Any) -> tuple[str, bool]:
    """Normalise a handler's return to ``(content, is_error)``.

    Handlers may return a plain string (every skill tool does) or a
    ``ToolOutcome`` when they need to flag failure. Keeping the content a
    string matters on the local path: the OpenAI-compatible translator only
    converts string or text-block tool_result content.
    """
    content = getattr(outcome, "content", None)
    if content is not None and hasattr(outcome, "is_error"):
        return str(content), bool(outcome.is_error)
    return str(outcome), False


class BaseAgent(ABC):
    name: str
    domain: str
    model: str
    use_deep_reasoning: bool = False

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    def effective_system_prompt(self) -> str:
        """System prompt with runtime override applied, if any."""
        from openexecutive.agents.overrides import get_override

        ov = get_override(self.name)
        if ov is not None and ov.prompt is not None:
            return ov.prompt
        return self.get_system_prompt()

    def effective_model(self) -> str:
        from openexecutive.agents.overrides import get_override

        ov = get_override(self.name)
        if ov is not None and ov.model is not None:
            return ov.model
        return self.model

    def effective_use_deep_reasoning(self) -> bool:
        from openexecutive.agents.overrides import get_override

        ov = get_override(self.name)
        if ov is not None and ov.use_deep_reasoning is not None:
            return ov.use_deep_reasoning
        return self.use_deep_reasoning

    async def analyze(
        self,
        query: str,
        context: str = "",
        retrieved_knowledge: str = "",
        episodic_context: str = "",
        failure_cases: str = "",
        department_memory: str = "",
        *,
        system_prompt_override: str | None = None,
        model_override: str | None = None,
        deep_reasoning_override: bool | None = None,
        actor: str = "specialist",
    ) -> str:
        """Run one prose specialist call and return its text.

        Every call records a ``cache_event`` usage row under ``actor``, so
        the call counts toward the session cost summary and ``/audit/usage``
        like every other model call. The router sets ``actor`` per path
        (``specialist`` for chat-turn consults, ``specialist_workflow`` for
        workflow steps); the Council test box passes ``agent_test``.
        """
        settings = get_settings()

        # Resolution order: explicit kwarg (for sandbox/test calls) → DB
        # override → class default. Keeping the override read inside this
        # method means a fresh DB row takes effect on the next request
        # without any process restart.
        system_prompt = (
            system_prompt_override
            if system_prompt_override is not None
            else self.effective_system_prompt()
        )
        model = (
            model_override
            if model_override is not None
            else self.effective_model()
        )
        use_deep = (
            deep_reasoning_override
            if deep_reasoning_override is not None
            else self.effective_use_deep_reasoning()
        )

        user_content = query
        if context:
            user_content = f"<conversation_context>\n{context}\n</conversation_context>\n\n{query}"
        if retrieved_knowledge:
            user_content = (
                f"<relevant_knowledge>\n{retrieved_knowledge}\n</relevant_knowledge>\n\n{user_content}"
            )
        if failure_cases:
            user_content = (
                f"<failure_cases>\n{failure_cases}\n</failure_cases>\n\n{user_content}"
            )
        if episodic_context:
            user_content = (
                f"<past_decisions>\n{episodic_context}\n</past_decisions>\n\n{user_content}"
            )
        if department_memory:
            # Placed adjacent to past_decisions so the specialist sees both
            # forms of institutional context together: the structured ledger
            # (past_decisions, from SQL) and the dept peer's synthesized
            # voice (department_memory, from Honcho). Order is intentional —
            # past_decisions stays closest to the query for cache stability
            # across turns; department_memory wraps it.
            user_content = (
                f"<department_memory>\n{department_memory}\n</department_memory>\n\n{user_content}"
            )

        create_kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            # Per-request timeout — keeps cancellation semantics aligned with
            # the previous per-client timeout, but the provider singleton no
            # longer needs to recreate the SDK client to set it.
            "timeout": _SPECIALIST_TIMEOUT,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_content}],
        }

        if use_deep and model_supports_deep_reasoning(model):
            # Adaptive thinking with effort capped via
            # output_config.effort. Default "low" — adaptive at default
            # effort routinely burned 20k+ thinking tokens. "low" still
            # leaves room for nuanced reasoning while being ~3x faster.
            # Tune via SPECIALIST_EFFORT (low / medium / high / xhigh / max).
            create_kwargs["thinking"] = {"type": "adaptive"}
            create_kwargs["output_config"] = {"effort": settings.specialist_effort}
            create_kwargs["max_tokens"] = 16000

        # Resolve provider per-call by model so a Council UI override that
        # flips this agent to a non-Anthropic slug routes correctly. Today
        # the registry always returns the Anthropic provider; OpenRouter
        # wiring lands in the next commit.
        provider = get_provider(model)
        message = await provider.messages_create(**create_kwargs)
        log_model_usage(message, model=model, actor=actor)

        text_blocks = [b for b in message.content if b.type == "text"]
        if not text_blocks:
            # A reasoning model can spend the whole max_tokens budget thinking
            # and return no text; the Executive would then synthesize as if
            # this specialist had nothing to say. Make that visible.
            logger.warning(
                "specialist %s (%s) returned no text block (stop_reason=%s, "
                "deep_reasoning=%s); its analysis will be empty",
                self.name,
                model,
                getattr(message, "stop_reason", None),
                use_deep,
            )
        return text_blocks[0].text if text_blocks else ""

    async def analyze_with_tools(
        self,
        user_content: str,
        *,
        tools: list[dict[str, Any]],
        system_addendum: str = "",
        max_tokens: int = 4096,
        timeout_seconds: float = _SPECIALIST_TIMEOUT,
        model_override: str | None = None,
        deep_reasoning_override: bool | None = None,
        actor: str = "specialist_tools",
        client_tool_handlers: dict[str, Any] | None = None,
        terminal_tool_names: set[str] | None = None,
        max_client_tool_rounds: int = 3,
    ) -> Any:
        """Tool-use variant of ``analyze`` — returns the raw provider Message.

        Used by workflows that need structured output from a specialist
        (e.g. the watchlist research workflow's
        ``propose_watchlist_entries`` tool). Differs from ``analyze``:

          - ``tools`` is required; the model is expected to call one of
            them rather than emit prose.
          - The user_content is passed verbatim — no
            ``<conversation_context>`` / ``<retrieved_knowledge>`` /
            ``<past_decisions>`` wrappers (those are chat-time
            primitives). The caller owns context formatting.
          - Returns the raw ``Message`` instead of joined text so the
            caller can pick out the tool_use block it expects.

        Honours every per-instance effective_* override (system prompt,
        model, deep-reasoning) so a Council override applies here too.

        ``model_override`` / ``deep_reasoning_override`` let a caller pin the
        model and reasoning tier for this one turn (e.g. the executive_research
        fan-out routes specialists to a cheaper RESEARCH_MODEL with deep
        reasoning off) without disturbing the agent's chat-time defaults. When
        None, the per-instance effective_* values apply, matching prior
        behavior.

        ``actor`` names the caller on the ``cache_event`` usage row recorded
        for the call (``specialist_research``, ``query_watch``, …), which is
        what the per-source usage breakdown groups on.

        ``client_tool_handlers`` turns the single call into a bounded tool
        loop: any tool in the map is executed here and fed back, so a model
        whose provider has no server-side search (the local backend, using
        the SearXNG ``web_search`` tool) can still search before answering.
        Empty or None keeps the historical single-shot behaviour exactly.
        ``terminal_tool_names`` names the output tools the caller extracts
        from the returned message — they have no handler and reaching one
        ends the loop. ``max_client_tool_rounds`` bounds the generations.
        """
        settings = get_settings()
        system_prompt = self.effective_system_prompt() + (
            system_addendum or ""
        )
        model = (
            model_override
            if model_override is not None
            else self.effective_model()
        )
        use_deep = (
            deep_reasoning_override
            if deep_reasoning_override is not None
            else self.effective_use_deep_reasoning()
        )

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout_seconds,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": tools,
            "messages": [{"role": "user", "content": user_content}],
        }

        if use_deep and model_supports_deep_reasoning(model):
            create_kwargs["thinking"] = {"type": "adaptive"}
            create_kwargs["output_config"] = {"effort": settings.specialist_effort}
            create_kwargs["max_tokens"] = max(max_tokens, 16000)

        provider = get_provider(model)
        if not client_tool_handlers:
            message = await provider.messages_create(**create_kwargs)
            log_model_usage(message, model=model, actor=actor)
            return message

        return await self._run_client_tool_loop(
            provider,
            create_kwargs,
            model=model,
            actor=actor,
            client_tool_handlers=client_tool_handlers,
            terminal_tool_names=set(terminal_tool_names or ()),
            max_rounds=max_client_tool_rounds,
        )

    async def _run_client_tool_loop(
        self,
        provider: Any,
        create_kwargs: dict[str, Any],
        *,
        model: str,
        actor: str,
        client_tool_handlers: dict[str, Any],
        terminal_tool_names: set[str],
        max_rounds: int,
    ) -> Any:
        """Run generations until the model produces its terminal output.

        Callers of ``analyze_with_tools`` advertise an output tool that has
        no handler (``emit_research_findings``, ``emit_query_results``) and
        then read it off the returned raw message. So a handler map alone
        cannot classify a tool_use block — ``terminal_tool_names`` is what
        separates "the model has answered" from "the model called a tool I
        must run" from "the model hallucinated a tool".

        Termination, in priority order:

          * A terminal tool appears — return that message immediately, even
            if a search appeared alongside it. Running a search whose result
            cannot reach the already-emitted output is pure cost.
          * No tool_use at all — return the message. The caller's extractor
            will find nothing, which is logged here because a specialist
            that answers in prose has silently produced no findings.
          * Only handled tools — run them, append the assistant turn plus one
            tool_result per tool_use, and go again.
          * An unknown tool — answer it with an error tool_result so the
            transcript stays valid (every tool_use must be answered) and let
            the model correct itself on the next round.

        On the final round the handled tools are withdrawn and only the
        terminal tools are offered, so a model that would otherwise keep
        searching is forced to emit. Without that, a search on the last
        generation leaves the caller with an unusable message.
        """
        kwargs = dict(create_kwargs)
        all_tools = list(kwargs.get("tools") or [])
        # At least one generation, always: with zero rounds the loop body
        # never runs and there is no message to return.
        max_rounds = max(1, max_rounds)

        for round_number in range(1, max_rounds + 1):
            if round_number == max_rounds and terminal_tool_names:
                # Last chance: withdraw the handled tools so the only move
                # left is the terminal one.
                kwargs["tools"] = [
                    t for t in all_tools
                    if t.get("name") not in client_tool_handlers
                ] or all_tools

            message = await provider.messages_create(**kwargs)
            # Every generation is billed, so every generation is accounted.
            log_model_usage(message, model=model, actor=actor)

            tool_uses = [
                b for b in (getattr(message, "content", None) or [])
                if getattr(b, "type", "") == "tool_use"
            ]
            if not tool_uses:
                logger.warning(
                    "%s: model returned no tool_use block on round %d/%d — "
                    "the caller's extractor will find nothing",
                    self.name, round_number, max_rounds,
                )
                return message
            if any(getattr(b, "name", "") in terminal_tool_names for b in tool_uses):
                return message
            if round_number == max_rounds:
                # A model that ignored the withdrawn tool list and searched
                # anyway: running that search now would spend budget on a
                # result no later generation can read.
                break

            assistant_content, tool_results = await self._dispatch_tool_uses(
                tool_uses, client_tool_handlers
            )
            kwargs["messages"] = [
                *kwargs["messages"],
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": tool_results},
            ]

        logger.warning(
            "%s: client tool loop exhausted %d rounds without a terminal tool",
            self.name, max_rounds,
        )
        return message

    async def _dispatch_tool_uses(
        self,
        tool_uses: list[Any],
        client_tool_handlers: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run one round's tool calls; return the assistant turn and its results.

        Every tool_use gets exactly one tool_result, including a tool with
        no handler — an unanswered tool_use makes the next request's
        transcript invalid. Handlers run sequentially: a round rarely holds
        more than one search, and the search budget is enforced inside each
        handler either way.
        """
        assistant_content: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in tool_uses:
            name = getattr(block, "name", "")
            block_id = getattr(block, "id", "")
            block_input = getattr(block, "input", None) or {}
            assistant_content.append({
                "type": "tool_use",
                "id": block_id,
                "name": name,
                "input": block_input,
            })
            handler = client_tool_handlers.get(name)
            if handler is None:
                logger.warning(
                    "%s: model called unknown tool %r — returning an error",
                    self.name, name,
                )
                result_content, is_error = f"Unknown tool: {name}", True
            else:
                result_content, is_error = _unwrap_tool_outcome(
                    await handler(block_input)
                )
            result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block_id,
                "content": result_content,
            }
            if is_error:
                result_block["is_error"] = True
            tool_results.append(result_block)
        return assistant_content, tool_results
