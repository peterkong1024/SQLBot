"""Langfuse tracing instrumentation for the chat/Q&A pipeline.

Uses the langfuse SDK v4 (OpenTelemetry-based) low-level API.

Design
------
* A single ``Langfuse`` client (singleton) is shared across requests.
* Trace-level attributes (``session_id`` = chat window, ``user_id``) are set
  via :func:`langfuse.propagate_attributes`, which propagates them to every
  observation created inside its scope.
* Each user request establishes its trace context **inside the worker
  thread** that runs ``LLMService.run_task`` (``ThreadPoolExecutor`` copies
  the contextvar scope into the worker). Because all observations in a
  request are created in that same thread, the OTel parent/child nesting
  works correctly even though the pipeline is driven by generators that
  ``yield`` SSE chunks back to the caller.
* Observations are created with ``client.start_as_current_observation`` as
  context managers (``with`` blocks) so they auto-end on exit, including the
  exception path.
* Everything is best-effort: when Langfuse is disabled (or unreachable at
  init) a no-op tracer is used so the request flow is never affected.

The mapping that satisfies the product requirement:
    trace          = one user request        (new trace id every request)
    session_id     = chat window (chat_id)   (same across requests in a chat)
    generation     = an LLM call (model/io/usage)
    span           = a tool call (table select, RAG recall, exec_sql, ...)
"""
import contextlib
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from common.core.config import settings
from common.utils.utils import SQLBotLogUtil

_langfuse_client = None


class _NullObs:
    """No-op observation mirroring LangfuseSpan/LangfuseGeneration API."""

    def update(self, **_kwargs: Any) -> "_NullObs":
        return self

    def end(self, **_kwargs: Any) -> "_NullObs":
        return self


class _NullCM:
    """No-op context manager yielding a :class:`_NullObs`."""

    def __enter__(self) -> _NullObs:
        return _NullObs()

    def __exit__(self, *_exc: Any) -> bool:
        return False


NULL_OBSERVATION = _NullObs()
NULL_CONTEXT_MANAGER = _NullCM()


def _is_configured() -> bool:
    return bool(
        getattr(settings, "LANGFUSE_ENABLED", False)
        and settings.LANGFUSE_HOST
        and settings.LANGFUSE_PUBLIC_KEY
        and settings.LANGFUSE_SECRET_KEY
    )


def get_langfuse():
    """Return the singleton Langfuse client, or ``None`` when disabled.

    Never raises — tracing must not break the application.
    """
    global _langfuse_client
    if not _is_configured():
        return None
    if _langfuse_client is None:
        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                host=settings.LANGFUSE_HOST,
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
            )
        except Exception as e:  # pragma: no cover - best-effort init
            SQLBotLogUtil.exception(f"Langfuse init failed, tracing disabled: {e}")
            return None
    return _langfuse_client


def flush() -> None:
    """Flush pending observations to Langfuse (best-effort)."""
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception:  # pragma: no cover - best-effort
            pass


def shutdown() -> None:
    """Flush and shut down the Langfuse client (best-effort, for app exit)."""
    client = get_langfuse()
    if client is not None:
        try:
            client.shutdown()
        except Exception:  # pragma: no cover - best-effort
            pass


# Map langchain message types to the OpenAI/industry-standard role names so the
# captured input matches what users expect (user/assistant, not human/ai).
_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "function": "function",
}


def _std_role(raw: Any) -> Any:
    return _ROLE_MAP.get(raw, raw)


def serialize_messages(messages) -> List[Dict[str, Any]]:
    """Serialize langchain ``BaseMessage`` (or dict) list to plain dicts.

    Output is JSON-serializable so it can be stored as a generation ``input``.
    Roles are normalized to the OpenAI convention (user/assistant/system/tool).
    """
    result: List[Dict[str, Any]] = []
    if not messages:
        return result
    for m in messages:
        try:
            if isinstance(m, BaseMessage):
                result.append({"role": _std_role(m.type), "content": m.content})
            elif isinstance(m, dict):
                result.append({
                    "role": _std_role(m.get("role") or m.get("type")),
                    "content": m.get("content"),
                })
            else:
                result.append({"content": str(m)})
        except Exception:
            result.append({"content": str(m)})
    return result


def map_usage(token_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """Map sqlbot ``token_usage`` to langfuse ``usage_details``.

    sqlbot: ``{input_tokens, output_tokens, total_tokens}``
    langfuse: ``{input, output, total}``
    """
    if not token_usage:
        return None
    try:
        usage = {
            "input": int(token_usage.get("input_tokens") or 0),
            "output": int(token_usage.get("output_tokens") or 0),
            "total": int(token_usage.get("total_tokens") or 0),
        }
    except Exception:
        return None
    if not any(usage.values()):
        return None
    return usage


class Tracer:
    """Per-request tracing helper bound to a Langfuse client.

    Every method is no-op-safe when the client is ``None`` (disabled). The
    returned context managers / observations always expose ``.update()`` and
    ``.end()`` so call sites never need to branch on whether tracing is on.
    """

    def __init__(self, client):
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextlib.contextmanager
    def trace_context(self, *, name: str, session_id: Optional[str], user_id: Optional[str],
                      metadata: Optional[Dict[str, Any]] = None, input: Any = None):
        """Establish the trace for a request.

        Sets trace-level attributes (session_id/user_id/trace name) via
        ``propagate_attributes`` and creates a root observation that all
        child observations (spans/generations) nest under. Must be called
        inside the worker thread that runs the pipeline.
        """
        if not self.enabled:
            yield NULL_OBSERVATION
            return
        try:
            from langfuse import propagate_attributes

            pa_cm = propagate_attributes(
                session_id=session_id,
                user_id=user_id,
                trace_name=name,
                metadata=metadata or {},
            )
            root_cm = self._client.start_as_current_observation(
                name=name, as_type="span", input=input, metadata=metadata or {},
                end_on_exit=True,
            )
        except Exception as e:  # pragma: no cover - best-effort
            SQLBotLogUtil.exception(f"Langfuse trace_context setup failed: {e}")
            yield NULL_OBSERVATION
            return

        # NOTE: the body is outside the try/except so user exceptions still
        # propagate (and the OTel context managers mark the span on the way out).
        with pa_cm:
            with root_cm as root:
                yield root

    def span(self, *, name: str, input: Any = None, metadata: Any = None):
        """Context manager creating a (tool/step) span under the current trace."""
        if not self.enabled:
            return NULL_CONTEXT_MANAGER
        try:
            return self._client.start_as_current_observation(
                name=name, as_type="span", input=input, metadata=metadata, end_on_exit=True,
            )
        except Exception:  # pragma: no cover - best-effort
            return NULL_CONTEXT_MANAGER

    def generation(self, *, name: str, model: Optional[str] = None, input: Any = None,
                   metadata: Any = None):
        """Context manager creating a generation (LLM call) under the current trace."""
        if not self.enabled:
            return NULL_CONTEXT_MANAGER
        try:
            return self._client.start_as_current_observation(
                name=name, as_type="generation", model=model, input=input, metadata=metadata,
                end_on_exit=True,
            )
        except Exception:  # pragma: no cover - best-effort
            return NULL_CONTEXT_MANAGER

    def flush(self) -> None:
        flush()


def get_tracer() -> Tracer:
    """Build a :class:`Tracer` bound to the singleton client (or a no-op one)."""
    return Tracer(get_langfuse())
