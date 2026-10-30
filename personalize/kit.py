"""Middleware and tools that customize one user's answers and leave everyone else alone."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import SystemMessage

from personalize.feedback import (
    build_example,
    feedback_ask,
    feedback_ready,
    review_feedback,
    states_a_fact,
)

Fold = Callable[[dict[str, Any], str, str], dict[str, Any]]
Lines = Callable[[dict[str, Any], str], list[str]]
Render = Callable[[dict[str, Any], str], str]
Blank = Callable[[], dict[str, Any]]
Inject = Callable[[str, str], None]


@dataclass
class UserContext:
    """Context for an agent that has no context type of its own."""

    user_id: str


def _blank() -> dict[str, Any]:
    return {"notes": []}


def _fold(profile: dict[str, Any], note: str, thread_id: str) -> dict[str, Any]:
    text = note.strip()
    if not text:
        return profile
    notes = profile.setdefault("notes", [])
    if text not in notes:
        notes.append(text[:240])
    return profile


def _lines(profile: dict[str, Any], question: str) -> list[str]:
    return [str(note) for note in profile.get("notes") or []]


def _render(profile: dict[str, Any], user_id: str) -> str:
    notes = _lines(profile, "")
    if not notes:
        return f"Nothing saved yet for {user_id}."
    return f"On file for {user_id}: " + " | ".join(notes)


def user_of(context: Any) -> str:
    """Read the user from context. The model does not supply this."""

    for field in ("user_id", "customer_id"):
        value = getattr(context, field, "") or ""
        if value:
            return str(value)
    return ""


class Personalize:
    """Personal memory, feedback, and a monthly review for any ``create_agent`` call.

    A preference customizes answers for one user. Feedback is a judgment that
    should make the agent better for every user. ``review`` suggests a knowledge
    note, a prompt change, a tool, or leaving the note on that user. It does
    not edit the agent.
    """

    context_schema = UserContext

    def __init__(
        self,
        *,
        fold: Fold | None = None,
        lines: Lines | None = None,
        render: Render | None = None,
        blank: Blank | None = None,
        on_inject: Inject | None = None,
        on_save: Inject | None = None,
        on_recall: Callable[[str], None] | None = None,
        on_feedback: Callable[[str, str], None] | None = None,
        audience: str = "this user",
        tool_gaps: tuple[tuple[str, str, str], ...] = (),
        personal_markers: tuple[str, ...] | None = None,
    ) -> None:
        self.fold = fold or _fold
        self.lines = lines or _lines
        self.render = render or _render
        self.blank = blank or _blank
        self.on_inject = on_inject
        self.on_save = on_save
        self.on_recall = on_recall
        self.on_feedback = on_feedback
        self.audience = audience
        self.tool_gaps = tool_gaps
        self.personal_markers = personal_markers
        self.feedback: list[dict[str, Any]] = []
        self.save_preference = self._save_tool()
        self.recall_preferences = self._recall_tool()
        self.save_feedback = self._feedback_tool()
        self.middleware = _PersonalizeMiddleware(self)

    @property
    def tools(self) -> list:
        return [self.save_preference, self.recall_preferences, self.save_feedback]

    def read(self, store: Any, user_id: str) -> dict[str, Any]:
        if store is None or not user_id:
            return self.blank()
        current = store.get((user_id,), "profile")
        if current is None:
            return self.blank()
        return dict(current.value)

    def write(self, store: Any, user_id: str, profile: dict[str, Any]) -> None:
        store.put((user_id,), "profile", profile)

    def remember(self, store: Any, user_id: str, note: str, thread_id: str = "") -> dict[str, Any]:
        profile = self.read(store, user_id)
        self.fold(profile, note, thread_id)
        self.write(store, user_id, profile)
        return profile

    def recall(self, store: Any, user_id: str) -> str:
        return self.render(self.read(store, user_id), user_id)

    def record(
        self,
        *,
        user_id: str,
        rating: int,
        scope: str,
        reason: str | None = None,
        corrected_answer: str | None = None,
        consent_for_training: bool = False,
        messages: list | None = None,
        thread_id: str = "",
        response_id: str = "",
    ) -> dict[str, Any]:
        """Store one feedback example. Personal notes are not a shop fact."""

        if scope not in {"personal", "shared"}:
            raise ValueError("Feedback scope must be personal or shared.")
        row = {
            "id": f"fb-{len(self.feedback) + 1}",
            "thread_id": thread_id,
            "customer_id": user_id,
            "user_id": user_id,
            "response_id": response_id,
            "scope": scope,
            "rating": rating,
            "reason": (reason or "")[:500],
            "corrected_answer": (corrected_answer or "")[:500],
            "consent_for_training": consent_for_training,
            "used_for_training": False,
            "example": build_example(messages or [], reason, corrected_answer),
        }
        self.feedback.append(row)
        return row

    def review(
        self,
        *,
        prompts: list[str] | tuple[str, ...] | str = (),
        prompt: str = "",
        tools: list | tuple = (),
        knowledge: Any = (),
        knowledge_titles: list[str] | tuple[str, ...] = (),
        mark_used: bool = True,
    ) -> dict[str, Any]:
        """Suggest changes from consented feedback that has not been reviewed.

        ``prompts`` is the list of standing instructions. ``knowledge`` may be
        those texts or a LangChain vector store.
        """

        chosen = tools or self.tools
        return review_feedback(
            self.feedback,
            prompts=prompts,
            prompt=prompt,
            tools=chosen,
            knowledge=knowledge,
            knowledge_titles=knowledge_titles,
            tool_gaps=self.tool_gaps,
            personal_markers=self.personal_markers or ("size", "arabic", "english", "for my", "for me", "i wear", "i prefer", "my gift"),
            mark_used=mark_used,
        )

    def prepare(self, request: ModelRequest) -> ModelRequest:
        user_id = user_of(request.runtime.context)
        question = _question(request)
        found = self.lines(self.read(request.runtime.store, user_id), question) if user_id else []
        extra = ""
        if found:
            extra = f" Personal memory for {self.audience} only: " + "; ".join(found)
        if self.on_inject:
            self.on_inject(user_id, extra)
        if not extra:
            return request
        current = ""
        if request.system_message is not None:
            content = request.system_message.content
            current = content if isinstance(content, str) else str(content)
        return request.override(system_message=SystemMessage(content=f"{current}{extra}".strip()))

    def _save_tool(self):
        owner = self

        @tool
        def save_preference(note: str, runtime: ToolRuntime) -> str:
            """Customize future answers for this user only.

            Use this for language, length, a name, or what they want next.
            Do not use it to change the agent for every user.
            """

            user_id = user_of(runtime.context)
            if runtime.store is None or not user_id:
                return "No personal memory is attached to this chat."
            thread_id = ""
            if runtime.config:
                thread_id = str((runtime.config.get("configurable") or {}).get("thread_id") or "")
            owner.remember(runtime.store, user_id, note, thread_id)
            if owner.on_save:
                owner.on_save(user_id, note[:80])
            return owner.recall(runtime.store, user_id)

        return save_preference

    def _recall_tool(self):
        owner = self

        @tool
        def recall_preferences(runtime: ToolRuntime) -> str:
            """Read this user's personal memory. It does not include anyone else's."""

            user_id = user_of(runtime.context)
            if runtime.store is None or not user_id:
                return "No personal memory is attached to this chat."
            if owner.on_recall:
                owner.on_recall(user_id)
            return owner.recall(runtime.store, user_id)

        return recall_preferences

    def _feedback_tool(self):
        owner = self

        @tool
        def save_feedback(
            rating: int,
            scope: str,
            reason: str,
            corrected_answer: str,
            consent_for_training: bool,
            runtime: ToolRuntime,
        ) -> str:
            """File a judgment that should make the agent better for every user.

            scope shared: a wrong answer, a missing fact, or a missing capability.
            scope personal: only when they say the correction is for them alone.
            corrected_answer is the fact or capability every user should get.
            Never put an apology there.
            consent_for_training is true only when they agree to keep the example.
            """

            user_id = user_of(runtime.context)
            parsed = {
                "rating": int(rating) if str(rating).isdigit() else rating,
                "scope": scope,
                "reason": reason,
                "corrected_answer": corrected_answer,
                "consent_for_training": consent_for_training,
            }
            if not feedback_ready(parsed):
                return feedback_ask(parsed)
            if not user_id:
                return "No user is attached to this chat."
            state = getattr(runtime, "state", None) or {}
            messages = state.get("messages", []) if isinstance(state, dict) else []
            thread_id = ""
            if runtime.config:
                thread_id = str((runtime.config.get("configurable") or {}).get("thread_id") or "")
            text = (corrected_answer or "").strip()
            if scope == "shared" and not states_a_fact(text):
                text = (reason or text).strip()
            owner.record(
                user_id=user_id,
                thread_id=thread_id,
                rating=int(rating),
                scope=scope,
                reason=reason or None,
                corrected_answer=corrected_answer or None,
                consent_for_training=bool(consent_for_training),
                messages=messages,
            )
            if scope == "personal" and runtime.store is not None:
                note = (reason or corrected_answer or "").strip()
                if note:
                    owner.remember(runtime.store, user_id, note, thread_id)
            if owner.on_feedback:
                owner.on_feedback(user_id, f"{scope} {rating}")
            kept = "Consent is stored for a future training set." if consent_for_training else "No training example was kept."
            if scope == "personal":
                return f"Saved personal feedback, rating {rating}, on this user's profile only. {kept}"
            if text:
                return f"Saved shared feedback, rating {rating}. It can be reviewed for every user. {kept} Suggested note: {text[:180]}"
            return f"Saved shared feedback, rating {rating}. It can be reviewed for every user. {kept}"

        return save_feedback


class _PersonalizeMiddleware(AgentMiddleware):
    def __init__(self, owner: Personalize) -> None:
        super().__init__()
        self.owner = owner

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self.owner.prepare(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        return await handler(self.owner.prepare(request))


def _question(request: ModelRequest) -> str:
    if not request.messages:
        return ""
    last = request.messages[-1]
    content = getattr(last, "content", "")
    return content if isinstance(content, str) else str(content)
