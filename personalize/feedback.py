"""Feedback examples and the monthly review for any agent.

A preference customizes one user. Feedback is a judgment that should make the
agent better for every user. ``review_feedback`` groups consented rows that
have not been reviewed and suggests a knowledge note, a prompt change, a tool,
or leaving the note on that user. It does not edit the agent.
"""

from __future__ import annotations

import re
from typing import Any

_RATING = re.compile(r"\brating\s*([1-5])\b|\b([1-5])\s*/\s*5\b", re.IGNORECASE)
_RIGHT = re.compile(r"the right answer is\s+(.+)", re.IGNORECASE)
_BECAUSE = re.compile(r"because\s+(.+)", re.IGNORECASE)
_NO_CONSENT = re.compile(r"\b(no training|don't keep|do not keep|no consent)\b", re.IGNORECASE)
_YES_CONSENT = re.compile(r"\b(training example|keep this example|you may keep|consent)\b", re.IGNORECASE)
_NOT_A_FACT = (
    "i'm sorry",
    "i am sorry",
    "i can't",
    "i cannot",
    "i can not",
    "i don't",
    "i do not",
    "i shouldn't",
    "i should not",
)
_PERSONAL = ("size", "arabic", "english", "for my", "for me", "i wear", "i prefer", "my gift")


def parse_feedback(text: str) -> dict:
    """Pull rating, scope, reason, correction, and consent out of one message."""

    rating_match = _RATING.search(text)
    rating = None
    if rating_match:
        rating = int(next(group for group in rating_match.groups() if group))
    lowered = text.lower()
    scope = None
    if "shared" in lowered or "handbook" in lowered:
        scope = "shared"
    elif "personal" in lowered or "for me" in lowered or "only me" in lowered:
        scope = "personal"
    corrected = ""
    right = _RIGHT.search(text)
    if right:
        corrected = right.group(1).strip().strip(".")
    reason = ""
    because = _BECAUSE.search(text)
    if because:
        reason = because.group(1).strip().strip(".")
    consent = None
    if _NO_CONSENT.search(text):
        consent = False
    elif _YES_CONSENT.search(text):
        consent = True
    return {
        "rating": rating,
        "scope": scope,
        "reason": reason,
        "corrected_answer": corrected,
        "consent_for_training": consent,
    }


def feedback_ready(parsed: dict) -> bool:
    return (
        parsed["rating"] is not None
        and parsed["scope"] in {"personal", "shared"}
        and bool(parsed["reason"] or parsed["corrected_answer"])
    )


def feedback_ask(parsed: dict) -> str:
    missing: list[str] = []
    if parsed["rating"] is None:
        missing.append("a rating from 1 to 5")
    if parsed["scope"] is None:
        missing.append("whether it is personal (only this user) or shared (every user)")
    if not parsed["reason"] and not parsed["corrected_answer"]:
        missing.append("the right answer, or why the reply missed")
    consent = ""
    if parsed["consent_for_training"] is None:
        consent = " Add “keep this example” if I may store it for a future training set."
    return f"I can file that on this chat. I still need {', '.join(missing)}.{consent}"


def states_a_fact(text: str) -> bool:
    """A shared note states a fact for every user. An apology does not."""

    lowered = (text or "").lower()
    if not lowered.strip():
        return False
    return not any(phrase in lowered for phrase in _NOT_A_FACT)


def build_example(messages: list, reason: str | None, corrected_answer: str | None) -> dict[str, str]:
    """Shape one rating into the question, the judged reply, and the user's own words."""

    rows = [row for row in (_row(message) for message in messages) if row]
    start = _feedback_start(rows)
    feedback_message = "\n".join(row["text"] for row in rows[start:] if row["role"] == "human" and row["text"]).strip()
    earlier = rows[:start]
    judged_reply = ""
    judged_index = None
    for index in range(len(earlier) - 1, -1, -1):
        if earlier[index]["role"] == "ai" and earlier[index]["text"]:
            judged_reply = earlier[index]["text"]
            judged_index = index
            break
    question = ""
    if judged_index is not None:
        for index in range(judged_index - 1, -1, -1):
            if earlier[index]["role"] == "human" and earlier[index]["text"]:
                question = earlier[index]["text"]
                break
    problem = (reason or "").strip() or feedback_message
    preferred = (corrected_answer or "").strip()
    return {
        "question": question[:500],
        "judged_reply": judged_reply[:500],
        "feedback_message": feedback_message[:500],
        "problem": problem[:500],
        "preferred_reply": preferred[:500],
    }


def review_feedback(
    rows: list[dict[str, Any]],
    *,
    prompts: list[str] | tuple[str, ...] | str = (),
    prompt: str = "",
    tools: list | tuple = (),
    knowledge: Any = (),
    knowledge_titles: list[str] | tuple[str, ...] = (),
    tool_gaps: tuple[tuple[str, str, str], ...] = (),
    personal_markers: tuple[str, ...] = _PERSONAL,
    mark_used: bool = True,
) -> dict[str, Any]:
    """Suggest where unused consented feedback should change the agent.

    ``prompts`` is every standing instruction the agent uses. ``knowledge`` is
    either those texts, or a LangChain vector store searched with each correction.
    """

    pending = [row for row in rows if row.get("consent_for_training") and not row.get("used_for_training")]
    prompt_list = _prompt_list(prompts, prompt)
    known = _Knowledge(knowledge, knowledge_titles)
    names = " ".join(getattr(tool, "name", str(tool)) for tool in tools)
    suggestions = [
        _suggestion(text, group, prompt_list, known, names, tool_gaps, personal_markers)
        for text, group in _groups(pending).items()
    ]
    suggestions.sort(key=lambda item: (-item["count"], item["where"], item["what"]))
    if mark_used:
        for row in pending:
            row["used_for_training"] = True
    return {
        "reviewed": len(pending),
        "prompts": prompt_list,
        "prompt_in_use": "\n".join(prompt_list),
        "tools": [getattr(tool, "name", str(tool)) for tool in tools],
        "handbook": known.titles,
        "suggestions": suggestions,
    }


def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        preferred, problem = _review_text(row)
        key = re.sub(r"\s+", " ", (preferred or problem).lower())
        grouped.setdefault(key, []).append(row)
    return grouped


def _review_text(row: dict[str, Any]) -> tuple[str, str]:
    example = row.get("example") or {}
    preferred = (example.get("preferred_reply") or row.get("corrected_answer") or "").strip()
    problem = (example.get("problem") or row.get("reason") or example.get("feedback_message") or "").strip()
    return preferred, problem


def _suggestion(
    text: str,
    rows: list[dict[str, Any]],
    prompts: list[str],
    knowledge: "_Knowledge",
    tool_names: str,
    tool_gaps: tuple[tuple[str, str, str], ...],
    personal_markers: tuple[str, ...],
) -> dict[str, Any]:
    preferred, problem = _review_text(rows[0])
    where, why = _place(
        preferred or text,
        problem,
        rows[0].get("scope", ""),
        prompts,
        knowledge,
        tool_names,
        tool_gaps,
        personal_markers,
    )
    shown = (preferred or problem or text) or "No correction text."
    return {
        "where": where,
        "what": _what(where, shown),
        "why": why,
        "problem": problem,
        "count": len(rows),
        "feedback_ids": [row["id"] for row in rows],
    }


def _place(
    text: str,
    problem: str,
    scope: str,
    prompts: list[str],
    knowledge: "_Knowledge",
    tool_names: str,
    tool_gaps: tuple[tuple[str, str, str], ...],
    personal_markers: tuple[str, ...],
) -> tuple[str, str]:
    lowered = f"{text}\n{problem}".lower()
    if scope != "shared" or any(marker in lowered for marker in personal_markers):
        return "ignore", "This is about one user. Leave it on their profile."
    for needle, _name, why in tool_gaps:
        if needle in lowered and needle not in tool_names:
            return "tool", why
    prompt_blob = "\n".join(prompts).lower()
    if text and _covered(lowered, prompt_blob):
        return "prompt", "A standing prompt already says this, and users still correct the reply."
    if text and knowledge.covers(text):
        return "prompt", "The knowledge base already says this, and users still correct the reply."
    return "handbook", "Agreed fact that is not in the knowledge base yet."


class _Knowledge:
    """Texts, or a LangChain vector store searched per correction."""

    def __init__(self, knowledge: Any, titles: list[str] | tuple[str, ...]) -> None:
        self.titles = [str(title) for title in titles]
        self._store = knowledge if _is_store(knowledge) else None
        self._blob = ""
        if self._store is None:
            texts = [knowledge] if isinstance(knowledge, str) else list(knowledge or [])
            self._blob = "\n".join(str(text) for text in texts).lower()
        elif not self.titles:
            self.titles = _titles_from_store(self._store)

    def covers(self, text: str) -> bool:
        if self._store is None:
            return _covered(text.lower(), self._blob)
        docs = _search(self._store, text)
        for doc in docs:
            title = _doc_title(doc)
            if title and title not in self.titles:
                self.titles.append(title)
        blob = "\n".join(_doc_text(doc).lower() for doc in docs)
        return _covered(text.lower(), blob)


def _prompt_list(prompts: list[str] | tuple[str, ...] | str, prompt: str) -> list[str]:
    items: list[str] = []
    if isinstance(prompts, str):
        items.append(prompts)
    else:
        items.extend(str(item) for item in prompts if str(item).strip())
    if prompt and prompt not in items:
        items.append(prompt)
    return items


def _is_store(knowledge: Any) -> bool:
    return hasattr(knowledge, "similarity_search") or hasattr(knowledge, "similarity_search_with_score")


def _search(store: Any, query: str, k: int = 4) -> list:
    try:
        if hasattr(store, "similarity_search_with_score"):
            hits = store.similarity_search_with_score(query, k=k)
            return [hit[0] if isinstance(hit, tuple) else hit for hit in hits]
        return list(store.similarity_search(query, k=k))
    except ValueError:
        return []


def _doc_text(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("text") or doc.get("page_content") or "")
    return str(getattr(doc, "page_content", "") or "")


def _doc_title(doc: Any) -> str:
    meta = doc.get("metadata") if isinstance(doc, dict) else getattr(doc, "metadata", None)
    meta = meta or {}
    return str(meta.get("title") or meta.get("id") or "")


def _titles_from_store(store: Any) -> list[str]:
    raw = getattr(store, "store", None)
    if not isinstance(raw, dict):
        return []
    titles: list[str] = []
    for item in raw.values():
        title = _doc_title(item if isinstance(item, dict) else item)
        if title and title not in titles:
            titles.append(title)
    return titles


def _covered(lowered: str, knowledge: str) -> bool:
    words = [word for word in re.findall(r"[a-z0-9]+", lowered) if len(word) > 3]
    if len(words) < 3:
        return lowered in knowledge
    hits = sum(1 for word in words if word in knowledge)
    return hits / len(words) >= 0.6


def _what(where: str, text: str) -> str:
    if where == "handbook":
        return f"Add a handbook note: {text}"
    if where == "prompt":
        return f"Tell the agent to answer from the handbook before guessing: {text}"
    if where == "tool":
        return f"Add a tool for this request: {text}"
    return f"Do not train on this: {text}"


def _row(message: object) -> dict[str, str] | None:
    if isinstance(message, dict):
        role = str(message.get("type") or message.get("role") or "")
        content = message.get("content") or ""
    else:
        role = str(getattr(message, "type", "") or "")
        content = getattr(message, "content", "")
    text = content if isinstance(content, str) else _blocks(content)
    role = {"user": "human", "assistant": "ai"}.get(role, role)
    if role not in {"human", "ai"}:
        return None
    return {"role": role, "text": text.strip()}


def _blocks(content: object) -> str:
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


def _feedback_start(rows: list[dict[str, str]]) -> int:
    start = len(rows)
    seen_human = False
    for index in range(len(rows) - 1, -1, -1):
        if rows[index]["role"] != "human":
            continue
        if not seen_human or _looks_like_feedback(rows[index]["text"]):
            start = index
            seen_human = True
            continue
        break
    return start


def _looks_like_feedback(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("that was wrong", "that reply", "that answer", "rate this", "rating", "/5")):
        return True
    return "the right answer is" in lowered
