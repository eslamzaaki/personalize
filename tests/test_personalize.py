"""The personalization library customizes one user and leaves the other alone."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langgraph.store.memory import InMemoryStore

from personalize import Personalize, UserContext


class _Store:
    """A stand-in with the LangChain vector-store search method."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        words = [word for word in query.lower().split() if len(word) > 3]
        hits = [doc for doc in self.documents if any(word in doc.page_content.lower() for word in words)]
        return hits[:k]


def test_two_users_do_not_share_notes():
    store = InMemoryStore()
    kit = Personalize()
    kit.remember(store, "sara", "Answer in Arabic.")
    kit.remember(store, "omar", "Answer in English.")

    assert "Arabic" in kit.recall(store, "sara")
    assert "Arabic" not in kit.recall(store, "omar")
    assert "English" in kit.recall(store, "omar")


def test_review_groups_feedback_for_any_agent():
    kit = Personalize(
        tool_gaps=(("address", "update_delivery", "Customers ask to change a delivery address. No tool does that."),)
    )

    def keep(scope: str, text: str, consent: bool = True) -> None:
        kit.record(user_id="sara", rating=2, scope=scope, corrected_answer=text, consent_for_training=consent)

    keep("shared", "Gift cards are non-returnable.")
    keep("personal", "Use 14 days for my gifts.")
    keep("shared", "The return window is 30 days after delivery.")
    keep("shared", "Please change the delivery address.")
    keep("shared", "A secret shop rule.", False)
    report = kit.review(
        prompts=["Answer from the notes.", "Check the handbook before you state a shop rule."],
        tools=[],
        knowledge=["The return window is 30 days after delivery."],
        knowledge_titles=["Returns"],
    )
    by_where = {item["where"]: item for item in report["suggestions"]}

    assert report["reviewed"] == 4
    assert "gift cards" in by_where["handbook"]["what"].lower()
    assert "handbook" in by_where["prompt"]["what"].lower()
    assert "address" in by_where["tool"]["what"].lower()
    assert by_where["ignore"]["where"] == "ignore"
    assert kit.review(knowledge=["The return window is 30 days after delivery."])["reviewed"] == 0


def test_review_reads_a_vector_store():
    kit = Personalize()
    kit.record(
        user_id="sara",
        rating=1,
        scope="shared",
        corrected_answer="The return window is 30 days after delivery.",
        consent_for_training=True,
    )
    store = _Store(
        [Document(page_content="The return window is 30 days after delivery.", metadata={"title": "Returns"})]
    )
    report = kit.review(prompts=["Answer from the notes."], tools=[], knowledge=store)

    assert report["handbook"] == ["Returns"]
    assert report["prompts"] == ["Answer from the notes."]
    assert report["suggestions"][0]["where"] == "prompt"


def test_middleware_appends_memory_without_replacing_the_prompt():
    store = InMemoryStore()
    kit = Personalize(audience="this customer")
    kit.remember(store, "sara", "Keep answers short.")
    request = SimpleNamespace(
        runtime=SimpleNamespace(context=UserContext(user_id="sara"), store=store),
        messages=[SimpleNamespace(content="Hello")],
        system_message=SystemMessage(content="You are helpful."),
        override=lambda **kwargs: SimpleNamespace(system_message=kwargs["system_message"]),
    )

    prepared = kit.prepare(request)

    text = prepared.system_message.content
    assert text.startswith("You are helpful.")
    assert "Keep answers short." in text
    assert "this customer only" in text
