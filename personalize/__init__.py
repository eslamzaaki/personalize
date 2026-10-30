"""Personal memory, feedback, and a monthly review for any LangChain agent.

Pass ``middleware`` and ``tools`` into ``create_agent``. Invoke with a context
that carries ``user_id`` (or ``customer_id``). The agent's store holds the
profile. The model never chooses whose profile to open.

.. code-block:: python

    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from personalize import Personalize, UserContext

    personal = Personalize()
    agent = create_agent(
        model="openai:gpt-6.1-sol",
        tools=[*personal.tools],
        middleware=[personal.middleware],
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "I prefer short answers."}]},
        {"configurable": {"thread_id": "t1"}},
        context=UserContext(user_id="sara"),
    )
"""

from personalize.kit import Personalize, UserContext

__all__ = ["Personalize", "UserContext"]
