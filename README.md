# personalize

Personal memory, feedback gathering, and a monthly review for any LangChain agent.

A preference customizes answers for one user. Feedback, with consent, waits for a review. The review reads that agent's standing prompts, its tools, and a knowledge base (a list of texts or a LangChain vector store) and suggests a handbook note, a prompt change, a tool, or leaving the row on that user's profile. It does not edit the agent.

```bash
pip install git+https://github.com/eslamzaaki/personalize.git
```

```python
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
```

`personal.tools` is `save_preference`, `recall_preferences`, and `save_feedback`. The model never receives the user id.

```python
report = personal.review(
    prompts=["The standing instructions for this agent.", "A second prompt, if the agent has one."],
    tools=personal.tools,
    knowledge=vector_store,  # any LangChain vector store, or a list of texts
)
```

`report["suggestions"]` says handbook, prompt, tool, or leave it on that user. A second call skips rows the first call reviewed. Pass `fold`, `lines`, `render`, and `tool_gaps` when the profile or the missing tools are not the defaults.

The [Lumen counter](https://github.com/eslamzaaki/city-desk) is an example that installs this library and wires it in `citydesk/personal.py`.

## Tests

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

On Windows, use `.venv\Scripts\pip` and `.venv\Scripts\pytest`.
