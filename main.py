# main.py

from graph import create_app, CHECKPOINTER_NAME
from langchain_core.messages import HumanMessage, AIMessage

import json

app = create_app()

def summarize_state_snapshot(snapshot_values: dict) -> str:
    """Create a small human-readable summary of the current LangGraph state."""
    messages = snapshot_values.get("messages", [])
    structured = snapshot_values.get("structured_answer", None)

    num_messages = len(messages)
    last_user = None
    last_answer = None

    # Walk messages from the end backwards
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and last_user is None:
            last_user = msg.content
        if isinstance(msg, AIMessage) and last_answer is None:
            last_answer = msg.content
        if last_user and last_answer:
            break

    summary = {
        "num_messages": num_messages,
        "last_user_message": last_user,
        "last_answer_preview": (last_answer[:120] + "...") if last_answer and len(last_answer) > 120 else last_answer,
        "structured_answer": structured,
    }
    return json.dumps(summary, indent=2, ensure_ascii=False)


def chat():
    print(f"[Checkpointer] Active implementation: {CHECKPOINTER_NAME}")
    print("LMKR RAG Chatbot (type 'exit' to quit)\n")
    thread = {"configurable": {"thread_id": "local"}}

    while True:
        user = input("You: ")

        if user.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        inputs = {
            "messages": [HumanMessage(content=user)],
            "retrieved_docs": [],
            "structured_answer": None
        }

        print("Assistant:", end=" ", flush=True)

        # ---- STREAMING LOOP (handles both dict and tuple events) ----
        for event in app.stream(inputs, config=thread, stream_mode="messages"):

            # Case 1: event is a dict: { node_name: [messages] }
            if isinstance(event, dict):
                for node_name, msgs in event.items():
                    for msg in msgs:
                        if isinstance(msg, AIMessage):
                            print(msg.content, end="", flush=True)

            # Case 2: event is a tuple: (event_type, payload)
            elif isinstance(event, tuple):
                event_type, payload = event
                if isinstance(payload, list):
                    for msg in payload:
                        if isinstance(msg, AIMessage):
                            print(msg.content, end="", flush=True)

        print("\n" + "-" * 50)

        # ---- STATE SNAPSHOT AFTER EACH TURN ----
        state = app.get_state(thread)
        snapshot_summary = summarize_state_snapshot(state.values)
        print("[State Snapshot]")
        print(snapshot_summary)
        print("=" * 50)


if __name__ == "__main__":
    chat()
