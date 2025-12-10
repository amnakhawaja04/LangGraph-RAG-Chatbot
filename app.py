# streamlit_app.py

import json
import uuid
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from graph import create_app, CHECKPOINTER_NAME


# Center layout, wide look, without sidebar
st.set_page_config(page_title="LMKR RAG Chatbot", page_icon="💬", layout="wide")


# --------------------------------------------------------
# Helper: Summarize LangGraph state snapshot
# --------------------------------------------------------
def summarize_state_snapshot(snapshot_values: dict) -> str:
    messages = snapshot_values.get("messages", [])
    structured = snapshot_values.get("structured_answer", None)

    last_user = None
    last_assistant = None

    for m in reversed(messages):
        if isinstance(m, HumanMessage) and last_user is None:
            last_user = m.content
        if isinstance(m, AIMessage) and last_assistant is None:
            last_assistant = m.content
        if last_user and last_assistant:
            break

    summary = {
        "num_messages": len(messages),
        "last_user_message": last_user,
        "last_answer_preview": (
            last_assistant[:120] + "..."
            if last_assistant and len(last_assistant) > 120
            else last_assistant
        ),
        "structured_answer": structured,
    }

    return json.dumps(summary, indent=2, ensure_ascii=False)



# --------------------------------------------------------
# Load LangGraph App
# --------------------------------------------------------
app = create_app()

# Center alignment container
container = st.container()

with container:
    st.title("💬 LMKR RAG Chatbot")
    st.caption(f"LangGraph + FAISS RAG • Checkpointer: **{CHECKPOINTER_NAME}**")



# --------------------------------------------------------
# Chat State Initialization
# --------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread" not in st.session_state:
    st.session_state.thread = {
        "configurable": {"thread_id": f"web-{uuid.uuid4().hex[:8]}"}
    }


# --------------------------------------------------------
# Display conversation so far
# --------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



# --------------------------------------------------------
# Chat Input
# --------------------------------------------------------
user_input = st.chat_input("Ask me something about LMKR / GVERSE / Geoscience...")



# --------------------------------------------------------
# When user sends a message
# --------------------------------------------------------
if user_input:

    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # PREPARE LLM INPUT
    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "retrieved_docs": [],
        "structured_answer": None,
    }
    thread = st.session_state.thread

    # --------------------------------------------------------
    # Assistant message placeholder + Loading Indicator
    # --------------------------------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()

        # Show loading / typing indicator
        typing_indicator = st.empty()
        typing_indicator.markdown("⏳ **Assistant is typing…**")

        full_response = ""

        # STREAM EVENTS FROM LangGraph
        for event in app.stream(inputs, config=thread, stream_mode="messages"):

            # Case 1: {node_name: [messages]}
            if isinstance(event, dict):
                for node, items in event.items():
                    for item in items:
                        if hasattr(item, "content"):
                            full_response += str(item.content)
                            placeholder.markdown(full_response)
                        elif isinstance(item, str):
                            full_response += item
                            placeholder.markdown(full_response)

            # Case 2: (event_type, payload)
            elif isinstance(event, tuple):
                event_type, payload = event

                if isinstance(payload, list):
                    for item in payload:
                        if hasattr(item, "content"):
                            full_response += str(item.content)
                            placeholder.markdown(full_response)
                        elif isinstance(item, str):
                            full_response += item
                            placeholder.markdown(full_response)

                elif isinstance(payload, str):
                    full_response += payload
                    placeholder.markdown(full_response)

        # Remove typing indicator after streaming ends
        typing_indicator.empty()

        # FALLBACK if nothing streamed
        if not full_response.strip():
            final_state = app.get_state(thread)
            msgs = final_state.values.get("messages", [])
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    full_response = m.content
                    break

            placeholder.markdown(full_response or "(No output produced.)")

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    # --------------------------------------------------------
    # Debug Snapshot (collapsible)
    # --------------------------------------------------------
    final_state = app.get_state(thread)
    snapshot = summarize_state_snapshot(final_state.values)

    with st.expander("📊 LangGraph state snapshot (debug)", expanded=False):
        st.code(snapshot, language="json")
