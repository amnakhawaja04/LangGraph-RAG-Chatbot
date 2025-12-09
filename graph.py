# graph.py

import json
import re

from state import AgentState
from utils import build_prompt
from retriever import load_faiss, faiss_search
from llm import run_llm  # <- from your updated llm.py (InferenceClient wrapper)

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load FAISS index + chunks
index, chunks = load_faiss()

# Checkpointer (in-memory for now)
checkpointer = MemorySaver()
CHECKPOINTER_NAME = type(checkpointer).__name__  # exported so main.py can print it


# -----------------------------------
# Finite State Machine using REGEX
# -----------------------------------
def fsm_route(user_text: str) -> str:
    """
    Very simple FSM using regex to decide next state:
    - If user says bye/exit -> end
    - If greeting -> go directly to generate (no retrieval)
    - If LMKR-related keywords -> retrieve
    - Otherwise -> retrieve by default
    """
    text = user_text.lower()

    # Exit state
    if re.search(r"\b(bye|goodbye|exit|quit|see you)\b", text):
        return "__end__"

    # Greetings / smalltalk -> we can still answer, but skip retrieval
    if re.search(r"\b(hi|hello|hey|salam|assalamualaikum|good morning|good evening)\b", text):
        return "generate_direct"

    # LMKR / GVERSE / domain terms -> use retrieval
    if re.search(r"\b(lmkr|gverse|geoscience|seismic|petrel|reservoir|petrophysics|interpretation)\b", text):
        return "retrieve"

    # Default: treat like an info question and retrieve
    return "retrieve"


# -----------------------------------
# ROUTER NODE
# -----------------------------------
def router_node(state: AgentState) -> AgentState:
    # No state modification here, just pass things through.
    # But we keep keys consistent with AgentState.
    return {
        "messages": state.get("messages", []),
        "retrieved_docs": state.get("retrieved_docs", []),
        "structured_answer": state.get("structured_answer", None),
    }


# -----------------------------------
# ROUTER CONDITION (uses FSM)
# -----------------------------------
def router_cond(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "retrieve"

    last = messages[-1]
    if isinstance(last, HumanMessage):
        route = fsm_route(last.content)
        return route
    return "retrieve"


# -----------------------------------
# RETRIEVE NODE
# -----------------------------------
def retrieve_node(state: AgentState) -> AgentState:
    print("🟦 retrieve_node called")

    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {
            "messages": messages,
            "retrieved_docs": [],
            "structured_answer": state.get("structured_answer", None),
        }

    query = messages[-1].content
    docs = faiss_search(query, index, chunks, k=6)

    retrieved = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        retrieved.append(f"[source: {src}]\n{d.page_content}")

    return {
        "messages": messages,
        "retrieved_docs": retrieved,
        "structured_answer": state.get("structured_answer", None),
    }


# -----------------------------------
# GENERATE NODE (with RAG)
# -----------------------------------
def generate_node(state: AgentState) -> AgentState:
    print("🟩 generate_node called (with RAG)")

    prompt = build_prompt(state)
    raw = run_llm(prompt)

    print("\n===== RAW MODEL OUTPUT =====")
    print(raw)
    print("================================\n")

    try:
        data = json.loads(raw)
        answer = data.get("answer", raw)
        sources = data.get("sources", [])
    except Exception:
        answer = raw
        sources = ["unstructured"]

    msg = AIMessage(content=answer)

    return {
        "messages": state.get("messages", []) + [msg],
        "retrieved_docs": state.get("retrieved_docs", []),
        "structured_answer": {"answer": answer, "sources": sources},
    }


# -----------------------------------
# GENERATE_DIRECT NODE (FSM path: greeting/smalltalk, no retrieval)
# -----------------------------------
def generate_direct_node(state: AgentState) -> AgentState:
    print("🟩 generate_direct_node called (NO RAG)")

    # Temporarily clear retrieved_docs so prompt doesn't mention context
    temp_state = {
        "messages": state.get("messages", []),
        "retrieved_docs": [],
        "structured_answer": state.get("structured_answer", None),
    }

    prompt = build_prompt(temp_state)
    raw = run_llm(prompt)

    print("\n===== RAW MODEL OUTPUT (direct) =====")
    print(raw)
    print("=====================================\n")

    try:
        data = json.loads(raw)
        answer = data.get("answer", raw)
        sources = data.get("sources", [])
    except Exception:
        answer = raw
        sources = ["unstructured"]

    msg = AIMessage(content=answer)

    return {
        "messages": state.get("messages", []) + [msg],
        "retrieved_docs": state.get("retrieved_docs", []),
        "structured_answer": {"answer": answer, "sources": sources},
    }


# -----------------------------------
# CREATE APP
# -----------------------------------
def create_app():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("generate_direct", generate_direct_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        router_cond,
        {
            "retrieve": "retrieve",
            "generate_direct": "generate_direct",
            "__end__": END,
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("generate_direct", END)

    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[]
    )

    # Log that checkpointer is in use
    print(f"[Checkpointer] Using {CHECKPOINTER_NAME} for LangGraph state persistence.")

    return app
