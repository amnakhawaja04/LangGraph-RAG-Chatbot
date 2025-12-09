# utils.py
from langchain_core.messages import HumanMessage, AIMessage

SYSTEM_PROMPT = """
You are LMKR's internal AI assistant.

You answer questions using the provided LMKR knowledge base (RAG context) when available.
Your answers should be **paragraph-style**, clear, and well-structured, typically 3–6 sentences,
not just one short line.

When giving your FINAL answer, you MUST reply ONLY in valid JSON:

{
  "answer": "<a helpful multi-sentence paragraph answer to the user>",
  "sources": ["<source_1>", "<source_2>", "..."]
}

Rules:
- "answer" should be a coherent paragraph, not bullet points, not a single sentence.
- "sources" should be a list of filenames, document ids, or short descriptors of where you found the information.
- Do NOT include markdown formatting (no ``` fences, no **bold**, etc.).
- Output MUST be valid JSON and nothing else.
"""


def build_prompt(state):
    parts = [SYSTEM_PROMPT.strip(), "\n\nConversation so far:\n"]

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            role = "Other"
        parts.append(f"{role}: {msg.content}")

    if state["retrieved_docs"]:
        parts.append("\n\nContext from LMKR knowledge base:\n")
        for i, doc in enumerate(state["retrieved_docs"], 1):
            parts.append(f"[DOC {i}]\n{doc}")

    parts.append("\nNow produce ONLY the FINAL JSON answer as specified above.\n")
    return "\n".join(parts)
