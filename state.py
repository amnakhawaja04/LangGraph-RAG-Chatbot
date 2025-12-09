# state.py
from typing import Optional, List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[str]
    structured_answer: Optional[dict]
