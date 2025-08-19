import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

class NewsAgentState(TypedDict):
    original_news: str
    plan: List[Dict[str, str]] | None
    current_task_idx: int
    intermediate_results: Dict[str, str]
    final_response: str | None
    error: str | None
    current_task_description: str | None
    current_specialist_type: str | None
    current_task_id: str | None
    specialist_result: str | None
    