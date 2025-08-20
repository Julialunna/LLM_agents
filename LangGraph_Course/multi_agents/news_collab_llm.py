import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from models import models

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

def planner_node(state: NewsAgentState) -> Dict[str, Any]:
    news = state["original_news"]
    plan= [
        {
            "task_id": "summarize_news",
            "specialist_type": "summarizer",
            "description": f"Resuma a seguinte notícia de forma clara e objetiva: {news}"
        },
        {
            "task_id": "analyze_news",
            "specialist_type": "analyst",
            "description": "Analise o resumo da notícia, destacando pontos importantes, possíveis vieses e impactio social/político"

        },
        {
            "task_id": "suggest_questions",
            "specialist_type": "questioner",
            "description": "Sugira perguntas para reflexão oude debate baseados na análise da na análise da notícia. Use o resultado da tarefa"
        }
    ]
    return{
        "plan": plan,
        "current_task_idx": 0,
        "intermediate_results": {},
        "error": None,
        "specialist_result": None
    }
    
def prepare_next_task_node(state: NewsAgentState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)
    if not plan or current_task_idx >= len(plan):
        return {"current_task_id": None, 
                "current_task_description": None,
                "current_specialist_type": None}
    current_task =  plan[current_task_idx]
    return{
        "current_task_id":current_task["task_id"],
        "current_task_description": current_task["description"],
        "current_specialist_type": current_task["specialist_type"]
    }    
    
def summarizer_node(state: NewsAgentState) -> Dict[str, str]:
    news = state.get("original_news", "")
    prompt = f"Resuma a seguinte notícia de forma clara, objetiva e em até 5 linhas:\n{news}"
    try:
        response=models["meta_llama_3"].invoke([HumanMessage(content = prompt)])
        return {"specialist_result": response.content.strip()}
    except Exception as e:
        return {"specialist_result": f"Erro ao resumir: {e}"}
    
  

def analyst_node(state: NewsAgentState) -> Dict[str, str]:
    prev_results = state.get("intermediate_results", {})
    resumo = prev_results.get("summarize_news", "sem resumo")
    prompt = (
        f"Analise o seguinte resumo de notícia, destacando pontos importantes, possíveis vieses e impacto social/político.\n"
        f"Resumo: {resumo}"
    )
    
    try:
        response = models["meta_llama_3"].invoke([HumanMessage(content=prompt)])
        return {"specialist_result": response.content.strip()}
    except Exception as e:
        return {"specialist_result": f"Erro na análise: {e}"}
    

def questioner_node(state:NewsAgentState) -> Dict[str, str]:
    prev_results = state.get("intermediate_results", {})
    analise = prev_results.get("analyzie_news", "sem análise")
    perguntas = [
        "Quais as fontes dessa notícia?",
        "Como essa notícia pode afetar diferentes grupos sociais?",
        "Há outros pontos de vista sobre o tema?"
    ]
    return {"specialist_result": "Perguntas para reflexão: " + ", ".join(perguntas)}

def collect_result_and_advance_node(state:NewsAgentState) -> Dict[str, Any]:
    current_task_id = state.get("current_task_id")
    specialist_output = state.get("specialist_result", "Nenhum resultado do especialista encontrado no estado.")
    update_intermediate_results = state.get("intermediate_results", {}).copy()
    if current_task_id:
        update_intermediate_results[current_task_id] = specialist_output
    new_idx= state.get("current_task_idx", 0) + 1
    return {
        "intermediate_results": update_intermediate_results,
        "current_task_idx": new_idx,
        "specialist_result": None
    }
    
def systhesis_node(state:NewsAgentState) -> Dict[str, str] | None:
    original_news = state["original_news"]
    intermediate_results = state.get("intermediate_results", {})
    response = f"Pergunta original: {original_news}\n"
    for task_id, result in intermediate_results.items():
        response+= f"-{task_id}: {result}"   
        
    return{"final_response": response, "error": None}


def should_execute_task_or_sythesize(state:NewsAgentState) -> str:
    if state.get("error"):
        return "error_handler"
    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)
    if(current_task_idx < len(plan)):
        return "prepare_next_task"
    else:
        return "sythesize_response"


def specialist_router_node(state: NewsAgentState) ->str:
    specialist_type = state.get("current_specialist_type")
    if specialist_type == "summarizer":
        return "summarizer"
    elif specialist_type == "analyst":
        return "analyst"
    elif specialist_type == "questioner":
        return "questioner"
    else:
        return "error_handler"

def error_node(state: NewsAgentState)->Dict[str,str]:
    erro_message = state.get("error", "Erro desconhecido no workflow")
    return {"final_response": f"Ocorreu um erro: {erro_message}"}  


#workflow definitiion
workflow_builder = StateGraph(NewsAgentState)
workflow_builder.add_node("planner", planner_node)
workflow_builder.add_node("prepare_next_task",prepare_next_task_node)
workflow_builder.add_node("summarizer", summarizer_node)
workflow_builder.add_node("analyst",analyst_node)
workflow_builder.add_node("questioner",questioner_node)
workflow_builder.add_node("collect_and_advance", collect_result_and_advance_node)
workflow_builder.add_node("sythesize_response", systhesis_node)
workflow_builder.add_node("error_handler", error_node)

workflow_builder.set_entry_point("planner")
workflow_builder.add_conditional_edges(
    "planner", should_execute_task_or_sythesize, {
        "prepare_next_task": "prepare_next_task",
        "sythesize_response": "sythesize_response",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_conditional_edges(
    "prepare_next_task", specialist_router_node, {
        "summarizer": "summarizer",
        "analyst": "analyst",
        "questioner": "questioner",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_edge("summarizer", "collect_and_advance")
workflow_builder.add_edge("analyst", "collect_and_advance")
workflow_builder.add_edge("questioner", "collect_and_advance")
workflow_builder.add_conditional_edges(
    "collect_and_advance", should_execute_task_or_sythesize, {
        "prepare_next_task": "prepare_next_task",
        "sythesize_response": "sythesize_response",
        "error_handler": "error_handler"
    }
)

workflow_builder.add_edge("sythesize_response", END)
workflow_builder.add_edge("error_handler", END)

news_workflow = workflow_builder.compile() 