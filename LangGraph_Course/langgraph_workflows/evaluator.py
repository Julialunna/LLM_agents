from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from models import models
import logging


#states
class StoryState(TypedDict):
    prompt: str
    story: str
    feedback:str
    act:int
    
#nodes
def generate_story(state: StoryState):
    """Gera um novo trecho da história baseado no feedcack"""
    feedback = state.get("feedback", "")
    story = state.get("story", "")
    act = state.get("act", 1)
    
    act_descriptions = {
        1: "primeiro ato, onde apresentamos os personagens e o conflito inicial", 
        2: "segundo ate, onde o conflito se intensifica", 
        3: "terceiro ato, onde chegamos ao clímax da história", 
        4: "quarto aot, onde resolvemos o conflito e concluímos a história"
    }
    
    msg = f"""
    Você é um escritor criativo especializado em criar histórias envolventes.
    Sua tarefa é continuar a história {act_descriptions[act]}.
    
    ***Instruções:***
    -Crie um trecho da história que se conecte naturalmente
    com o que já foi escrito 
    -Mantenha consistência com os personagens e eventos anteriores.
    -Seja criativo e envolvente.
    
    Prompt Inicial:
    {state["prompt"]}

    História até agora:
    {story}
    
    Feedback anterior:
    {feedback}
    
    Escreva apenas o novo trecho da história, sem explicações adicionais 
    """
    messages =[ HumanMessage(content=msg)]
    response = models["meta_llama_3"].invoke(messages)
    
    new_story = story + "\n\n" +response.content if story else response.content
    return{"story":new_story}

def get_feedback(state:StoryState):
    """Solicita feedback do usuário sobre a história"""    
    feedback = interrupt(
        {
            "prompt": state["prompt"], 
            "story": state["story"], 
            "act": state["act"]
        }
    )
    logging.info(feedback)
    return{"feedback": feedback}

def route_story(state:StoryState):
    """Roteria para próximo ato ou finaliza a história"""
    f=state.get("feedback")
    act = state.get("act")
    
    if f == 'aprovado':
        if(act < 4):
            return "next_act"
        return "final"
    return "revise"
    

def next_act(state:StoryState):
    """Avança para o próximo ato"""
    current_act = state.get("act",1)
    return {"act": current_act+1}

def final_story(state:StoryState):
    """Finaliza workflow e retorna história completa"""
    return {"story": state["story"]}

builder = StateGraph(StoryState)
builder.add_node("generate_story", generate_story)
builder.add_node("get_feedback", get_feedback)
builder.add_node("next_act", next_act)
builder.add_node("final_story", final_story)

builder.add_edge(START, "generate_story")
builder.add_edge( "generate_story", "get_feedback")
builder.add_conditional_edges(
    "get_feedback", route_story, {
        "revise": "generate_story", 
        "next_act": "next_act", 
        "final": "final_story"
    }
)

builder.add_edge("next_act", "generate_story")
builder.add_edge("final_story", END)

story_workflow = builder.compile()