from langchain_community.chat_models import ChatDeepInfra
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel 
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")

#model
llm_model = ChatDeepInfra(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
    temperature=0,  
    api_token=API_KEY
)

#defining graph state
class GraphState(BaseModel):
    input:str
    output:str
    tipo: str = None

def realizar_calculo(state:GraphState) -> GraphState:
    return GraphState(input=state.input, output = "Resposta de cálculo fictício: 42")

def responder_curiosidade(state:GraphState) -> GraphState:
    response = llm_model.invoke([HumanMessage(content=state.input)])
    return GraphState(input = state.input, output = response.content)

#funtion to deal with question non recognized
def responder_erro(state:GraphState) -> GraphState:
    return GraphState(input = state.input, output= "Desculpe, não entendi sua pergunta.")

#node classification function
def classificar(state:GraphState) -> GraphState:
    pergunta = state.input.lower()
    if any(palavra in pergunta for palavra in ["soma", "quanto é", "+", "calcular"]):
        tipo = "calculo"
    elif any(palavra in pergunta for palavra in ["quem", "onde", "por que", "qual"]):
        tipo = "curiosidade"
    else:
        tipo = "desconhecido"
    return GraphState(input = state.input, output="", tipo = tipo)

#creating graph and adding nodes
graph = StateGraph(GraphState)
graph.add_node("classificar", classificar)
graph.add_node("realizar_calculo", realizar_calculo)
graph.add_node("responder_curiosidade", responder_curiosidade)
graph.add_node("responder_erro", responder_erro)

#adding conditionals
graph.add_conditional_edges(
    "classificar",
    lambda state: {
        "calculo": "realizar_calculo",
        "curiosidade": "responder_curiosidade",
        "desconhecido": "responder_erro"
    }[state.tipo] 
)

#defining input and output

graph.set_entry_point("classificar")
graph.set_finish_point(["responder_curiosidade", "realizar_calculo", "responder_erro"])
export_graph = graph.compile()

if __name__=="__main__":
    exemplos = [
        "Quanto é 10 + 5?", 
        "Quem inventou a lâmpada", 
        "Me diga um comando especial"
    ]

    for exemplo in exemplos:
        result = export_graph.invoke(GraphState(input = exemplo, output = ""))
        print(f"Pergunta: {exemplo}\nResposta: {result['output']}\n")

