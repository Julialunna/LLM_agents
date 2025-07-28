from langchain_community.chat_models import ChatDeepInfra
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langchain_core.runnables.graph import MermaidDrawMethod
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

#state graph
class GraphState(BaseModel):
    input: str
    output: str

#answer function
def responder(state):
    input_message = state.input
    response = llm_model.invoke([HumanMessage(content = input_message)])
    return GraphState(input = state.input, output = response.content)

#Graph
graph = StateGraph(GraphState)
graph.add_node("responder", responder)
graph.set_entry_point("responder")
graph.set_finish_point("responder")

#compiling graph
export_graph = graph.compile()

#generanting PNG graphimage 
png_bytes = export_graph.get_graph().draw_mermaid_png(
    draw_method= MermaidDrawMethod.API
)
with open("grafo_exemplo1.png", "wb") as f:
    f.write(png_bytes)