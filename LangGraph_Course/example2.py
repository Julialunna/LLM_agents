from langchain_community.chat_models import ChatDeepInfra
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import tool
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

#defining prompt
system_messsage = SystemMessage(content = """
Você é um assistente que só calcula somas usando a ferramenta abaixo.  
Ferramentas disponíveis:
- somar(valores: str): soma dois números separados por vírgula e retorna o resultado.
Quando o usuário pedir uma soma, faça **sempre**:
  Action: somar
  Action Input: <n1>,<n2>
Depois use a resposta da ferramenta para formular a resposta final.
""")

#defining tool
@tool("somar")
def somar(valores: str)->str:
    """Soma dois números separados por vírgula"""
    try:
        a, b = map(float, valores.split(","))
        return str(a+b)
    except Exception as e:
        return f"Erro ao somar: {str(e)}"

#creating agent with langgraph
tools = [somar]
graph = create_react_agent(
    model = llm_model, 
    tools = tools,
    prompt = system_messsage
)

export_graph = graph

png_bytes = export_graph.get_graph().draw_mermaid_png(
    draw_method= MermaidDrawMethod.API
)
with open("grafo_exemplo2.png", "wb") as f:
    f.write(png_bytes)

def extrair_resposta_final(result):
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    if ai_messages:
        return ai_messages[-1].content
    else:
        return "Nenhuma resposta final foi encontrada"
    
#testing agent
if __name__ == "__main__":
    entrada1 = HumanMessage(content= "Quanto é 8 + 5?")
    result1 = export_graph.invoke({"messages":[entrada1]})
    #to verify if we are using the tool made
    for m in result1["messages"]:
        print(m)
    resposta_texto_1 = extrair_resposta_final(result1)
    print("Resposta 1: ", resposta_texto_1)
    print()

    entrada2 = HumanMessage(content= "Quem pintou a Monalisa?")
    result2 = export_graph.invoke({"messages":[entrada2]})
    #to verify if we are using the tool made
    for m in result2["messages"]:
        print(m)
    resposta_texto_2 = extrair_resposta_final(result2)
    print("Resposta 2: ", resposta_texto_2)
