from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_community.chat_models import ChatDeepInfra
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

#model
model = ChatDeepInfra(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", 
    temperature=0,  
    api_token=API_KEY,
    streaming=True 
)

#system prompt
system_message = SystemMessage(content = """
Você é um pesquisador muito sarcástico e irônico.
Use a ferramenta 'search' sempre que necessário, especialmente
para perguntas que exigem informações da web.
""")

#creating tool search
@tool("search")
def search_web(query:str= "")->str:
    """
    Busca iformações na web baseada na consulta fornecida

    Args:
        quert: Termos para buscar dados na web

    Returns:
        As informações encontradas na web ou uma mensagem indicando que nenhuma informação foi encontrada.
    """
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(query)
    return search_docs

#creating react agent
tools = [search_web]
graph = create_react_agent(
    model, 
    tools=tools,
    prompt = system_message, 
)
export_graph = graph
