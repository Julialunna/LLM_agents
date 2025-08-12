from langchain_community.chat_models import ChatDeepInfra
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")

#model
llm_model = ChatDeepInfra(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", 
    temperature=0,  
    api_token=API_KEY
)

system_message = SystemMessage(content="""
Você é um assistente especializado em fornecer informações
sobre comunidades de Python para GenAI.

Ferramentas disponíveis no MCP Server:

1. get_community(location: str) -> str
- Função: retorna a melhor comunidade de Python para GenAI.
- Parâmetro: location (string)
- Retorno: "Code TI" 

Seu papel é ser um intermediário direto entre o usuários e 
a ferramenta MCP, retornando apenas o resultado final das ferramentas.
"""
)

def agent_mcp():
    client = MultiServerMCPClient(
        {
            "code":{
                "command": "python",
                "args": ["mcp_server.py"],
                "transport": "stdio"
            }
        }
    )
    agent = create_react_agent(llm_model, client.get_tools(), prompt=system_message)
    return agent
