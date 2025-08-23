from pydantic import BaseModel
from langchain_community.chat_models import ChatDeepInfra
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from schemas import *
from prompts import *
from dotenv import load_dotenv
import os
from tavily import TavilyClient

load_dotenv()
API_KEY= os.getenv("DEEPINFRA_API_KEY")

#models
llm = ChatDeepInfra(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
    temperature=0,  
    api_token=API_KEY
)

reasoning_llm = ChatDeepInfra(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", 
    temperature=0,  
    api_token=API_KEY
)

#nodes
#this node generates queries
def build_first_queries(state: ReportState):
    class QueryList(BaseModel):
        queries: List[str]
        
    user_input = state.user_input
    #.format() is needed here because agent_promtp requires a variable
    prompt = build_queries.format(user_input = user_input)
    #structuring output to list of strings
    query_llm = llm.with_structured_output(QueryList)
    result = query_llm.invoke(prompt)
    return{"queries": result.queries}

def search_tavily(query:str):
    tavily_client = TavilyClient()
    results = tavily_client.search(query, max_results=1, include_raw_content=False)
    url = results["results"][0]["url"]
    #extracting information from url 
    url_extraction = tavily_client.extract(url)
    if(len(url_extraction["results"])> 0):
        raw_context = url_extraction["results"][0]["raw_content"]
        prompt = resume_search.format(user_input=user_input, search_results = raw_context)
        llm_result = llm.invoke(prompt)
        query_results = QueryResult(
            title=results["results"][0]["title"],
            url =url,
            resume = llm_result.content,
        )
    return{
        "queries_results": [query_results]
    }



#edges

#graph
builder = StateGraph(ReportState)
graph = builder.compile()

#execution
if __name__ == "__main__":
    user_input = """
    Quero que você me explique o processo total para construir um agente de IA
    """
    graph.invoke({"user_input": user_input})