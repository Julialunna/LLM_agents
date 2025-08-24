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

def researcher(state: ReportState):
    #new function Send!
    #it sends a task for another node immediately, it has the next node and its data 
    return [Send("search_tavily", query) for query in state.queries]

def final_writer(state:ReportState):
    search_results = ""
    references = ""
    
    for i, result in enumerate(state.queries_results):
        search_results+= f"[{i+1}]\n\n"
        search_results += f"Title: {result.title}\n"
        search_results += f"URL: {result.url}\n"
        search_results += f"Content: {result.resume}\n"
        search_results += f"=========================\n\n"
        
        references += f"[{i+1}] - [{result.title}]({result.url})\n"
        
        
        prompt = building_final_response.format(user_input = user_input, search_results = search_results)
        llm_result = reasoning_llm.invoke(prompt)
        final_response= llm_result.content + "\n\nReferences: \n" + references
        return {"final_response": final_response}

#graph
builder = StateGraph(ReportState)
builder.add_node("build_first_queries", build_first_queries)
builder.add_node("search_tavily", search_tavily)
builder.add_node("final_writer", final_writer)

builder.add_edge(START, "build_first_queries")
builder.add_conditional_edges("build_first_queries",
                              researcher,
                              ["search_tavily"])
builder.add_edge("search_tavily", "final_writer")
builder.add_edge("final_writer", END)

graph = builder.compile()

#execution
if __name__ == "__main__":
    user_input = """
    Quero que você me explique o processo total para construir um agente de IA
    """
    graph.invoke({"user_input": user_input})