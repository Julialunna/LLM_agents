from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from models import models

#System Messages
SYSTEMA_MESSAGEM_LLMS = SystemMessage(content = """
Você é um especialista em análise de código e boas práticas de programação.
Sua tarefa é analisar o código fornecido e sugerir melhorias em termos de:
1. Performance e otimização
2. Boas práticas e padrão de códigos 
3. Segurança e tratamento de eros 
4. Legibilidade e confiabilidade
                                      
Forneça suas sugestões de forma estruturada e clara,
com exemplos práticos de como implementar as melhorias sugeridas.
Seja específico e detalhado em suas recomendações                                       
                                      
""")

#defining state
class State(TypedDict):
    query: str
    llm1: str #análise gemin
    llm2: str #análise llama 3
    best_llm: str #melhor análise escolhida

#nodes
def call_llm_1(state: State):
    """Recebe código e retorna análise do modelo Gemini"""
    messages = [
        SystemMessage(content = SYSTEMA_MESSAGEM_LLMS.content), 
        HumanMessage(content = f"Analise o seguinte código e forneça sugestões de melhorias:\n\n{state['query']}")
    ]
    response = models['gemini_flash'].invoke(messages)
    return {"llm1": response.content}

def call_llm_2(state: State):
    """Recebe código e retorna análise do modelo meta llama 3"""
    messages = [
        SystemMessage(content = SYSTEMA_MESSAGEM_LLMS.content), 
        HumanMessage(content = f"Analise o seguinte código e forneça sugestões de melhorias:\n\n{state['query']}")
    ]
    response = models['meta_llama_3'].invoke(messages)
    return {"llm2": response.content}

def judge(state: State):
    """Avalia qual análise foi mais completa e útil"""
    msg = f"""
    Aja como revisor técnico sênior e avalie a quantidade das análises 
    de código fornecida por dois especialistas.

    Sua tarefa é escolher a análise que:
    1. Identifica mais problemas potenciais
    2. Fornece sugestõe smais práticas e implementáveis.
    3. Considera aspectos do código, como performance, segurança, legibilidade, etc.
    4. Explica melhor o raciocínio por trás das sugestões

    [Código Analisado]
    {state['query']}
    [Análise do Especialista A]
    {state['llm1']}
    [Análise do Especialista B]
    {state['llm2']}

    Forneça sua avaliação comparativa e conclua com seu veredito final usando exatamente um desses formatos:
    '[[A]] se a análise A for melhor'
    '[[B]] se a análise B for melhor'
    '[[C]] em caso de empate'
"""
    
    messages = [SystemMessage(content = msg)]
    response = models["meta_llama_4"].invoke(messages)
    return {"best_llm": response.content}

#building workflow
code_analysis_builder = StateGraph(State)

#adding nodes
code_analysis_builder.add_node("call_llm_1", call_llm_1)
code_analysis_builder.add_node("call_llm_2", call_llm_2)
code_analysis_builder.add_node("judge", judge)

#adding edges
code_analysis_builder.add_edge(START, "call_llm_1")
code_analysis_builder.add_edge(START, "call_llm_2")
code_analysis_builder.add_edge("call_llm_1", "judge")
code_analysis_builder.add_edge("call_llm_2", "judge")
code_analysis_builder.add_edge("judge", END)

code_analysis_workflow =  code_analysis_builder.compile()



