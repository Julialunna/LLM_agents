from news_collab_llm import news_workflow,  NewsAgentState

def main():
    print("=== Multiagente Análise de Notícia ===")
    news = input("Digite o texto da notícia (ou cole o conteúdo):")
    #initial state

    state=  NewsAgentState(
        original_news=news,
        plan=None,
        current_task_idx=0,
        intermediate_results={},
        final_response=None,
        error=None,
        current_task_description=None,
        current_specialist_type=None,
        current_task_id=None,
        specialist_result=None
    )
    result = news_workflow.invoke(state)
    print("\n=== Resposta Final ===\n")
    print(result.get("final_response", "Nenhuma resposta foi gerada "))
    
if __name__ == "__main__":
    main()