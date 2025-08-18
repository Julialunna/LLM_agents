from mathcollab2 import simple_workflow, SimplesAgentState

def main():
    print("=== Multiagente Matemático ===")
    query = input("Digite uma expressão matemática (ex: 2 + 2 * 3):")
    #initial state
    
    state= SimplesAgentState(
        original_query=query, 
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
    result = simple_workflow.invoke(state)
    print("\n=== Resposta Final ===\n")
    print(result.get("final_response", "Nenhuma resposta foi gerada "))
    
if __name__ == "__main__":
    main()