from router import workflow_router
def test_workflow():
    resultado_tecnico = workflow_router.invoke({
        "query":"Como posso aprender Python?"
    })
    print("\n===Consulta Técnica===")
    print(f"Pergunta: Como posso aprender Python?")
    print(f"Resposta: {resultado_tecnico['answer']}")

    resultado_saude = workflow_router.invoke({
        "query":"Quais os benefícios de uma alimentação saudável?", 
    })
    print("\n===Consulta de Saúde===")
    print(f"Pergunta: Quais os benefícios de uma alimentação saudável?")
    print(f"Resposta: {resultado_saude['answer']}")
    
    resultado_geral = workflow_router.invoke({
        "query":"Qual é a capital do Brasil?", 
    })
    print("\n===Consulta Geral===")
    print(f"Pergunta: Qual é a capital do Brasil?")
    print(f"Resposta: {resultado_geral['answer']}")

if __name__ == "__main__":
    test_workflow()