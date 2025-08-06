from parallelization import code_analysis_workflow

#example code 
codigo_test = """
def calcular_media(lista):
    soma=0
    for i in range(len(lista)):
        soma = soma + lista[i]
    media = soma / len(lista)
    return media

#testando a função
numeros = [1, 2, 3, 4, 5]
resultado = calcular_media(numeros)
print(f' A média é: {resultado}')
"""

#executing workflow

resultado = code_analysis_workflow.invoke({
    "query": codigo_test
})
print("\n=== Análise do Gemini ===")
print(resultado["llm1"])
print("\n=== Análise do Llama 4 ===")
print(resultado["llm2"])
print("\n=== Avaliação Final ===")
print(resultado["best_llm"])