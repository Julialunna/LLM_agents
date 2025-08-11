from chain import detective_workflow

def test_detective_history():
    result = detective_workflow.invoke({
    "detective": "",
    "crime": "",
    "location": "",
    "clue": "",
    "story": {}
})
    print("\n=== HISTÓRIA DO DETETIVE ===\n")
    print("\n=== PRIMEIRO ATO ===\n")
    print(result["story"]["act_1"])
    print("\n=== SEGUNDO ATO ===\n")
    print(result["story"]["act_2"])
    print("\n=== TERCEIRO ATO ===\n")
    print(result["story"]["act_3"])
    print("\n=== QUARTO ATO ===\n")
    print(result["story"]["act_4"])
    
    print("\n=== ELEMENTOS DA HISTÓRIA ===\n")
    print(f"Detective: {result['detective']}")
    print(f"Crime: {result['crime']}")
    print(f"Local: {result['location']}")
    print(f"Pista Incial: {result['clue']}")

if __name__ == "__main__":
    test_detective_history()