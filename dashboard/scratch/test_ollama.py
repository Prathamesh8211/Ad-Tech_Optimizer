import ollama

print("Testing llama3.2:1b call...")
try:
    res = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": "hi"}])
    print("Success with llama3.2:1b:", res)
except Exception as e:
    print("Error with llama3.2:1b:", type(e), e)

print("\nTesting minimax-m3:cloud call...")
try:
    res = ollama.chat(model="minimax-m3:cloud", messages=[{"role": "user", "content": "hi"}])
    print("Success with minimax-m3:cloud:", res)
except Exception as e:
    print("Error with minimax-m3:cloud:", type(e), e)
