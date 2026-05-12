import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

LOCAL_MODELS = {
    "Qwen 2.5 3B": "qwen2.5:3b",
    "Llama 3.2 3B": "llama3.2:3b",
    "Mistral 7B": "mistral:7b",
}


def ask_ollama(model_name: str, prompt: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 350,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Erreur avec le modèle {model_name}: {e}"


def compare_ollama_models(prompt: str) -> list[dict]:
    results = []

    for display_name, ollama_name in LOCAL_MODELS.items():
        answer = ask_ollama(ollama_name, prompt)
        results.append({
            "modele": display_name,
            "reponse": answer
        })

    return results