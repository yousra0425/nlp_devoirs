from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

# Three local Ollama models used for the assignment comparison.
# Install them on Windows PowerShell with:
#   ollama pull qwen2.5:3b
#   ollama pull llama3.2:3b
#   ollama pull mistral:7b
LOCAL_MODELS: dict[str, str] = {
    "Qwen 2.5 3B": "qwen2.5:3b",
    "Llama 3.2 3B": "llama3.2:3b",
    "Mistral 7B": "mistral:7b",
}

DEFAULT_MODEL_DISPLAY_NAME = "Qwen 2.5 3B"
DEFAULT_MODEL = LOCAL_MODELS[DEFAULT_MODEL_DISPLAY_NAME]


def check_ollama_server() -> tuple[bool, str]:
    """Return a small status message to help the Gradio UI diagnose setup issues."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
        if not models:
            return False, "Ollama fonctionne, mais aucun modèle n'est installé. Lancez: ollama pull qwen2.5:3b"
        return True, "Modèles Ollama détectés: " + ", ".join(models)
    except Exception as exc:
        return False, (
            "Ollama n'est pas joignable sur http://localhost:11434. "
            "Ouvrez Ollama puis relancez l'application. Détail: " + str(exc)
        )


def ask_ollama(model_name: str, prompt: str, max_tokens: int = 350) -> str:
    """Generate a response with a local Ollama model."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return answer or "Le modèle Ollama n'a pas généré de réponse."
    except requests.exceptions.ConnectionError:
        return (
            f"Erreur Ollama avec {model_name}: le serveur Ollama n'est pas lancé. "
            "Sur Windows, ouvrez Ollama ou exécutez `ollama serve`, puis relancez."
        )
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("error", "")
        except Exception:
            detail = response.text[:300]
        return (
            f"Erreur Ollama avec {model_name}: {exc}. {detail}\n"
            f"Vérifiez que le modèle est installé: ollama pull {model_name}"
        )
    except Exception as exc:
        return f"Erreur Ollama avec {model_name}: {exc}"


def compare_ollama_models(prompt: str) -> list[dict[str, str]]:
    """Run the same RAG prompt on the three local models."""
    results: list[dict[str, str]] = []
    for display_name, ollama_name in LOCAL_MODELS.items():
        results.append(
            {
                "modele": f"{display_name} ({ollama_name})",
                "reponse": ask_ollama(ollama_name, prompt),
            }
        )
    return results
