# RAG juridique avec Gradio + Ollama local

Cette version remplace les modèles API/OpenAI et Hugging Face chargés directement par des modèles locaux Ollama.

## 1. Installer Ollama sur Windows

Dans PowerShell:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Fermez puis rouvrez PowerShell, puis testez:

```powershell
ollama --version
```

## 2. Télécharger les 3 modèles locaux

```powershell
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull mistral:7b
```

Si votre PC est lent, commencez seulement par:

```powershell
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

## 3. Installer le projet

```powershell
cd C:\Users\amami\Downloads\week3-RAG-ollama
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Lancer l'interface Gradio

```powershell
python gradio_app.py
```

Ouvrez ensuite:

```text
http://127.0.0.1:7860
```

## Notes importantes

- Le projet n'utilise plus l'API OpenAI.
- Le fichier `ollama_models.py` contient la liste des trois modèles.
- Le fichier `rag_core.py` utilise un prompt strict pour limiter les hallucinations.
- Le seuil de confiance par défaut est `0.20`. Si le score est trop faible, le système refuse de donner une réponse juridique non fiable.
