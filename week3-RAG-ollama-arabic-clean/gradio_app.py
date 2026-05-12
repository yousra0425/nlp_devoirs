from __future__ import annotations

import pandas as pd
import gradio as gr

from ollama_models import DEFAULT_MODEL, LOCAL_MODELS, check_ollama_server
from rag_core import RAGPipeline, parse_expected_ids


# Chargement unique du pipeline RAG
pipeline = RAGPipeline()


def references_table(documents) -> pd.DataFrame:
    """Convertit les documents récupérés en tableau lisible dans Gradio."""
    return pd.DataFrame(
        [
            {
                "article_id": doc.article_id,
                "score": round(doc.score, 4),
                "mots_cles": doc.keywords,
                "source": doc.source,
                "extrait": doc.text[:280] + ("..." if len(doc.text) > 280 else ""),
            }
            for doc in documents
        ]
    )


def comparison_table(comparison_rows) -> pd.DataFrame:
    """Tableau de comparaison des 3 modèles Ollama."""
    return pd.DataFrame(comparison_rows)


def format_references(documents) -> str:
    """Affichage détaillé des références utilisées."""
    if not documents:
        return "Aucune référence trouvée."

    blocks = []
    for doc in documents:
        blocks.append(
            f"### {doc.title} — score {doc.score:.3f}\n"
            f"**Source :** {doc.source or 'non précisée'}  \n"
            f"**Mots-clés :** {doc.keywords or 'non précisés'}\n\n"
            f"> {doc.text}\n"
        )
    return "\n---\n".join(blocks)


def run_rag(
    question: str,
    top_k: int,
    threshold: float,
    expected_raw: str,
    use_ollama: bool,
    selected_model_display: str,
    show_prompt: bool,
):
    """Fonction appelée par l'interface Gradio."""
    question = (question or "").strip()
    if not question:
        empty_df = pd.DataFrame()
        return (
            "Veuillez saisir une question.",
            "",
            empty_df,
            empty_df,
            "",
            "",
            "",
            "",
        )

    pipeline.domain_threshold = float(threshold)
    expected_ids = parse_expected_ids(expected_raw or "")
    selected_model = LOCAL_MODELS.get(selected_model_display, DEFAULT_MODEL)

    result = pipeline.answer(
        question,
        k=int(top_k),
        expected_ids=expected_ids,
        use_ollama=bool(use_ollama),
        selected_model=selected_model,
    )

    documents = result["documents"]
    evaluation = result["evaluation"]

    status = "🔴 Hors domaine / confiance faible" if result["out_of_domain"] else "🟢 Dans le domaine"
    domain_message = f"## {status}\n\n{result['domain_reason']}"

    metrics = (
        f"### Évaluation\n"
        f"- **Precision :** {evaluation['precision']}\n"
        f"- **Recall :** {evaluation['recall']}\n"
        f"- **Mode :** {evaluation['mode']}\n"
        f"- **Articles retrouvés :** {', '.join(evaluation['retrieved_ids']) or 'aucun'}\n"
        f"- **Articles pertinents :** {', '.join(evaluation['relevant_ids']) or 'aucun'}"
    )

    prompt_output = result["prompt"] if show_prompt else "Prompt masqué. Cochez l’option pour l’afficher."

    ok, ollama_status = check_ollama_server()
    corpus_message = f"Corpus chargé : {len(pipeline.df)} articles.\n\nStatut Ollama : {ollama_status}"

    return (
        domain_message,
        result["answer"],
        references_table(documents),
        comparison_table(result["comparison"]),
        metrics,
        format_references(documents),
        prompt_output,
        corpus_message,
    )


with gr.Blocks(title="RAG juridique - Code de la route marocain", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Assistant RAG juridique — Code de la route marocain

        Cette version utilise une interface **Gradio** et des modèles **Ollama locaux**, sans API OpenAI.
        Elle accepte les questions en **arabe** et répond automatiquement en **arabe** si la question est en arabe.
        Elle permet aussi de récupérer les articles pertinents, comparer 3 modèles locaux et afficher les références.
        """
    )

    ok, ollama_status = check_ollama_server()
    gr.Markdown(
        f"""
        **Modèles à installer sur Windows PowerShell :**
        ```powershell
        ollama pull qwen2.5:3b
        ollama pull llama3.2:3b
        ollama pull mistral:7b
        ```

        **Statut Ollama :** {ollama_status}
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Question utilisateur / سؤال المستخدم",
                value="ما هي الوثائق التي يجب على السائق تقديمها عند المراقبة الطرقية؟",
                lines=4,
                placeholder="مثال: ما هي شروط الحصول على رخصة السياقة؟",
            )

            run_button = gr.Button("Interroger le système", variant="primary")

        with gr.Column(scale=1):
            top_k = gr.Slider(
                label="Nombre de documents récupérés",
                minimum=1,
                maximum=max(1, len(pipeline.df)),
                value=min(3, len(pipeline.df)),
                step=1,
            )
            threshold = gr.Slider(
                label="Seuil de confiance minimal",
                minimum=0.0,
                maximum=0.5,
                value=0.12,
                step=0.01,
            )
            expected_raw = gr.Textbox(
                label="Articles attendus pour l’évaluation",
                placeholder="Exemple : 1, 2, 15",
            )
            use_ollama = gr.Checkbox(
                label="Utiliser les modèles Ollama locaux",
                value=True,
            )
            selected_model_display = gr.Dropdown(
                label="Modèle principal pour la réponse",
                choices=list(LOCAL_MODELS.keys()),
                value="Qwen 2.5 3B",
            )
            show_prompt = gr.Checkbox(
                label="Afficher le prompt injecté",
                value=False,
            )
            corpus_info = gr.Markdown(f"Corpus chargé : {len(pipeline.df)} articles.")

    with gr.Tab("Réponse"):
        domain_output = gr.Markdown()
        answer_output = gr.Textbox(label="Réponse contextualisée / الجواب", lines=8)

    with gr.Tab("Documents pertinents"):
        refs_df = gr.Dataframe(label="Documents récupérés", wrap=True)
        refs_details = gr.Markdown(label="Références détaillées")

    with gr.Tab("Comparaison des LLMs"):
        comparison_df = gr.Dataframe(label="Comparaison de 3 modèles Ollama", wrap=True)

    with gr.Tab("Évaluation"):
        metrics_output = gr.Markdown()

    with gr.Tab("Prompt RAG"):
        prompt_output = gr.Textbox(label="Prompt injecté", lines=15)

    run_button.click(
        fn=run_rag,
        inputs=[question, top_k, threshold, expected_raw, use_ollama, selected_model_display, show_prompt],
        outputs=[
            domain_output,
            answer_output,
            refs_df,
            comparison_df,
            metrics_output,
            refs_details,
            prompt_output,
            corpus_info,
        ],
    )


if __name__ == "__main__":
    demo.launch()
