from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_core import RAGPipeline, parse_expected_ids


st.set_page_config(
    page_title="Pipeline RAG - Code de la route",
    page_icon="",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


def references_table(documents) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": doc.article_id,
                "score": round(doc.score, 4),
                "mots_cles": doc.keywords,
                "source": doc.source,
                "extrait": doc.text[:260] + ("..." if len(doc.text) > 260 else ""),
            }
            for doc in documents
        ]
    )


pipeline = load_pipeline()

st.title("Pipeline RAG - Code de la route marocain")

with st.sidebar:
    st.header("Parametres")
    top_k = st.slider("Nombre de documents", min_value=1, max_value=len(pipeline.df), value=min(3, len(pipeline.df)))
    threshold = st.slider("Seuil hors domaine", min_value=0.0, max_value=0.5, value=0.08, step=0.01)
    expected_raw = st.text_input("Articles attendus pour evaluation", placeholder="ex: 1, 2")
    use_real_models = st.toggle("Comparer Qwen, GPT et Llama reels", value=False)
    show_prompt = st.checkbox("Afficher le prompt injecte", value=False)
    st.caption(f"Corpus charge: {len(pipeline.df)} articles")
    if use_real_models:
        st.caption("Qwen et Llama utilisent des modeles Hugging Face locaux, GPT utilise OPENAI_API_KEY.")

pipeline.domain_threshold = threshold

question = st.text_area(
    "Question utilisateur",
    value="Quelles sont les regles concernant le permis de conduire ?",
    height=100,
)

run = st.button("Interroger le systeme", type="primary")

if run and question.strip():
    expected_ids = parse_expected_ids(expected_raw)
    with st.spinner("Generation de la reponse et comparaison des modeles..."):
        result = pipeline.answer(
            question.strip(),
            k=top_k,
            expected_ids=expected_ids,
            use_real_models=use_real_models,
        )
    documents = result["documents"]
    evaluation = result["evaluation"]

    status = "Hors domaine" if result["out_of_domain"] else "Dans le domaine"
    st.subheader(status)
    st.write(result["domain_reason"])

    st.subheader("Reponse contextualisee")
    st.write(result["answer"])

    st.subheader("Documents pertinents")
    st.dataframe(references_table(documents), width="stretch", hide_index=True)

    st.subheader("Comparaison de 3 LLMs")
    comparison_df = pd.DataFrame(result["comparison"])
    st.dataframe(comparison_df, width="stretch", hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", evaluation["precision"])
    col2.metric("Recall", evaluation["recall"])
    col3.metric("Mode evaluation", evaluation["mode"])

    with st.expander("Details evaluation"):
        st.write("Articles retrouves:", ", ".join(evaluation["retrieved_ids"]) or "aucun")
        st.write("Articles pertinents:", ", ".join(evaluation["relevant_ids"]) or "aucun")

    if show_prompt:
        with st.expander("Prompt injecte dans le generateur", expanded=True):
            st.code(result["prompt"], language="text")

    st.subheader("References aux articles utilises")
    for doc in documents:
        with st.expander(f"{doc.title} - score {doc.score:.3f}"):
            st.write(doc.text)
            st.caption(f"Source: {doc.source} | Mots-cles: {doc.keywords or 'non precises'}")
else:
    st.info("Saisissez une question puis lancez le pipeline.")
