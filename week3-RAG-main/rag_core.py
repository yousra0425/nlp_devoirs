from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).with_name("export_final.csv")
DEFAULT_QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_LLAMA_MODEL = os.getenv("LLAMA_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")



DOMAIN_TERMS = {
    "amende",
    "alcool",
    "article",
    "code",
    "conduire",
    "conduite",
    "conducteur",
    "depassement",
    "domaine",
    "infraction",
    "licence",
    "permis",
    "points",
    "prison",
    "route",
    "routier",
    "sanction",
    "vehicule",
    "vitesse",
    "سياقه",
    "رخصه",
    "طريق",
    "مركبه",
}


STOP_WORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "en",
    "est",
    "et",
    "la",
    "le",
    "les",
    "ma",
    "mon",
    "ou",
    "pour",
    "que",
    "qui",
    "sur",
    "un",
    "une",
}


@dataclass(frozen=True)
class RetrievedDocument:
    article_id: str
    score: float
    source: str
    title: str
    text: str
    keywords: str
    metadata: dict[str, object]


def _clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text.lower(), flags=re.UNICODE)
    return {token for token in tokens if len(token) > 2 and token not in STOP_WORDS}


def _shorten(text: str, limit: int = 520) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_penalties(row: pd.Series) -> list[str]:
    facts: list[str] = []
    if _clean_value(row.get("amende_fixe")):
        facts.append(f"amende fixe: {row['amende_fixe']}")
    if _clean_value(row.get("amende_min")) or _clean_value(row.get("amende_max")):
        facts.append(f"amende min/max: {row.get('amende_min', '')}/{row.get('amende_max', '')}")
    if _clean_value(row.get("points_retrait")):
        facts.append(f"points retires: {row['points_retrait']}")
    if bool(row.get("has_prison")):
        facts.append("peine de prison possible")
    if bool(row.get("has_license_penalty")):
        facts.append("sanction liee au permis possible")
    return facts


def _select_torch_dtype(torch_module):
    return torch_module.float16 if getattr(torch_module.cuda, "is_available", lambda: False)() else torch_module.float32


def _response_error_detail(response: object) -> str:
    try:
        payload = response.json()
    except Exception:
        return _shorten(getattr(response, "text", ""), 220)

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return _shorten(str(error.get("message", "")), 220)
        if error:
            return _shorten(str(error), 220)
    return _shorten(str(payload), 220)


class RAGPipeline:
    """Small, offline RAG pipeline for the Moroccan road-code article CSV."""

    def __init__(self, data_path: str | Path = DATA_PATH, domain_threshold: float = 0.08) -> None:
        self.data_path = Path(data_path)
        self.domain_threshold = domain_threshold
        self.df = self._load_data(self.data_path)
        self.documents = self.df.apply(self._document_text, axis=1).tolist()
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(self.documents)

    @staticmethod
    def _load_data(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Dataset introuvable: {path}")

        df = pd.read_csv(path)
        required = {"article_id", "infraction_desc", "mots_cles", "source"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV: {', '.join(sorted(missing))}")

        df = df.copy()
        df["article_id"] = df["article_id"].astype(str)
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].fillna("").astype(str)
        return df

    @staticmethod
    def _document_text(row: pd.Series) -> str:
        metadata = [
            f"article {row.get('article_id', '')}",
            _clean_value(row.get("mots_cles")),
            _clean_value(row.get("role_paragraphe")),
            _clean_value(row.get("categorie_vehicule")),
            _clean_value(row.get("source")),
            _clean_value(row.get("infraction_desc")),
        ]
        penalties = _format_penalties(row)
        return " | ".join(part for part in metadata + penalties if part)

    def retrieve(self, question: str, k: int = 3) -> list[RetrievedDocument]:
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        top_indices = np.argsort(scores)[::-1][: max(1, min(k, len(scores)))]

        results: list[RetrievedDocument] = []
        for idx in top_indices:
            row = self.df.iloc[int(idx)]
            article_id = _clean_value(row["article_id"])
            results.append(
                RetrievedDocument(
                    article_id=article_id,
                    score=float(scores[idx]),
                    source=_clean_value(row.get("source")),
                    title=f"Article {article_id}",
                    text=_clean_value(row.get("infraction_desc")),
                    keywords=_clean_value(row.get("mots_cles")),
                    metadata={
                        "categorie_vehicule": _clean_value(row.get("categorie_vehicule")),
                        "role_paragraphe": _clean_value(row.get("role_paragraphe")),
                        "amende_fixe": _clean_value(row.get("amende_fixe")),
                        "amende_min": _clean_value(row.get("amende_min")),
                        "amende_max": _clean_value(row.get("amende_max")),
                        "points_retrait": _clean_value(row.get("points_retrait")),
                        "has_prison": bool(row.get("has_prison")),
                        "has_license_penalty": bool(row.get("has_license_penalty")),
                    },
                )
            )
        return results

    def is_out_of_domain(self, question: str, docs: list[RetrievedDocument]) -> tuple[bool, str]:
        tokens = _tokenize(question)
        has_domain_term = bool(tokens.intersection(DOMAIN_TERMS))
        best_score = docs[0].score if docs else 0.0

        if best_score < self.domain_threshold and not has_domain_term:
            return True, (
                "La question semble hors domaine: elle ne correspond pas assez aux articles "
                "du code de la route disponibles."
            )
        return False, "Question consideree dans le domaine du corpus."

    def build_prompt(self, question: str, docs: list[RetrievedDocument]) -> str:
        context_blocks = []
        for doc in docs:
            context_blocks.append(
                f"[{doc.title} | score={doc.score:.3f} | source={doc.source}]\n{doc.text}"
            )
        context = "\n\n".join(context_blocks)
        return (
            "Tu es un assistant juridique RAG. Reponds uniquement avec le contexte fourni. "
            "Si le contexte est insuffisant, dis-le clairement.\n\n"
            f"Question utilisateur:\n{question}\n\n"
            f"Documents pertinents:\n{context}\n\n"
            "Reponse contextualisee avec references:"
        )

    def _unavailable_message(self, provider: str, exc: Exception) -> str:
        return (
            f"{provider} indisponible dans cet environnement: {exc}. "
            "Verifiez l'installation, la cle API ou le serveur local, puis relancez."
        )

    def generate_with_qwen(
        self,
        prompt: str,
        model_name: str = DEFAULT_QWEN_MODEL,
        max_new_tokens: int = 220,
    ) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not hasattr(self, "_qwen_model"):
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=_select_torch_dtype(torch),
                )
                model.eval()
                self._qwen_tokenizer = tokenizer
                self._qwen_model = model

            tokenizer = self._qwen_tokenizer
            model = self._qwen_model
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            input_length = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer_ids = output_ids[0][input_length:]
            return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            return self._unavailable_message("Qwen", exc)

    def generate_with_gpt(
        self,
        prompt: str,
        model_name: str = DEFAULT_GPT_MODEL,
        max_output_tokens: int = 300,
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "GPT indisponible: definissez la variable d'environnement OPENAI_API_KEY."

        try:
            import requests

            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={   # ✅ better than data=json.dumps(...)
                    "model": model_name,
                    "input": prompt,
                    "max_output_tokens": max_output_tokens,
                },
                timeout=90,
            )

            if response.status_code == 429:
                detail = _response_error_detail(response)
                detail_text = f" Detail OpenAI: {detail}" if detail else ""
                return (
                    "GPT indisponible: limite OpenAI atteinte (429 Too Many Requests). "
                    "Attendez un moment, reduisez les appels, ou verifiez le quota/billing "
                    f"du projet OpenAI.{detail_text}"
                )
            if response.status_code in {401, 403}:
                detail = _response_error_detail(response)
                detail_text = f" Detail OpenAI: {detail}" if detail else ""
                return (
                    "GPT indisponible: la cle OPENAI_API_KEY est absente, invalide, "
                    f"ou non autorisee pour ce projet.{detail_text}"
                )

            response.raise_for_status()
            payload = response.json()

            # ✅ easiest extraction
            if payload.get("output_text"):
                return payload["output_text"].strip()

            # fallback parsing
            chunks: list[str] = []
            for item in payload.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"}:
                        chunks.append(content.get("text", ""))

            return "\n".join(chunk for chunk in chunks if chunk).strip() or str(payload)

        except Exception as exc:
            return self._unavailable_message("GPT", exc)

    def generate_with_llama(
        self,
        prompt: str,
        model_name: str = DEFAULT_LLAMA_MODEL,
        max_new_tokens: int = 220,
    ) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            if not hasattr(self, "_llama_generator"):
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=_select_torch_dtype(torch),
                )
                self._llama_generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )

            output = self._llama_generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.2,
                return_full_text=False,
            )
            return output[0]["generated_text"].strip()
        except Exception as exc:
            return self._unavailable_message("Llama/HuggingFace", exc)

    def generate_answer(
        self,
        question: str,
        docs: list[RetrievedDocument],
        model_name: str = "LLM-A extractif",
    ) -> str:
        out_of_domain, reason = self.is_out_of_domain(question, docs)
        if out_of_domain:
            return (
                f"{reason} Je ne peux pas donner une reponse fiable avec ce corpus. "
                "Essayez une question sur le permis, les infractions, les vehicules ou les sanctions routieres."
            )

        if not docs:
            return "Aucun document pertinent n'a ete trouve dans le corpus."

        best = docs[0]
        references = ", ".join(f"Article {doc.article_id}" for doc in docs)

        if model_name == "LLM-B structure":
            lines = [
                "Reponse:",
                f"La question est rapprochee principalement de l'{best.title.lower()} "
                f"(mots-cles: {best.keywords or 'non precises'}).",
                "",
                "Elements du contexte:",
            ]
            for doc in docs:
                lines.append(f"- {doc.title}: {_shorten(doc.text, 260)}")
            lines.extend(["", f"References utilisees: {references}."])
            return "\n".join(lines)

        if model_name == "LLM-C prudent":
            return (
                f"D'apres les articles retrouves, la reponse la plus pertinente se trouve dans "
                f"{best.title}. Le passage indique: {_shorten(best.text, 360)} "
                f"Cette reponse doit etre lue avec prudence car elle depend uniquement des "
                f"{len(docs)} document(s) fournis. References: {references}."
            )

        return (
            f"Le corpus associe votre question a {best.title}. "
            f"Contenu pertinent: {_shorten(best.text, 420)} "
            f"Documents consultes: {references}."
        )

    def compare_llms(self, question: str, k: int = 3, use_real_models: bool = False) -> list[dict[str, object]]:
        docs = self.retrieve(question, k=k)
        rows = []

        if use_real_models:
            prompt = self.build_prompt(question, docs)
            model_calls = [
                (f"Qwen ({DEFAULT_QWEN_MODEL})", lambda: self.generate_with_qwen(prompt)),
                (f"GPT ({DEFAULT_GPT_MODEL})", lambda: self.generate_with_gpt(prompt)),
                (f"Llama/HuggingFace ({DEFAULT_LLAMA_MODEL})", lambda: self.generate_with_llama(prompt)),
            ]
            for model_name, call in model_calls:
                answer = call()
                rows.append(
                    {
                        "modele": model_name,
                        "reponse": answer,
                        "nb_references": len(docs),
                        "meilleur_score": round(docs[0].score if docs else 0.0, 4),
                    }
                )
            return rows

        for model_name in ["LLM-A extractif", "LLM-B structure", "LLM-C prudent"]:
            answer = self.generate_answer(question, docs, model_name=model_name)
            rows.append(
                {
                    "modele": model_name,
                    "reponse": answer,
                    "nb_references": len(docs),
                    "meilleur_score": round(docs[0].score if docs else 0.0, 4),
                }
            )
        return rows

    def evaluate(
        self,
        question: str,
        docs: list[RetrievedDocument],
        expected_article_ids: Iterable[str] | None = None,
    ) -> dict[str, object]:
        retrieved_ids = {doc.article_id for doc in docs if doc.score > 0}

        if expected_article_ids:
            relevant_ids = {str(article_id).strip() for article_id in expected_article_ids if str(article_id).strip()}
            mode = "verite terrain utilisateur"
        else:
            query_tokens = _tokenize(question)
            relevant_ids = set()
            for _, row in self.df.iterrows():
                row_tokens = _tokenize(self._document_text(row))
                if query_tokens.intersection(row_tokens):
                    relevant_ids.add(_clean_value(row["article_id"]))
            mode = "estimation par recouvrement lexical"

        true_positive = len(retrieved_ids.intersection(relevant_ids))
        precision = true_positive / len(retrieved_ids) if retrieved_ids else 0.0
        recall = true_positive / len(relevant_ids) if relevant_ids else 0.0

        return {
            "mode": mode,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "retrieved_ids": sorted(retrieved_ids),
            "relevant_ids": sorted(relevant_ids),
        }

    def answer(
        self,
        question: str,
        k: int = 3,
        expected_ids: Iterable[str] | None = None,
        use_real_models: bool = False,
    ) -> dict[str, object]:
        docs = self.retrieve(question, k=k)
        out_of_domain, reason = self.is_out_of_domain(question, docs)
        return {
            "question": question,
            "documents": docs,
            "prompt": self.build_prompt(question, docs),
            "answer": self.generate_answer(question, docs),
            "comparison": self.compare_llms(question, k=k, use_real_models=use_real_models),
            "evaluation": self.evaluate(question, docs, expected_ids),
            "out_of_domain": out_of_domain,
            "domain_reason": reason,
        }


def parse_expected_ids(raw: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;\s]+", raw) if part.strip()]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    pipeline = RAGPipeline()
    sample_question = "Quelles sont les regles concernant le permis de conduire ?"
    result = pipeline.answer(sample_question)
    print(result["answer"])
    print(result["evaluation"])
