from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ollama_models import DEFAULT_MODEL, LOCAL_MODELS, ask_ollama, compare_ollama_models


DATA_PATH = Path(__file__).with_name("export_final.csv")

DOMAIN_TERMS = {
    "amende", "alcool", "article", "code", "conduire", "conduite", "conducteur",
    "depassement", "dépassement", "domaine", "infraction", "licence", "permis",
    "points", "prison", "route", "routier", "sanction", "vehicule", "véhicule", "vitesse",
    "سياقه", "سياقة", "رخصه", "رخصة", "طريق", "مركبه", "مركبة", "السائق", "سايق",
    "مراقبه", "مراقبة", "نقط", "غرامه", "غرامة", "مخالفه", "مخالفة", "وثيقه", "وثيقة",
}

STOP_WORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "en", "est", "et",
    "la", "le", "les", "ma", "mon", "ou", "pour", "que", "qui", "sur", "un", "une",
    "ما", "هي", "هو", "على", "علي", "في", "من", "إلى", "الى", "عن", "و", "او", "أو", "كل", "ذلك",
}

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")

ARABIC_QUERY_EXPANSIONS = {
    "وثائق": "وثيقه الوثيقه رخصه السياقه المراقبه الاعوان الادالء",
    "وثيقة": "وثيقه الوثيقه رخصه السياقه المراقبه الاعوان الادالء",
    "الوثائق": "وثيقه الوثيقه رخصه السياقه المراقبه الاعوان الادالء",
    "السائق": "السايق السايقين صاحب رخصه السياقه",
    "سائق": "سايق السايق صاحب رخصه السياقه",
    "تقديمها": "الادلاء يدلي الادالء",
    "المراقبة": "المراقبه الاعوان مراقبه تطبيق احكام القانون",
    "شروط": "الشروط الحصول المترشح اختبار فحص طبي سن سنه",
    "الحصول": "الحصول المترشح اجتياز اختبار رخصه السياقه",
    "فقدان": "فقدان مجموع النقط تلغي الرخصه خصم النقط",
    "جميع": "مجموع كل النقط",
    "نقط": "النقط رصيد النقط خصم النقط",
    "تسجيل": "التسجيل شهاده تسجيل مركبه",
    "مالك": "مالك المركبه نقل الملكيه تغيير مالكها",
    "مهنية": "مهني سايق منهي بطاقه سايق منهي",
    "مهني": "منهي سايق منهي بطاقه سايق منهي",
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


def _contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _normalize_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text or "")
    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
        "ﻻ": "لا", "ٱ": "ا",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _normalize_for_retrieval(text: str) -> str:
    text = _normalize_arabic(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_arabic_query(question: str) -> str:
    if not _contains_arabic(question):
        return question
    normalized = _normalize_for_retrieval(question)
    extras: list[str] = []
    for key, expansion in ARABIC_QUERY_EXPANSIONS.items():
        if _normalize_arabic(key) in normalized:
            extras.append(expansion)
    return normalized + " " + " ".join(extras)


def _tokenize(text: str) -> set[str]:
    text = _normalize_for_retrieval(text)
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text, flags=re.UNICODE)
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
        facts.append(f"points retirés: {row['points_retrait']}")
    if bool(row.get("has_prison")):
        facts.append("peine de prison possible")
    if bool(row.get("has_license_penalty")):
        facts.append("sanction liée au permis possible")
    return facts


def _arabic_no_context_answer() -> str:
    return "لم أجد مادة كافية الصلة في قاعدة البيانات للإجابة بثقة."


def _french_no_context_answer() -> str:
    return "Je n’ai pas trouvé d’article suffisamment pertinent dans le corpus pour répondre avec fiabilité."


class RAGPipeline:
    """Offline RAG pipeline using TF-IDF retrieval + local Ollama models."""

    def __init__(self, data_path: str | Path = DATA_PATH, domain_threshold: float = 0.12) -> None:
        self.data_path = Path(data_path)
        self.domain_threshold = domain_threshold
        self.df = self._load_data(self.data_path)
        self.documents = self.df.apply(self._document_text, axis=1).tolist()
        self.search_documents = [_normalize_for_retrieval(doc) for doc in self.documents]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(self.search_documents)

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
        expanded_question = _expand_arabic_query(question)
        query_vector = self.vectorizer.transform([_normalize_for_retrieval(expanded_question)])
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
        arabic = _contains_arabic(question)
        tokens = _tokenize(question)
        normalized_domain_terms = {_normalize_for_retrieval(term) for term in DOMAIN_TERMS}
        has_domain_term = bool(tokens.intersection(normalized_domain_terms))
        best_score = docs[0].score if docs else 0.0

        if not docs:
            return True, "لم يتم استرجاع أي وثيقة من قاعدة البيانات." if arabic else "Aucun document n'a été récupéré dans le corpus."

        if best_score < self.domain_threshold:
            if arabic:
                return True, f"درجة التشابه ضعيفة ({best_score:.3f} < {self.domain_threshold:.2f})."
            return True, (
                f"Score de similarité insuffisant ({best_score:.3f} < {self.domain_threshold:.2f}). "
                "Le système ne doit pas générer de réponse juridique fiable avec ce contexte."
            )

        if not has_domain_term and best_score < self.domain_threshold + 0.05:
            if arabic:
                return True, "يبدو أن السؤال خارج نطاق مواد مدونة السير المتوفرة."
            return True, "La question semble hors domaine: elle ne correspond pas assez aux articles du code de la route disponibles."

        if arabic:
            return False, f"السؤال داخل نطاق قاعدة البيانات. أفضل درجة تشابه: {best_score:.3f}."
        return False, f"Question considérée dans le domaine du corpus. Meilleur score: {best_score:.3f}."

    def build_prompt(self, question: str, docs: list[RetrievedDocument]) -> str:
        arabic = _contains_arabic(question)
        context_blocks = []
        for doc in docs:
            penalties = []
            for key in ["amende_fixe", "amende_min", "amende_max", "points_retrait"]:
                value = doc.metadata.get(key)
                if value:
                    penalties.append(f"{key}: {value}")
            extra = " | ".join(penalties)
            context_blocks.append(
                f"[{doc.title} | score={doc.score:.3f} | source={doc.source}]\n"
                f"Mots-clés/كلمات مفتاحية: {doc.keywords}\n"
                f"Sanctions structurées/عقوبات منظمة: {extra or 'غير محددة'}\n"
                f"Texte/النص: {doc.text}"
            )
        context = "\n\n".join(context_blocks)

        if arabic:
            return f"""
أنت مساعد قانوني متخصص في مدونة السير المغربية.

قواعد إلزامية:
1. أجب باللغة العربية فقط.
2. اعتمد فقط على السياق المسترجع من قاعدة البيانات.
3. إذا لم يتضمن السياق الجواب، أجب بالضبط: "لم أجد مادة كافية الصلة في قاعدة البيانات للإجابة بثقة."
4. لا تخترع أي عقوبة أو مبلغ أو رقم مادة.
5. لا تتحدث عن قانون فرنسا أو أي بلد آخر.
6. اذكر المراجع في النهاية بهذه الصيغة: المراجع: المادة X، المادة Y.
7. اجعل الجواب قصيرا وواضحا.

السؤال:
{question}

السياق المسترجع:
{context}

الجواب بالعربية:
""".strip()

        return f"""
Tu es un assistant juridique spécialisé dans le code de la route marocain.

Règles obligatoires:
1. Réponds uniquement à partir du contexte fourni.
2. Si le contexte ne contient pas la réponse, réponds exactement:
   "Je n’ai pas trouvé d’article suffisamment pertinent dans le corpus pour répondre avec fiabilité."
3. N'invente aucune sanction, aucun montant, aucun article.
4. Ne parle pas du droit français, européen ou d'un autre pays.
5. Cite les articles utilisés sous la forme: Références: Article X, Article Y.
6. Réponds en français clair et court.

Question:
{question}

Contexte récupéré:
{context}

Réponse:
""".strip()

    def generate_with_ollama(self, prompt: str, model_name: str = DEFAULT_MODEL) -> str:
        return ask_ollama(model_name, prompt)

    def generate_answer(
        self,
        question: str,
        docs: list[RetrievedDocument],
        use_ollama: bool = True,
        model_name: str = DEFAULT_MODEL,
    ) -> str:
        arabic = _contains_arabic(question)
        out_of_domain, reason = self.is_out_of_domain(question, docs)
        if out_of_domain:
            return f"{reason}\n\n{_arabic_no_context_answer() if arabic else _french_no_context_answer()}"

        if not docs:
            return _arabic_no_context_answer() if arabic else "Aucun document pertinent n'a été trouvé dans le corpus."

        if use_ollama:
            return self.generate_with_ollama(self.build_prompt(question, docs), model_name=model_name)

        best = docs[0]
        if arabic:
            references = "، ".join(f"المادة {doc.article_id}" for doc in docs)
            return f"حسب {best.title}، النص الأقرب هو: {_shorten(best.text, 420)}\n\nالمراجع: {references}."
        references = ", ".join(f"Article {doc.article_id}" for doc in docs)
        return f"Le corpus associe votre question à {best.title}. Contenu pertinent: {_shorten(best.text, 420)} Références: {references}."

    def compare_llms(self, question: str, k: int = 3, use_ollama: bool = True) -> list[dict[str, object]]:
        docs = self.retrieve(question, k=k)
        out_of_domain, reason = self.is_out_of_domain(question, docs)
        best_score = round(docs[0].score if docs else 0.0, 4)
        arabic = _contains_arabic(question)

        if out_of_domain:
            safe_answer = f"{reason} {_arabic_no_context_answer() if arabic else _french_no_context_answer()}"
            return [
                {
                    "modele": f"{display_name} ({ollama_name})",
                    "reponse": safe_answer,
                    "nb_references": len(docs),
                    "meilleur_score": best_score,
                }
                for display_name, ollama_name in LOCAL_MODELS.items()
            ]

        if use_ollama:
            rows = []
            prompt = self.build_prompt(question, docs)
            for row in compare_ollama_models(prompt):
                row["nb_references"] = len(docs)
                row["meilleur_score"] = best_score
                rows.append(row)
            return rows

        answer = self.generate_answer(question, docs, use_ollama=False)
        return [{"modele": "Réponse extractive sans LLM", "reponse": answer, "nb_references": len(docs), "meilleur_score": best_score}]

    def evaluate(
        self,
        question: str,
        docs: list[RetrievedDocument],
        expected_article_ids: Iterable[str] | None = None,
    ) -> dict[str, object]:
        retrieved_ids = {doc.article_id for doc in docs if doc.score > 0}

        if expected_article_ids:
            relevant_ids = {str(article_id).strip() for article_id in expected_article_ids if str(article_id).strip()}
            mode = "vérité terrain utilisateur"
        else:
            query_tokens = _tokenize(_expand_arabic_query(question))
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
        use_ollama: bool = True,
        selected_model: str = DEFAULT_MODEL,
    ) -> dict[str, object]:
        docs = self.retrieve(question, k=k)
        out_of_domain, reason = self.is_out_of_domain(question, docs)
        comparison = self.compare_llms(question, k=k, use_ollama=use_ollama)
        return {
            "question": question,
            "documents": docs,
            "prompt": self.build_prompt(question, docs),
            "answer": self.generate_answer(question, docs, use_ollama=use_ollama, model_name=selected_model),
            "comparison": comparison,
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
    sample_question = "ما هي الوثائق التي يجب على السائق تقديمها عند المراقبة الطرقية؟"
    result = pipeline.answer(sample_question, use_ollama=False)
    print(result["answer"])
    print(result["evaluation"])
