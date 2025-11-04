"""
evaluate_ragas_sql.py — Évaluation combinée RAG + SQL Tool (Mistral + PostgreSQL)
"""

import sys
import os
import json
import time
import random
import logging
import warnings
import logfire  # ✅ ajout logfire moderne

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_similarity, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.exceptions import MistralAPIStatusException

# --- Imports internes ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K, EMBEDDING_MODEL
from utils.vector_store import VectorStoreManager
from db.sql_tool import NBAQueryTool

# --- Configuration générale ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Initialisation Logfire (connexion automatique à ton projet Pydantic Cloud)
logfire.configure()
logfire.info("🚀 Lancement de l’évaluation RAGAS (RAG + SQL Tool)")

# --- Initialisation des composants ---
with logfire.span("Initialisation des outils"):
    client = MistralClient(api_key=MISTRAL_API_KEY)
    vector_store_manager = VectorStoreManager()
    sql_tool = NBAQueryTool()

# --- Schéma de la base PostgreSQL ---
DB_SCHEMA = """
TABLE teams (...);
TABLE players (...);
TABLE stats (...);
TABLE matches (...);
TABLE player_match_stats (...);
"""

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert de la NBA.
Tu as accès à deux sources :
1️⃣ La base de connaissances textuelles (FAISS)
2️⃣ La base de données SQL chiffrée

Si la question contient des éléments chiffrés (%, points, moyennes, victoires...),
utilise le SQL Tool. Sinon, base-toi sur les documents FAISS.
Réponds de façon concise, fiable et analytique.

{context_str}

Question : {question}
Réponse :
"""

# --- Détection des questions numériques ---
def is_numeric_question(question: str) -> bool:
    keywords = ["%", "points", "victoires", "moyenne", "rebonds", "passes", "stats", "classement", "meilleur", "total"]
    return any(k.lower() in question.lower() for k in keywords)

# --- Attente entre requêtes ---
def wait_before_next_request(min_delay=5, max_delay=10):
    time.sleep(random.uniform(min_delay, max_delay))

# --- Génération de réponse (SQL ou FAISS) ---
def get_answer_and_context(question: str, max_retries=4):
    for attempt in range(1, max_retries + 1):
        try:
            wait_before_next_request(2, 4)

            # Cas 1️⃣ : question chiffrée → SQL Tool
            if is_numeric_question(question):
                logfire.info("🔢 Question chiffrée détectée → utilisation du SQL Tool", extra={"question": question})
                with logfire.span("Génération SQL via Mistral"):
                    sql_query_prompt = [
                        ChatMessage(
                            role="system",
                            content=(
                                "Tu es un assistant SQL PostgreSQL. "
                                "Utilise EXCLUSIVEMENT les tables et colonnes suivantes :\n"
                                f"{DB_SCHEMA}\n"
                                "Donne uniquement la requête SQL correcte, sans explication, sans ```sql."
                            ),
                        ),
                        ChatMessage(role="user", content=question),
                    ]
                    response = client.chat(model=MODEL_NAME, messages=sql_query_prompt, temperature=0)
                    sql_query = response.choices[0].message.content.strip()
                    logfire.info("💾 Requête SQL générée", extra={"sql": sql_query})

                with logfire.span("Exécution SQL"):
                    answer = sql_tool._run(sql_query)

                context_str = f"Résultat SQL : {answer[:800]}"
                return answer, context_str

            # Cas 2️⃣ : question textuelle → RAG (FAISS)
            logfire.info("🔎 Question textuelle → recherche FAISS", extra={"question": question})
            with logfire.span("Recherche FAISS"):
                search_results = vector_store_manager.search(question, k=SEARCH_K)

            context_str = "\n\n---\n\n".join([
                f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
                for res in search_results
            ]) if search_results else "Aucun contexte pertinent trouvé."

            final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=question)
            messages = [ChatMessage(role="user", content=final_prompt)]

            with logfire.span("Génération de réponse FAISS"):
                response = client.chat(model=MODEL_NAME, messages=messages, temperature=0.1)
                answer = response.choices[0].message.content if response.choices else "Réponse vide."

            return answer, context_str

        except MistralAPIStatusException as e:
            if "429" in str(e):
                delay = 5 * attempt
                logfire.warning("⚠️ Trop de requêtes (429). Pause temporaire", extra={"attempt": attempt, "delay": delay})
                time.sleep(delay)
            else:
                logfire.error("Erreur API Mistral", extra={"error": str(e)})
                break
        except Exception as e:
            logfire.error("Erreur pendant la génération", extra={"error": str(e)})
            break

    return "Erreur : aucune réponse obtenue", ""

# --- Chargement du jeu d’évaluation ---
EVAL_FILE = os.path.join("eval", "eval_data.json")
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

questions, answers, contexts, ground_truths = [], [], [], []

for i, item in enumerate(eval_data, 1):
    q = item["question"]
    gt = item["ground_truth"]
    logfire.info(f"🧠 ({i}/{len(eval_data)}) Question : {q}")
    wait_before_next_request(2, 5)

    with logfire.span(f"Traitement question {i}"):
        answer, ctx = get_answer_and_context(q)
        questions.append(q)
        answers.append(answer)
        contexts.append([ctx])
        ground_truths.append(gt)
        logfire.info("✅ Question traitée", extra={"question": q})

# --- Création Dataset RAGAS ---
with logfire.span("Création du dataset RAGAS"):
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

# --- LLM & embeddings Mistral ---
with logfire.span("Initialisation des modèles RAGAS"):
    llm_for_ragas = LangchainLLMWrapper(ChatMistralAI(api_key=MISTRAL_API_KEY, model=MODEL_NAME))
    embeddings_for_ragas = LangchainEmbeddingsWrapper(MistralAIEmbeddings(api_key=MISTRAL_API_KEY, model=EMBEDDING_MODEL))

# --- Évaluation RAGAS ---
with logfire.span("Calcul des métriques RAGAS"):
    logfire.info("📊 Calcul des métriques RAGAS (RAG + SQL)...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_similarity, context_precision, context_recall],
        llm=llm_for_ragas,
        embeddings=embeddings_for_ragas
    )

# --- Sauvegarde ---
RESULTS_PATH = os.path.join("eval", "results_rag_sql.json")
results_data = {
    "metrics": {k: float(v) for k, v in results.items()},
    "details": [
        {"question": q, "answer": a, "ground_truth": gt, "context": c[0]}
        for q, a, gt, c in zip(questions, answers, ground_truths, contexts)
    ]
}
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=4, ensure_ascii=False)

logfire.info("✅ Résultats enregistrés", extra={"file": RESULTS_PATH, **{k: float(v) for k, v in results.items()}})

print("\n===== 📈 RÉSULTATS RAGAS (RAG + SQL) =====")
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")


