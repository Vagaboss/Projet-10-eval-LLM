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

# --- Config générale ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

client = MistralClient(api_key=MISTRAL_API_KEY)
vector_store_manager = VectorStoreManager()
sql_tool = NBAQueryTool()

# --- Schéma de la base PostgreSQL ---
DB_SCHEMA = """
La base PostgreSQL contient les tables suivantes :

TABLE teams (
    team_id SERIAL PRIMARY KEY,
    team_code VARCHAR(10),
    team_name VARCHAR(100),
    wins INTEGER,
    losses INTEGER,
    offensive_rating REAL,
    defensive_rating REAL,
    net_rating REAL
);

TABLE players (
    player_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    team_code VARCHAR(10),
    age INTEGER,
    position VARCHAR(20)
);

TABLE stats (
    stat_id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(player_id),
    games_played INTEGER,
    points REAL,
    rebounds REAL,
    assists REAL,
    steals REAL,
    blocks REAL,
    fg_pct REAL,
    three_pct REAL,
    ft_pct REAL,
    net_rating REAL
);

TABLE matches (
    match_id SERIAL PRIMARY KEY,
    team_home VARCHAR(10),
    team_away VARCHAR(10),
    match_date DATE,
    home_score INTEGER,
    away_score INTEGER
);

TABLE player_match_stats (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(match_id),
    player_id INTEGER REFERENCES players(player_id),
    points REAL,
    rebounds REAL,
    assists REAL,
    steals REAL,
    blocks REAL
);
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
                logging.info("🔢 Question chiffrée détectée → utilisation du SQL Tool")

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
                logging.info(f"💾 Requête SQL générée : {sql_query}")

                answer = sql_tool._run(sql_query)
                time.sleep(random.uniform(3, 6))
                context_str = f"Résultat SQL : {answer[:800]}"
                return answer, context_str

            # Cas 2️⃣ : question textuelle → RAG (FAISS)
            logging.info("🔎 Question textuelle → recherche FAISS")
            search_results = vector_store_manager.search(question, k=SEARCH_K)

            context_str = "\n\n---\n\n".join([
                f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
                for res in search_results
            ]) if search_results else "Aucun contexte pertinent trouvé."

            final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=question)
            messages = [ChatMessage(role="user", content=final_prompt)]

            wait_before_next_request(3, 7)
            response = client.chat(model=MODEL_NAME, messages=messages, temperature=0.1)
            answer = response.choices[0].message.content if response.choices else "Réponse vide."

            return answer, context_str

        except MistralAPIStatusException as e:
            if "429" in str(e):
                delay = 5 * attempt
                logging.warning(f"⚠️ Trop de requêtes (429). Pause {delay}s...")
                time.sleep(delay)
            else:
                logging.error(f"Erreur API : {e}")
                break
        except Exception as e:
            logging.error(f"Erreur pendant la génération : {e}")
            break

    return "Erreur : aucune réponse obtenue", ""

# --- Charger le jeu d’évaluation ---
EVAL_FILE = os.path.join("eval", "eval_data.json")
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

questions, answers, contexts, ground_truths = [], [], [], []

for i, item in enumerate(eval_data, 1):
    q = item["question"]
    gt = item["ground_truth"]
    logging.info(f"\n🧠 ({i}/{len(eval_data)}) Question : {q}")
    wait_before_next_request(2, 5)

    answer, ctx = get_answer_and_context(q)
    questions.append(q)
    answers.append(answer)
    contexts.append([ctx])
    ground_truths.append(gt)

# --- Création Dataset RAGAS ---
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# --- LLM & embeddings Mistral ---
llm_for_ragas = LangchainLLMWrapper(ChatMistralAI(api_key=MISTRAL_API_KEY, model=MODEL_NAME))
embeddings_for_ragas = LangchainEmbeddingsWrapper(MistralAIEmbeddings(api_key=MISTRAL_API_KEY, model=EMBEDDING_MODEL))

# --- Évaluation RAGAS ---
logging.info("📊 Calcul des métriques RAGAS (RAG + SQL)...")
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

logging.info(f"✅ Résultats enregistrés dans {RESULTS_PATH}")

print("\n===== 📈 RÉSULTATS RAGAS (RAG + SQL) =====")
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")

