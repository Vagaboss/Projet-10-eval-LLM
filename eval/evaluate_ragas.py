"""
evaluate_ragas.py — Évaluation RAGAS utilisant Mistral pour le LLM et les embeddings
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
from langchain_mistralai.embeddings import MistralAIEmbeddings  # ✅ pour remplacer OpenAIEmbeddings
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.exceptions import MistralAPIStatusException

# --- Charger les modules du projet ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K, EMBEDDING_MODEL
from utils.vector_store import VectorStoreManager

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

client = MistralClient(api_key=MISTRAL_API_KEY)
vector_store_manager = VectorStoreManager()

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert de la NBA.
Tu réponds aux questions des analystes en t'appuyant sur les données contextuelles suivantes :

{context_str}

Question : {question}
Réponse :
"""

# --- Gestion des délais et de la charge API ---
def wait_before_next_request(min_delay=3, max_delay=6):
    """Pause aléatoire entre les appels API pour éviter le rate limit."""
    delay = random.uniform(min_delay, max_delay)
    logging.info(f"⏳ Pause de {delay:.1f} secondes avant la prochaine requête...")
    time.sleep(delay)


def get_answer_and_context(question: str, max_retries=5):
    """Récupère le contexte et génère une réponse avec Mistral"""
    for attempt in range(1, max_retries + 1):
        try:
            # Petite pause avant chaque recherche de contexte
            wait_before_next_request(2, 4)

            logging.info(f"Recherche de contexte pour la question : {question}")
            search_results = vector_store_manager.search(question, k=SEARCH_K)

            context_str = "\n\n---\n\n".join([
                f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
                for res in search_results
            ]) if search_results else "Aucun contexte pertinent trouvé."

            final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=question)
            messages = [ChatMessage(role="user", content=final_prompt)]

            # Délai avant l'appel chat Mistral
            wait_before_next_request(3, 6)
            response = client.chat(model=MODEL_NAME, messages=messages, temperature=0.1)
            answer = response.choices[0].message.content if response.choices else "Réponse vide."
            return answer, context_str

        except MistralAPIStatusException as e:
            if "429" in str(e):
                wait = 5 * attempt
                logging.warning(f"⚠️ Limite atteinte. Nouvelle tentative dans {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"Erreur API inattendue : {e}")
                break
        except Exception as e:
            logging.error(f"Erreur pendant la génération : {e}")
            break

    logging.error(f"❌ Impossible d'obtenir la réponse après {max_retries} tentatives.")
    return "", ""

# --- Charger le jeu d’évaluation ---
EVAL_FILE = os.path.join("eval", "eval_data.json")
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

questions, answers, contexts, ground_truths = [], [], [], []

for i, item in enumerate(eval_data, 1):
    q = item["question"]
    gt = item["ground_truth"]
    logging.info(f"\n🧠 ({i}/{len(eval_data)}) Question : {q}")

    # Pause entre chaque question pour éviter 429
    wait_before_next_request(2, 5)

    answer, ctx = get_answer_and_context(q)
    questions.append(q)
    answers.append(answer)
    contexts.append([ctx])
    ground_truths.append(gt)

# --- Créer le Dataset RAGAS ---
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# --- ✅ Utiliser Mistral pour les LLMs et embeddings ---
llm_for_ragas = LangchainLLMWrapper(
    ChatMistralAI(api_key=MISTRAL_API_KEY, model=MODEL_NAME)
)

embeddings_for_ragas = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(api_key=MISTRAL_API_KEY, model=EMBEDDING_MODEL)
)

# --- Évaluation RAGAS ---
logging.info("📊 Calcul des métriques RAGAS avec Mistral...")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_similarity, context_precision, context_recall],
    llm=llm_for_ragas,
    embeddings=embeddings_for_ragas  # ✅ empêche RAGAS d'appeler OpenAI
)

print("\n===== 📈 RÉSULTATS RAGAS (Évaluation via Mistral) =====")
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")

# --- Sauvegarde ---
RESULTS_PATH = os.path.join("eval", "results.json")
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



