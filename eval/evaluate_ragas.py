"""
evaluate_ragas.py
-----------------
Script d'évaluation automatique du chatbot RAG (Mistral) à l'aide des métriques RAGAS.
Ce script :
  1. Charge le fichier eval_data.json (questions + ground_truths)
  2. Utilise le pipeline RAG local (VectorStoreManager + MistralClient)
  3. Génère automatiquement les réponses et les contextes
  4. Calcule les métriques RAGAS pour évaluer la qualité du RAG
"""

import os
import json
import logging
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall

# --- Importation de ton pipeline RAG ---
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialiser les composants
client = MistralClient(api_key=MISTRAL_API_KEY)
vector_store_manager = VectorStoreManager()

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert de la NBA.
Tu réponds aux questions des analystes en t'appuyant sur les données contextuelles suivantes :

{context_str}

Question : {question}
Réponse :"""

# --- Fonction utilitaire : génération de réponse et récupération du contexte ---
def get_answer_and_context(question: str):
    """
    Pour une question donnée :
      1. Recherche le contexte pertinent dans le VectorStore
      2. Envoie le prompt complet au modèle Mistral
      3. Retourne la réponse et le contexte textuel
    """
    try:
        # --- Étape 1 : Récupérer le contexte depuis le vector store ---
        logging.info(f"Recherche de contexte pour la question : {question}")
        search_results = vector_store_manager.search(question, k=SEARCH_K)

        # Mise en forme du contexte pour le prompt
        context_str = "\n\n---\n\n".join([
            f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
            for res in search_results
        ]) if search_results else "Aucun contexte pertinent trouvé."

        # --- Étape 2 : Générer la réponse via Mistral ---
        final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=question)
        prompt_messages = [ChatMessage(role="user", content=final_prompt)]

        # 🔹 Petite pause pour éviter les erreurs 429 Too Many Requests
        time.sleep(1.5)

        response = client.chat(model=MODEL_NAME, messages=prompt_messages, temperature=0.1)
        answer = response.choices[0].message.content if response.choices else "Réponse vide."

        # On renvoie la réponse et le contexte sous forme texte simple
        return answer, context_str

    except Exception as e:
        logging.error(f"Erreur pendant la génération pour '{question}': {e}")
        return "", ""

# --- Étape 1 : Charger le jeu de test ---
with open("eval/eval_data.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

questions = []
answers = []
contexts = []
ground_truths = []

# --- Étape 2 : Boucle d'évaluation ---
for i, item in enumerate(eval_data, 1):
    question = item["question"]
    ground_truth = item["ground_truth"]

    logging.info(f"\n🧠 ({i}/{len(eval_data)}) Question : {question}")
    answer, context_str = get_answer_and_context(question)

    questions.append(question)
    answers.append(answer)
    contexts.append(context_str)
    ground_truths.append(ground_truth)

    # Pause légère pour éviter d'épuiser l'API
    time.sleep(1.5)

# --- Étape 3 : Construire un dataset compatible RAGAS ---
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# --- Étape 4 : Calculer les métriques RAGAS ---
logging.info("📊 Calcul des métriques RAGAS...")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevance, context_precision, context_recall]
)

# --- Étape 5 : Afficher les résultats ---
print("\n===== 📈 RÉSULTATS RAGAS =====")
for k, v in results.items():
    print(f"{k}: {v:.3f}")
