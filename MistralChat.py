"""
MistralChat.py — Agent RAG + SQL pour analyse NBA
"""

import streamlit as st
import os
import logging
import time
import logfire
import re
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# --- Importations internes ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
    from db.sql_tool import NBAQueryTool
except ImportError as e:
    st.error(f"Erreur d'importation: {e}")
    st.stop()

# --- Configuration des logs ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logfire.configure()
logfire.info("🚀 Démarrage de l’application MistralChat (RAG + SQL)")

# --- Initialisation Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée dans le .env.")
    st.stop()

try:
    client = MistralClient(api_key=api_key)
    logfire.info("✅ Client Mistral initialisé", extra={"model": model})
except Exception as e:
    st.error(f"Erreur lors de l’initialisation du client Mistral : {e}")
    st.stop()

# --- Initialisation des outils ---
sql_tool = NBAQueryTool()
vector_store_manager = VectorStoreManager()

# --- Few-Shot SQL Prompts ---
FEW_SHOT_SQL = """
Voici des exemples de correspondance entre des questions NBA et des requêtes SQL valides.
Utilise toujours les bonnes colonnes en fonction de la statistique demandée.

EXEMPLES :

Question : Qui a marqué le plus de points cette saison ?
SQL :
SELECT p.name, s.points
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.points DESC
LIMIT 5;

---

Question : Quels sont les joueurs avec le meilleur pourcentage à 3 points ?
SQL :
SELECT p.name, s.three_pct
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.three_pct DESC
LIMIT 5;

---

Question : Quels sont les meilleurs passeurs de la ligue ?
SQL :
SELECT p.name, s.assists
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.assists DESC
LIMIT 5;

---

Question : Quels sont les meilleurs rebondeurs ?
SQL :
SELECT p.name, s.rebounds
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.rebounds DESC
LIMIT 5;

---

Question : Quels joueurs ont le meilleur pourcentage au tir global ?
SQL :
SELECT p.name, s.fg_pct
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.fg_pct DESC
LIMIT 5;

---

Question : Quels sont les meilleurs défenseurs selon les contres ?
SQL :
SELECT p.name, s.blocks
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.blocks DESC
LIMIT 5;

---

Question : Quels joueurs ont le meilleur nombre d'interceptions ?
SQL :
SELECT p.name, s.steals
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.steals DESC
LIMIT 5;

---

Question : Quelle équipe a le meilleur ratio de victoire ?
SQL :
SELECT team_name, wins, losses, ROUND(wins::float / (wins + losses), 3) AS win_ratio
FROM teams
ORDER BY win_ratio DESC
LIMIT 5;

---

Question : Quelle équipe a la meilleure évaluation offensive ?
SQL :
SELECT team_name, offensive_rating
FROM teams
ORDER BY offensive_rating DESC
LIMIT 5;

---

Question : Qui a le meilleur différentiel net (NETRTG) ?
SQL :
SELECT p.name, s.net_rating
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.net_rating DESC
LIMIT 5;

---

À partir de la question suivante, génère uniquement la requête SQL la plus pertinente sans texte explicatif :
"""

# --- Détection de question chiffrée ---
def is_sql_question(prompt: str) -> bool:
    """
    Détecte si une question nécessite des données chiffrées.
    """
    patterns = [
        r"\b(moyenne|pourcentage|classement|points|rebonds|passes|victoires|matchs|bilan)\b",
        r"\b(top|meilleur|plus|record)\b",
        r"\b(stat|score|équipe|joueur)\b"
    ]
    return any(re.search(p, prompt.lower()) for p in patterns)

# --- Génération via Mistral ---
def mistral_generate(prompt: str) -> str:
    try:
        response = client.chat(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erreur API Mistral : {e}")
        return f"Erreur API Mistral : {e}"

# --- Interface Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant NBA intelligent | Modèle : {model}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Bonjour ! Pose-moi des questions sur les joueurs, équipes ou statistiques NBA."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Entrée utilisateur ---
if prompt := st.chat_input("Pose ta question sur la NBA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # === Étape 1 : Détection ===
    if is_sql_question(prompt):
        logfire.info("🔎 Question détectée comme chiffrée → appel du SQL Tool")
        few_shot_prompt = FEW_SHOT_SQL + f"\nQuestion : {prompt}\nSQL :"
        generated_sql = mistral_generate(few_shot_prompt)

        st.markdown(f"```sql\n{generated_sql}\n```")

        # Exécution de la requête
        sql_result = sql_tool._run(generated_sql)
        with st.chat_message("assistant"):
            st.write(sql_result)

        st.session_state.messages.append({"role": "assistant", "content": sql_result})
    else:
        # === Étape 2 : RAG standard ===
        logfire.info("🧠 Question détectée comme contextuelle → RAG FAISS")
        with logfire.span("Recherche contexte FAISS"):
            results = vector_store_manager.search(prompt, k=SEARCH_K)
            context = "\n\n---\n\n".join([res["text"] for res in results]) if results else "Aucune information trouvée."

        rag_prompt = f"""Tu es un expert NBA.
{context}

Question : {prompt}
Réponse :"""
        response = mistral_generate(rag_prompt)

        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("⚙️ Powered by Mistral AI, FAISS & PostgreSQL | Agent RAG + SQL par Logfire")

