import streamlit as st
import os
import logging
import re
import pandas as pd
import logfire
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
    from db.plot_tool import PlotTool
except ImportError as e:
    st.error(f"Erreur d'importation: {e}")
    st.stop()

# --- Configuration logging + Logfire ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logfire.configure()
logfire.info("🚀 Démarrage de l’application MistralChat (RAG + SQL + PLOT complet)")

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
plot_tool = PlotTool()
vector_store_manager = VectorStoreManager()

# --- Few-Shot SQL Prompt ---
FEW_SHOT_SQL = """
Tu es un assistant expert en NBA et en SQL.
Utilise le schéma suivant pour répondre aux questions chiffrées :

TABLES :
- players(player_id, name, team_code, age)
- stats(player_id, games_played, points, rebounds, assists, steals, blocks, turnovers, fg_pct, three_pct, ft_pct, offensive_rating, defensive_rating, net_rating, pace, pie)
- teams(team_id, team_code, team_name, wins, losses, offensive_rating, defensive_rating, net_rating)

RÈGLES :
1️⃣ Retourne toujours du SQL PostgreSQL valide.
2️⃣ N’invente jamais de colonnes ni de tables (utilise les noms ci-dessus uniquement).
3️⃣ Ne renvoie **que** le code SQL, sans explication.

---

Question : {question}
SQL :
"""

# --- Few-Shot Plot Prompt ---
FEW_SHOT_PLOT = """
Voici des exemples de correspondance entre des questions NBA et des visualisations basées sur les statistiques globales.

SCHÉMA DISPONIBLE :
- players(player_id, name, team_code)
- stats(player_id, points, rebounds, assists, fg_pct, three_pct, ft_pct, net_rating, pie)
- teams(team_id, team_code, team_name, wins, losses, offensive_rating, defensive_rating, net_rating)

EXEMPLES :

Question : Compare les meilleurs scoreurs de la NBA.
Réponse attendue :
SQL :
SELECT p.name, s.points
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.points DESC
LIMIT 10;
Graphique : Barres

Question : Montre les 10 joueurs avec le meilleur pourcentage à 3 points.
Réponse attendue :
SQL :
SELECT p.name, s.three_pct
FROM players p
JOIN stats s ON p.player_id = s.player_id
ORDER BY s.three_pct DESC
LIMIT 10;
Graphique : Barres

Question : Compare le net rating des 10 meilleures équipes.
Réponse attendue :
SQL :
SELECT team_name, net_rating
FROM teams
ORDER BY net_rating DESC
LIMIT 10;
Graphique : Barres horizontales
"""

# --- Détection automatique ---
def is_sql_question(prompt: str) -> bool:
    patterns = [
        r"\b(points|rebonds|passes|victoires|défaites|pourcentage|classement|ratio|moyenne|record|rating|pie)\b",
        r"\b(top|meilleur|plus|moins|classement)\b",
        r"\b(équipe|joueur|performance|efficacité)\b"
    ]
    return any(re.search(p, prompt.lower()) for p in patterns)

def is_plot_question(prompt: str) -> bool:
    patterns = [
        r"\b(graphique|courbe|visualise|montre|compare|évolution|diagramme|barres|camembert|histogramme)\b"
    ]
    return any(re.search(p, prompt.lower()) for p in patterns)

# --- Génération Mistral ---
def mistral_generate(prompt: str) -> str:
    try:
        with logfire.span("Génération via Mistral"):
            response = client.chat(
                model=model,
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        logfire.error(f"Erreur API Mistral : {e}")
        return f"Erreur API Mistral : {e}"

# --- Interface Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant NBA intelligent | Modèle : {model}")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Bonjour ! Pose-moi des questions sur les joueurs, équipes ou statistiques NBA."
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Entrée utilisateur ---
if prompt := st.chat_input("Pose ta question sur la NBA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- Détection textuelle ---
    is_textual = any(word in prompt.lower() for word in [
        "selon", "sources", "rapport", "texte", "analyse", "écrites", "mentionné", "décrit", "articles"
    ])

    # --- Cas 1 : Question purement textuelle ---
    if is_textual:
        logfire.info("🧠 Question purement contextuelle → RAG FAISS prioritaire", extra={"prompt": prompt})
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
        logfire.info("✅ Réponse RAG générée avec succès", extra={"type": "RAG"})

    # --- Cas 2 : Question SQL ou graphique ---
    elif is_sql_question(prompt):
        use_plot = is_plot_question(prompt)
        logfire.info("🔢 Question SQL détectée", extra={"prompt": prompt, "with_plot": use_plot})

        few_shot_prompt = FEW_SHOT_PLOT + f"\nQuestion : {prompt}\nRéponse :" if use_plot else FEW_SHOT_SQL.format(question=prompt)
        generated_sql = mistral_generate(few_shot_prompt)
        st.markdown(f"```sql\n{generated_sql}\n```")

        with logfire.span("Exécution SQL"):
            sql_result = sql_tool._run(generated_sql)
        df = getattr(sql_tool, "last_df", pd.DataFrame())
        response = sql_result

        if use_plot:
            logfire.info("📊 Tentative de génération graphique", extra={"rows": len(df)})
            if df is not None and not df.empty and len(df.columns) >= 2:
                img_path = plot_tool._run(df.iloc[:, :2], chart_type="barres", title=f"Top {len(df)} {prompt}")
                st.image(img_path)
                logfire.info("✅ Graphique généré avec succès", extra={"file": img_path})
            else:
                st.warning("⚠️ Impossible de tracer le graphique : données insuffisantes.")
                logfire.warning("⚠️ Échec de génération du graphique : données insuffisantes")

        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        logfire.info("✅ Réponse SQL ou graphique générée avec succès", extra={"type": "SQL" if not use_plot else "SQL+PLOT"})

    # --- Cas 3 : Fallback → RAG standard ---
    else:
        logfire.info("🧠 Question contextuelle → RAG FAISS (fallback)", extra={"prompt": prompt})
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
        logfire.info("✅ Réponse RAG (fallback) générée", extra={"type": "RAG"})

st.markdown("---")
st.caption("⚙️ Powered by Mistral AI, FAISS, PostgreSQL, Matplotlib & Logfire | Agent RAG + SQL + PLOT complet et traçable")
