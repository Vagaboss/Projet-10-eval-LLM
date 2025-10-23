"""
MistralChat.py — Agent RAG + SQL + PLOT (corrigé pour prioriser les questions textuelles)
"""

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

# --- Configuration logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logfire.configure()
logfire.info("🚀 Démarrage de l’application MistralChat (RAG + SQL + PLOT)")

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
2️⃣ N’invente jamais de colonnes (utilise les noms ci-dessus).
3️⃣ Ne renvoie **que** le code SQL, sans explication.

---

Question : {question}
SQL :
"""

# --- Détection automatique ---
def is_sql_question(prompt: str) -> bool:
    patterns = [
        r"\b(points|rebonds|passes|victoires|défaites|pourcentage|bilan|classement|ratio|matchs|stat|score|moyenne|record)\b",
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
        response = client.chat(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erreur API Mistral : {e}")
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

    # --- Détection textuelle prioritaire ---
    is_textual = any(word in prompt.lower() for word in [
        "selon", "sources", "rapport", "texte", "analyse", "écrites", "mentionné", "décrit", "articles"
    ])

    # === Cas 1 : Question purement textuelle → RAG direct ===
    if is_textual:
        logfire.info("🧠 Question purement contextuelle → RAG FAISS prioritaire")
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

    # === Cas 2 : Question chiffrée (SQL/RAG/PLOT mixte) ===
    elif is_sql_question(prompt):
        use_rag = any(word in prompt.lower() for word in ["mentionné", "source", "rapport", "texte", "analyse"])
        use_plot = is_plot_question(prompt)
        logfire.info("🔢 Question SQL détectée" + (" + RAG" if use_rag else "") + (" + PLOT" if use_plot else ""))

        # --- Génération SQL ---
        few_shot_prompt = FEW_SHOT_SQL.format(question=prompt)
        generated_sql = mistral_generate(few_shot_prompt)
        st.markdown(f"```sql\n{generated_sql}\n```")

        # --- Exécution SQL ---
        sql_result = sql_tool._run(generated_sql)
        df = getattr(sql_tool, "last_df", pd.DataFrame())
        response = sql_result

        # --- Fusion SQL + RAG ---
        if use_rag:
            player_names = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", sql_result)
            if player_names:
                logfire.info("🧩 Fusion SQL + FAISS — recherche contextuelle sur les joueurs extraits")
                combined_contexts = []
                for name in player_names[:3]:
                    rag_results = vector_store_manager.search(name, k=2)
                    for res in rag_results:
                        combined_contexts.append(res["text"])
                context = "\n\n---\n\n".join(combined_contexts) if combined_contexts else "Aucune information textuelle trouvée."

                fusion_prompt = f"""
Tu es un analyste NBA. Combine les statistiques suivantes et les extraits textuels issus de discussions réelles.
Appuie-toi sur ces extraits pour justifier la réponse. Si une information chiffrée apparaît dans le texte, cite-la.

STATISTIQUES :
{sql_result}

SOURCES TEXTUELLES :
{context}

Question : {prompt}
Rédige une réponse concise et analytique (3 phrases maximum) :
"""
                response = mistral_generate(fusion_prompt)

        # --- Graphique SQL ---
        if use_plot:
            if df is not None and not df.empty and len(df.columns) >= 2:
                st.write("📊 Voici le graphique correspondant :")
                img_path = plot_tool._run(
                    df.iloc[:, :2],
                    chart_type="barres",
                    title=f"Top {len(df)} {prompt}"
                )
                st.image(img_path)
                st.success("✅ Graphique généré à partir des données SQL")
            else:
                st.warning("⚠️ Impossible de tracer le graphique : données insuffisantes.")

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # === Cas 3 : RAG standard ===
    else:
        logfire.info("🧠 Question contextuelle → RAG FAISS")
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
st.caption("⚙️ Powered by Mistral AI, FAISS, PostgreSQL & Matplotlib | Agent RAG + SQL + PLOT robuste par Logfire")
