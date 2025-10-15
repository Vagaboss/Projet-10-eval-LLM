import streamlit as st
import os
import logging
import time
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
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

# --- Configuration des logs ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialisation Logfire ---
logfire.configure()
logfire.info("🚀 Démarrage de l’application MistralChat (RAG)")

# --- Configuration de l’API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée dans le .env.")
    st.stop()

try:
    client = MistralClient(api_key=api_key)
    logfire.info("Client Mistral initialisé avec succès", extra={"model": model})
except Exception as e:
    st.error(f"Erreur lors de l’initialisation du client Mistral : {e}")
    logfire.error("Erreur client Mistral", extra={"exception": str(e)})
    st.stop()

# --- Chargement du Vector Store ---
@st.cache_resource
def get_vector_store_manager():
    with logfire.span("Chargement du VectorStoreManager"):
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            logfire.warning("VectorStoreManager vide ou non initialisé")
            st.warning("L’index FAISS est vide. Lancez `python indexer.py` avant.")
            return None
        logfire.info("VectorStoreManager chargé", extra={"vecteurs": manager.index.ntotal})
        return manager

vector_store_manager = get_vector_store_manager()

# --- Prompt système RAG ---
SYSTEM_PROMPT = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue NBA.
Réponds de façon claire et analytique aux questions des fans.

---
{{context_str}}
---

QUESTION DU FAN :
{{question}}

RÉPONSE DE L'ANALYSTE NBA :
"""

# --- Initialisation de la session ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"👋 Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les stats !"
    }]

# --- Fonction de génération de réponse ---
@logfire.instrument("Génération de réponse via Mistral")
def generer_reponse(prompt_messages: list[ChatMessage]) -> str:
    """Envoie le prompt à l’API Mistral et retourne la réponse."""
    if not prompt_messages:
        logfire.warning("Appel à generer_reponse avec prompt vide")
        return "Je ne peux pas traiter une demande vide."

    try:
        with logfire.span("Appel API Mistral"):
            start_time = time.time()
            response = client.chat(
                model=model,
                messages=prompt_messages,
                temperature=0.1,
            )
            elapsed = round(time.time() - start_time, 2)

        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            logfire.info("Réponse Mistral générée", extra={
                "temps_reponse_s": elapsed,
                "longueur": len(content)
            })
            return content
        else:
            logfire.warning("Réponse Mistral vide ou invalide")
            return "Désolé, je n’ai pas pu générer de réponse valide."

    except Exception as e:
        logfire.error("Erreur pendant l’appel API Mistral", extra={"exception": str(e)})
        return f"Erreur Mistral : {e}"

# --- Interface Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel NBA | Modèle : {model}")

# --- Historique de chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Saisie utilisateur ---
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    with logfire.span("Nouvelle question utilisateur", extra={"question": prompt}):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # --- Étape 1 : Recherche FAISS ---
        with logfire.span("Recherche de contexte dans Vector Store"):
            if vector_store_manager is None:
                logfire.error("VectorStoreManager non disponible")
                st.error("Index FAISS non disponible.")
                st.stop()

            start_time = time.time()
            search_results = vector_store_manager.search(prompt, k=SEARCH_K)
            elapsed = round(time.time() - start_time, 2)
            logfire.info("Recherche FAISS terminée", extra={
                "nb_chunks": len(search_results),
                "duree_s": elapsed
            })

        # --- Étape 2 : Préparation du contexte ---
        if search_results:
            context_str = "\n\n---\n\n".join([
                f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
                for res in search_results
            ])
            logfire.info("Contexte trouvé", extra={"nb_chunks": len(search_results)})
        else:
            context_str = "Aucune information pertinente trouvée."
            logfire.warning("Aucun contexte pertinent trouvé pour la question")

        # --- Étape 3 : Génération de la réponse ---
        with logfire.span("Appel Mistral avec contexte"):
            final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)
            messages_for_api = [ChatMessage(role="user", content=final_prompt)]
            response_content = generer_reponse(messages_for_api)

        # --- Étape 4 : Affichage et historique ---
        with st.chat_message("assistant"):
            st.write(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})

        logfire.info("Réponse affichée", extra={
            "question": prompt,
            "réponse_partielle": response_content[:150]
        })

# --- Pied de page ---
st.markdown("---")
st.caption("⚙️ Powered by Mistral AI & FAISS | Tracé en direct avec Pydantic Logfire")
