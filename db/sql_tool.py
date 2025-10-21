import psycopg2
import pandas as pd
import logging
from langchain.tools import BaseTool
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    logging.warning("⚠️ Aucune clé API Mistral détectée dans le .env")

DB_CONFIG = {
    "dbname": "nba_db",
    "user": "postgres",
    "password": "naruto",  # adapte à ton mot de passe
    "host": "localhost",
    "port": "5432"
}

# --- CLIENT MISTRAL ---
try:
    client = MistralClient(api_key=MISTRAL_API_KEY)
    MODEL_NAME = "mistral-small-latest"
    logging.info("✅ Client Mistral initialisé")
except Exception as e:
    client = None
    MODEL_NAME = None
    logging.warning(f"⚠️ Impossible d'initialiser Mistral : {e}")


# --- OUTIL SQL PRINCIPAL ---
class NBAQueryTool(BaseTool):
    name: str = "NBA SQL Tool"
    description: str = (
        "Permet d'exécuter des requêtes SQL sur la base NBA PostgreSQL "
        "et de générer une brève explication automatique du résultat."
    )

    def _run(self, query: str) -> str:
        """
        Exécute une requête SQL sur la base NBA et génère une explication via Mistral.
        """
        try:
            # --- Nettoyage de la requête ---
            cleaned_query = query.strip().replace("```sql", "").replace("```", "").strip()

            # --- Connexion et exécution ---
            conn = psycopg2.connect(**DB_CONFIG)
            df = pd.read_sql_query(cleaned_query, conn)
            conn.close()

            if df.empty:
                return "Aucun résultat trouvé pour cette requête."

            # --- Génération de l’explication ---
            explanation = self._generate_explanation(cleaned_query, df)

            # --- Formatage du tableau ---
            result_md = df.head(10).to_markdown(index=False)
            logging.info("✅ Requête exécutée avec succès :\n%s", cleaned_query)

            return f"{explanation}\n\n{result_md}"

        except Exception as e:
            logging.error(f"❌ Erreur lors de l'exécution SQL : {e}")
            return f"Erreur SQL : {e}"

    def _arun(self, query: str):
        raise NotImplementedError("L'exécution asynchrone n'est pas encore supportée.")

    # --- 🧠 Génération d'explication dynamique ---
    def _generate_explanation(self, query: str, df: pd.DataFrame) -> str:
        """
        Crée une brève explication du résultat SQL.
        Si Mistral est disponible, génère une explication naturelle.
        Sinon, utilise une version locale simple.
        """
        # Mode local (fallback)
        if client is None:
            return self._generate_simple_explanation(query)

        # Contexte : tableau des 5 premières lignes
        context = df.head(5).to_markdown(index=False)
        prompt = f"""
Tu es un expert en analyse NBA. Voici une requête SQL exécutée sur une base de données NBA :

REQUÊTE :
{query}

RÉSULTATS :
{context}

Rédige une courte explication (2-3 phrases maximum) en français,
comme si tu étais un commentateur NBA, expliquant ce que montrent ces résultats.
Ne redonne pas le code SQL ni le tableau.
"""

        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.2,
            )
            if response.choices and len(response.choices) > 0:
                return f"🧠 {response.choices[0].message.content.strip()}"
        except Exception as e:
            logging.warning(f"⚠️ Erreur pendant la génération de l'explication Mistral : {e}")

        # Fallback si erreur API
        return self._generate_simple_explanation(query)

    # --- 🔤 Explication locale simple si pas d'API ---
    def _generate_simple_explanation(self, query: str) -> str:
        if "three_pct" in query:
            return "🏀 Voici les joueurs avec le meilleur pourcentage à 3 points cette saison :"
        elif "points" in query and "three_pct" not in query:
            return "🔥 Voici les meilleurs scoreurs de la ligue :"
        elif "assists" in query:
            return "🎯 Voici les meilleurs passeurs de la ligue :"
        elif "rebounds" in query:
            return "🧱 Voici les meilleurs rebondeurs :"
        elif "blocks" in query:
            return "🛡️ Voici les meilleurs contreurs (défenseurs) :"
        elif "steals" in query:
            return "👀 Voici les joueurs les plus performants en interceptions :"
        elif "offensive_rating" in query:
            return "⚡ Voici les équipes avec la meilleure efficacité offensive :"
        elif "defensive_rating" in query:
            return "🧱 Voici les équipes les plus solides en défense :"
        elif "net_rating" in query:
            return "📈 Voici les joueurs avec le meilleur différentiel net (Net Rating) :"
        else:
            return "📊 Voici le résultat de ta requête SQL :"


# --- TEST LOCAL ---
if __name__ == "__main__":
    logging.info("🔍 Test local du SQL Tool avec explication dynamique...")
    tool = NBAQueryTool()
    q = """
    SELECT p.name, s.points, s.fg_pct, s.three_pct
    FROM players p
    JOIN stats s ON p.player_id = s.player_id
    ORDER BY s.points DESC
    LIMIT 5;
    """
    print(tool._run(q))



