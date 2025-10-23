"""
sql_tool.py — Outil SQL NBA (auto-correction, explication, et compatibilité Plot)
"""

import psycopg2
import pandas as pd
import logging
import re
from langchain.tools import BaseTool
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
from dotenv import load_dotenv

# --- CONFIGURATION LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- CHARGEMENT DES VARIABLES D’ENVIRONNEMENT ---
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    logging.warning("⚠️ Aucune clé API Mistral détectée dans le .env")

# --- CONFIGURATION BASE DE DONNÉES ---
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

# --- SYNONYMES DE COLONNES ---
COLUMN_SYNONYMS = {
    "free_throw_pct": "ft_pct",
    "free_throw_percentage": "ft_pct",
    "three_points_pct": "three_pct",
    "three_points_percentage": "three_pct",
    "field_goal_pct": "fg_pct",
    "field_goals_pct": "fg_pct",
    "rebounds_per_game": "rebounds",
    "assists_per_game": "assists",
    "points_per_game": "points",
    "netrating": "net_rating",
    "offrating": "offensive_rating",
    "defrating": "defensive_rating"
}


# --- OUTIL SQL PRINCIPAL ---
class NBAQueryTool(BaseTool):
    name: str = "NBA SQL Tool"
    description: str = (
        "Permet d'exécuter des requêtes SQL sur la base NBA PostgreSQL, "
        "corrige les erreurs de colonnes et génère une explication automatique du résultat."
    )

    # ✅ Sauvegarde du dernier DataFrame pour compatibilité PlotTool
    last_df: pd.DataFrame = pd.DataFrame()

    def _run(self, query: str) -> str:
        """Exécute une requête SQL sur la base NBA et génère une explication."""
        try:
            # --- Nettoyage de la requête ---
            cleaned_query = query.strip().replace("```sql", "").replace("```", "").strip()

            # --- Correction automatique des colonnes ---
            for wrong, correct in COLUMN_SYNONYMS.items():
                if re.search(rf"\b{wrong}\b", cleaned_query, re.IGNORECASE):
                    logging.info(f"🔧 Correction : '{wrong}' → '{correct}'")
                    cleaned_query = re.sub(rf"\b{wrong}\b", correct, cleaned_query, flags=re.IGNORECASE)

            # --- Connexion et exécution ---
            conn = psycopg2.connect(**DB_CONFIG)
            df = pd.read_sql_query(cleaned_query, conn)
            conn.close()

            # --- Si résultat vide ---
            if df.empty:
                self.last_df = pd.DataFrame()
                return "Aucun résultat trouvé pour cette requête."

            # ✅ Stockage du DataFrame pour le PlotTool
            self.last_df = df.copy()

            # --- Génération d'explication ---
            explanation = self._generate_explanation(cleaned_query, df)

            # --- Formatage Markdown ---
            result_md = df.head(10).to_markdown(index=False)
            logging.info("✅ Requête exécutée avec succès :\n%s", cleaned_query)

            return f"{explanation}\n\n{result_md}"

        except Exception as e:
            logging.error(f"❌ Erreur lors de l'exécution SQL : {e}")
            self.last_df = pd.DataFrame()  # vide en cas d’erreur
            return f"Erreur SQL : {e}"

    def _arun(self, query: str):
        raise NotImplementedError("L'exécution asynchrone n'est pas encore supportée.")

    # --- 🧠 Génération d'explication dynamique ---
    def _generate_explanation(self, query: str, df: pd.DataFrame) -> str:
        """Crée une brève explication du résultat SQL."""
        if client is None:
            return self._generate_simple_explanation(query)

        context = df.head(5).to_markdown(index=False)
        prompt = f"""
Tu es un expert en analyse NBA. Voici une requête SQL exécutée sur une base de données NBA.

SCHÉMA DES TABLES :
- players(player_id, name, team_code, age)
- stats(player_id, games_played, points, rebounds, assists, steals, blocks, turnovers, fg_pct, three_pct, ft_pct, offensive_rating, defensive_rating, net_rating, pace, pie)
- teams(team_id, team_code, team_name, wins, losses, offensive_rating, defensive_rating, net_rating)

REQUÊTE :
{query}

RÉSULTATS :
{context}

Rédige une explication courte (2 phrases max) pour un public de fans NBA.
Ne répète pas le code SQL.
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
        return self._generate_simple_explanation(query)

    # --- 💬 Fallback local (si pas d'API) ---
    def _generate_simple_explanation(self, query: str) -> str:
        if "three_pct" in query:
            return "🏀 Voici les joueurs avec le meilleur pourcentage à 3 points :"
        elif "points" in query and "three_pct" not in query:
            return "🔥 Voici les meilleurs scoreurs :"
        elif "assists" in query:
            return "🎯 Voici les meilleurs passeurs :"
        elif "rebounds" in query:
            return "🧱 Voici les meilleurs rebondeurs :"
        elif "blocks" in query:
            return "🛡️ Voici les meilleurs contreurs :"
        elif "steals" in query:
            return "👀 Voici les meilleurs intercepteurs :"
        elif "offensive_rating" in query:
            return "⚡ Voici les équipes les plus efficaces offensivement :"
        elif "defensive_rating" in query:
            return "🧱 Voici les équipes les plus solides défensivement :"
        elif "net_rating" in query:
            return "📈 Voici les joueurs avec le meilleur différentiel net :"
        else:
            return "📊 Voici le résultat de ta requête SQL :"


# --- TEST LOCAL ---
if __name__ == "__main__":
    logging.info("🔍 Test local du SQL Tool (avec correction + explication + compatibilité Plot)...")
    tool = NBAQueryTool()
    q = """
    SELECT p.name, s.points, s.fg_pct, s.three_points_percentage
    FROM players p
    JOIN stats s ON p.player_id = s.player_id
    ORDER BY s.points DESC
    LIMIT 5;
    """
    print(tool._run(q))
    print("\n✅ Aperçu DataFrame :")
    print(tool.last_df.head())




