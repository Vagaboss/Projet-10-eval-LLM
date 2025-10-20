import psycopg2
import pandas as pd
import logging
from langchain.tools import BaseTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_CONFIG = {
    "dbname": "nba_db",
    "user": "postgres",
    "password": "naruto",  # adapte à ton mot de passe
    "host": "localhost",
    "port": "5432"
}


class NBAQueryTool(BaseTool):
    name: str = "NBA SQL Tool"
    description: str = "Permet d'exécuter des requêtes SQL sur la base NBA PostgreSQL."

    def _run(self, query: str) -> str:
        """
        Exécute une requête SQL propre sur la base PostgreSQL NBA.
        Nettoie la requête avant exécution pour éviter les erreurs de formatage.
        """
        try:
            # Nettoyage de la requête (supprime les ```sql ... ``` éventuels)
            cleaned_query = query.strip().replace("```sql", "").replace("```", "").strip()

            conn = psycopg2.connect(**DB_CONFIG)
            df = pd.read_sql_query(cleaned_query, conn)
            conn.close()

            if df.empty:
                return "Aucun résultat trouvé pour cette requête."

            # Affichage propre
            result_md = df.head(10).to_markdown(index=False)
            logging.info("✅ Requête exécutée avec succès :\n%s", cleaned_query)
            return result_md

        except Exception as e:
            logging.error(f"❌ Erreur lors de l'exécution SQL : {e}")
            return f"Erreur SQL : {e}"

    def _arun(self, query: str):
        raise NotImplementedError("L'exécution asynchrone n'est pas encore supportée.")


if __name__ == "__main__":
    logging.info("🔍 Test local du SQL Tool...")
    tool = NBAQueryTool()
    q = """
    SELECT p.name, s.points, s.fg_pct, s.three_pct
    FROM players p
    JOIN stats s ON p.player_id = s.player_id
    ORDER BY s.points DESC
    LIMIT 5;
    """
    print(tool._run(q))


