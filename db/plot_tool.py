"""
plot_tool.py — Génération dynamique de graphiques NBA (outil compatible LangChain)
"""

import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import logging
from langchain.tools import BaseTool

# --- Configuration logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PlotTool(BaseTool):
    name: str = "NBA Plot Tool"
    description: str = (
        "Génère des graphiques (barres, courbes, camemberts) à partir de données NBA. "
        "Utilisable pour visualiser des statistiques SQL ou FAISS."
    )

    def _run(self, data, chart_type="barres", title="Visualisation NBA", x_label=None, y_label=None):
        """
        Génère un graphique à partir d'une liste, d'un JSON ou d'un DataFrame.
        Retourne le chemin du fichier image (.png) généré.
        """
        try:
            # --- Conversion en DataFrame ---
            if isinstance(data, str):
                df = pd.read_json(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Format de données non supporté : doit être DataFrame, liste ou JSON")

            if df.empty:
                return "⚠️ Aucune donnée à visualiser."

            # --- Choix du type de graphique ---
            fig, ax = plt.subplots(figsize=(8, 5))

            if chart_type.lower() in ["barres", "bar", "histogramme"]:
                ax.bar(df.iloc[:, 0], df.iloc[:, 1])
            elif chart_type.lower() in ["courbe", "line"]:
                ax.plot(df.iloc[:, 0], df.iloc[:, 1], marker="o")
            elif chart_type.lower() in ["camembert", "pie"]:
                ax.pie(df.iloc[:, 1], labels=df.iloc[:, 0], autopct="%1.1f%%", startangle=90)
                ax.axis("equal")  # Rendre le camembert circulaire
            else:
                return f"⚠️ Type de graphique non reconnu : {chart_type}"

            # --- Mise en forme ---
            ax.set_title(title)
            if x_label:
                ax.set_xlabel(x_label)
            if y_label:
                ax.set_ylabel(y_label)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # --- Sauvegarde temporaire ---
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name)
            plt.close(fig)

            logging.info(f"✅ Graphique généré : {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logging.error(f"❌ Erreur lors de la génération du graphique : {e}")
            return f"Erreur de génération de graphique : {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Exécution asynchrone non supportée.")


# --- Test local ---
if __name__ == "__main__":
    logging.info("🔍 Test du PlotTool en local...")

    # Exemple de données NBA
    data = [
        {"Joueur": "Shai Gilgeous-Alexander", "Points": 2485},
        {"Joueur": "Anthony Edwards", "Points": 2180},
        {"Joueur": "Nikola Jokic", "Points": 2072},
        {"Joueur": "Luka Doncic", "Points": 2313},
        {"Joueur": "Jayson Tatum", "Points": 2225},
    ]

    tool = PlotTool()
    path = tool._run(data, chart_type="barres", title="Top 5 Scoreurs NBA 2024", x_label="Joueur", y_label="Points")
    print(f"Graphique enregistré ici : {path}")
