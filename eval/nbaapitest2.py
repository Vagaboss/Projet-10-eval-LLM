"""
Trouve le joueur NBA ayant marqué le plus de points
sur ses 10 derniers matchs.
Source : stats.nba.com via nba_api
"""

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import pandas as pd
from time import sleep

def best_scorer_last10(season="2024-25"):
    print(f"🔍 Recherche du meilleur scoreur sur les 10 derniers matchs ({season})...\n")

    all_players = players.get_active_players()
    results = []

    for p in all_players:
        try:
            # Récupérer le log de matchs du joueur
            log = playergamelog.PlayerGameLog(player_id=p["id"], season=season, season_type_all_star="Regular Season")
            frames = log.get_data_frames()
            df = frames.pop(0)

            # Sauter les joueurs sans match
            if df.empty:
                continue

            # Prendre les 10 derniers matchs
            last10 = df.head(10)

            # Calcul du total de points
            total_points = last10["PTS"].sum()

            results.append({
                "player": p["full_name"],
                "total_points_last10": total_points
            })

            # Pause pour éviter d’être bloqué par l’API
            sleep(0.25)

        except Exception:
            continue

    if not results:
        print("❌ Aucune donnée récupérée. Vérifie la saison ou la connexion à l'API.")
        return

    # Créer un DataFrame trié par points décroissants
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values(by="total_points_last10", ascending=False).reset_index(drop=True)

    print("🏀 Top 10 des meilleurs scoreurs sur leurs 10 derniers matchs :\n")
    print(df_sorted.head(10).to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🎯 Meilleur scoreur : {best['player']} avec {best['total_points_last10']} points sur ses 10 derniers matchs.")

if __name__ == "__main__":
    best_scorer_last10()

