"""
Script : Trouver le joueur NBA avec le meilleur pourcentage à 3 points sur les 5 derniers matchs
Source : stats.nba.com via nba_api
Auteur : [Ton nom]
"""

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import pandas as pd
from time import sleep

def get_best_three_point_shooter_last5(season="2024-25"):
    all_players = players.get_active_players()
    results = []

    print(f"🔍 Analyse des joueurs actifs pour la saison {season}...")

    for p in all_players:
        try:
            # Récupération des logs de matchs
            gamelog = playergamelog.PlayerGameLog(
                player_id=p["id"], season=season, season_type_all_star="Regular Season"
            ).get_data_frames()[0]

            # Prendre les 5 derniers matchs
            last5 = gamelog.head(5)

            # Vérification qu’il y a bien des tirs tentés à 3 points
            if "FG3A" not in last5.columns or last5["FG3A"].sum() == 0:
                continue

            # Calcul du % de réussite à 3 points
            fg3m = last5["FG3M"].sum()
            fg3a = last5["FG3A"].sum()
            three_pct = (fg3m / fg3a) * 100 if fg3a > 0 else 0

            results.append({
                "player": p["full_name"],
                "3P%": round(three_pct, 2),
                "FG3M": fg3m,
                "FG3A": fg3a
            })

            # Pause légère pour éviter les limites de requêtes NBA
            sleep(0.3)

        except Exception:
            continue

    # Création du DataFrame de résultats
    df = pd.DataFrame(results)

    if df.empty:
        print("⚠️ Aucune donnée récupérée. Vérifie ta connexion ou la saison.")
        return

    # Tri décroissant par pourcentage
    df = df.sort_values(by="3P%", ascending=False).reset_index(drop=True)

    print("\n🏀 Joueurs avec le meilleur pourcentage à 3 points sur les 5 derniers matchs :\n")
    print(df.head(10).to_string(index=False))

    best_player = df.iloc[0]
    print("\n🎯 Joueur avec le meilleur % à 3 points :")
    print(f"{best_player['player']} → {best_player['3P%']}% ({int(best_player['FG3M'])}/{int(best_player['FG3A'])})")

if __name__ == "__main__":
    get_best_three_point_shooter_last5()
