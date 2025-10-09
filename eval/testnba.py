"""
Compare les statistiques de rebonds des Los Angeles Lakers
à domicile et à l’extérieur pour la saison 2024-25.
Utilise l’endpoint 'leaguedashteamstats' (données toujours disponibles).
"""

from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd

def compare_lakers_rebounds(season="2024-25"):
    print(f"🔍 Récupération des statistiques de la saison {season}...")

    # Charger les statistiques agrégées par équipe et par lieu (Home/Away)
    data = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Base",  # type de mesure
        per_mode_detailed="PerGame",           # par match
        location_nullable=None,                # récupère toutes les données
        season_type_all_star="Regular Season"  # saison régulière
    )

    frames = data.get_data_frames()
    df = frames.pop(0)

    if df.empty:
        print(f"❌ Aucune donnée trouvée pour la saison {season}.")
        return

    # Filtrer les lignes pour les Lakers à domicile et à l’extérieur
    # Le paramètre location_nullable=None renvoie un champ LOCATION
    # On appelle deux fois l’API, une pour Home et une pour Away

    home_data = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        location_nullable="Home",
        season_type_all_star="Regular Season"
    ).get_data_frames().pop(0)

    away_data = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        location_nullable="Road",
        season_type_all_star="Regular Season"
    ).get_data_frames().pop(0)

    # Extraire les rebonds pour les Lakers
    reb_home = home_data.loc[home_data["TEAM_NAME"] == "Los Angeles Lakers", "REB"].values[0]
    reb_away = away_data.loc[away_data["TEAM_NAME"] == "Los Angeles Lakers", "REB"].values[0]

    print(f"\n📊 Saison {season} - Los Angeles Lakers")
    print(f"🏠 Rebond moyen à domicile  : {reb_home:.2f}")
    print(f"🚌 Rebond moyen à l’extérieur : {reb_away:.2f}")

    diff = reb_home - reb_away
    if diff > 0:
        print(f"\n➡️ Les Lakers captent en moyenne {diff:.2f} rebonds de plus à domicile.")
    elif diff < 0:
        print(f"\n⬅️ Les Lakers captent en moyenne {-diff:.2f} rebonds de plus à l’extérieur.")
    else:
        print("\n⚖️ Les Lakers ont exactement la même moyenne de rebonds à domicile et à l’extérieur.")

if __name__ == "__main__":
    compare_lakers_rebounds()

