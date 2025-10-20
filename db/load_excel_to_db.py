"""
load_excel_to_db.py — Pipeline d’ingestion du fichier Excel vers PostgreSQL avec validation Pydantic avancée
"""

import pandas as pd
import psycopg2
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
import logging
import datetime
import numpy as np

# --- CONFIGURATION ---
DB_CONFIG = {
    "dbname": "nba_db",
    "user": "postgres",
    "password": "naruto",  # mot de passe PostgreSQL
    "host": "localhost",
    "port": "5432"
}

EXCEL_FILE = "inputs/regular NBA.xlsx"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- 📘 SCHÉMAS PYDANTIC AVANCÉS ---
class TeamModel(BaseModel):
    team_code: str = Field(..., min_length=2, max_length=10)
    team_name: str

    @field_validator("team_code")
    def uppercase_code(cls, v):
        return v.strip().upper()


class PlayerModel(BaseModel):
    name: str
    team_code: str
    age: int | None = None

    @field_validator("name")
    def clean_name(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Le nom du joueur est vide ou invalide.")
        return v.strip()

    @field_validator("age")
    def check_age(cls, v):
        if v is not None and (v < 15 or v > 50):
            raise ValueError("Âge du joueur hors bornes logiques (15-50).")
        return v


class StatModel(BaseModel):
    player_name: str
    games_played: int | None = None
    points: float | None = None
    assists: float | None = None
    rebounds: float | None = None
    fg_pct: float | None = None
    three_pct: float | None = None
    ft_pct: float | None = None
    steals: float | None = None
    blocks: float | None = None
    turnovers: float | None = None
    off_rating: float | None = None
    def_rating: float | None = None
    net_rating: float | None = None
    pace: float | None = None
    pie: float | None = None

    @field_validator("fg_pct", "three_pct", "ft_pct")
    def pct_between_0_and_100(cls, v, info):
        if v is not None and not (0 <= v <= 100):
            raise ValueError(f"{info.field_name} doit être compris entre 0 et 100.")
        return v

    @model_validator(mode="after")
    def sanitize_values(self):
        """Convertit les valeurs aberrantes (NaN, infinies) en None"""
        for attr, value in self.__dict__.items():
            if isinstance(value, (int, float)):
                if pd.isna(value) or value in [float("inf"), float("-inf")]:
                    setattr(self, attr, None)
        return self



# --- 🔌 CONNEXION À POSTGRES ---
def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# --- 📥 CHARGEMENT EXCEL ---
def load_excel_data():
    logging.info(f"Lecture du fichier Excel : {EXCEL_FILE}")
    xls = pd.ExcelFile(EXCEL_FILE)
    logging.info(f"Feuilles détectées : {xls.sheet_names}")

    # Lecture et nettoyage
    df_players = pd.read_excel(xls, sheet_name="Données NBA", header=1)
    df_players = df_players.rename(columns={datetime.time(15, 0): "15:00"})
    df_players = df_players.loc[:, ~df_players.columns.str.contains('^Unnamed')]

    # Nettoyage des NaN → None
    df_players = df_players.replace({np.nan: None})

    try:
        df_teams = pd.read_excel(xls, sheet_name="Equipe")
        df_teams = df_teams.dropna(subset=["Code"])
        df_teams = df_teams.replace({np.nan: None})
    except Exception as e:
        logging.warning(f"Impossible de lire la feuille 'Equipe' : {e}")
        df_teams = pd.DataFrame()

    return df_teams, df_players


# --- 🧹 VALIDATION + INSERTION ---
def validate_and_insert():
    conn = get_connection()
    cur = conn.cursor()

    df_teams, df_players = load_excel_data()

    # === TEAMS ===
    if not df_teams.empty:
        logging.info(f"Insertion de {len(df_teams)} équipes...")
        for _, row in df_teams.iterrows():
            try:
                team = TeamModel(
                    team_code=row["Code"],
                    team_name=row["Nom complet de l'équipe"]
                )
                cur.execute("""
                    INSERT INTO teams (team_code, team_name)
                    VALUES (%s, %s)
                    ON CONFLICT (team_code) DO NOTHING;
                """, (team.team_code, team.team_name))
            except ValidationError as e:
                logging.warning(f"⚠️ Validation échouée pour équipe {row.get('Code')} : {e}")

    # === PLAYERS + STATS ===
    logging.info(f"Insertion de {len(df_players)} joueurs et statistiques...")
    for _, row in df_players.iterrows():
        try:
            player = PlayerModel(
                name=row["Player"],
                team_code=row["Team"],
                age=row["Age"]
            )

            cur.execute("""
                INSERT INTO players (name, team_code, age)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, (player.name, player.team_code, player.age))

            # Validation des stats
            stat = StatModel(
                player_name=row["Player"],
                games_played=row["GP"],
                points=row["PTS"],
                assists=row["AST"],
                rebounds=row["REB"],
                fg_pct=row["FG%"],
                three_pct=row["3P%"],
                ft_pct=row["FT%"],
                steals=row["STL"],
                blocks=row["BLK"],
                turnovers=row["TOV"],
                off_rating=row["OFFRTG"],
                def_rating=row["DEFRTG"],
                net_rating=row["NETRTG"],
                pace=row["PACE"],
                pie=row["PIE"]
            )

            cur.execute("""
                INSERT INTO stats (
                    player_id, games_played, points, assists, rebounds,
                    fg_pct, three_pct, ft_pct, steals, blocks, turnovers,
                    offensive_rating, defensive_rating, net_rating, pace, pie
                )
                SELECT player_id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                FROM players WHERE name = %s;
            """, (
                stat.games_played, stat.points, stat.assists, stat.rebounds,
                stat.fg_pct, stat.three_pct, stat.ft_pct, stat.steals,
                stat.blocks, stat.turnovers, stat.off_rating, stat.def_rating,
                stat.net_rating, stat.pace, stat.pie, stat.player_name
            ))

        except ValidationError as e:
            logging.warning(f"⚠️ Validation échouée pour joueur {row.get('Player')} : {e}")

    conn.commit()
    cur.close()
    conn.close()
    logging.info("✅ Données Excel insérées avec succès dans PostgreSQL !")


# --- MAIN ---
if __name__ == "__main__":
    validate_and_insert()



