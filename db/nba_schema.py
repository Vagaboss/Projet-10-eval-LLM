"""
nba_schema.py — Création du schéma relationnel NBA pour PostgreSQL
"""

import psycopg2
import logging
from psycopg2 import sql

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration de la base de données ---
DB_CONFIG = {
    "dbname": "nba_db",
    "user": "postgres",
    "password": "naruto",  # Remplace par ton mot de passe PostgreSQL
    "host": "localhost",
    "port": "5432"
}

# --- Script principal ---
def create_nba_schema():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        logging.info("Connexion à PostgreSQL réussie ✅")
    except Exception as e:
        logging.error(f"Erreur de connexion à PostgreSQL : {e}")
        raise

    # --- Suppression des tables existantes ---
    logging.info("Suppression des tables existantes (si présentes)...")
    cursor.execute("""
        DROP TABLE IF EXISTS player_match_stats CASCADE;
        DROP TABLE IF EXISTS matches CASCADE;
        DROP TABLE IF EXISTS stats CASCADE;
        DROP TABLE IF EXISTS players CASCADE;
        DROP TABLE IF EXISTS teams CASCADE;
        DROP TABLE IF EXISTS reports CASCADE;
    """)

    # --- Table TEAMS ---
    cursor.execute("""
        CREATE TABLE teams (
            team_id SERIAL PRIMARY KEY,
            team_code VARCHAR(10) UNIQUE NOT NULL,
            team_name VARCHAR(100) NOT NULL,
            wins INTEGER,
            losses INTEGER,
            points_avg REAL,
            rebounds_avg REAL,
            assists_avg REAL,
            fg_pct REAL,
            three_pct REAL,
            ft_pct REAL,
            offensive_rating REAL,
            defensive_rating REAL,
            net_rating REAL
        );
    """)
    logging.info("✅ Table 'teams' créée.")

    # --- Table PLAYERS ---
    cursor.execute("""
        CREATE TABLE players (
            player_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            team_code VARCHAR(10),
            age INTEGER,
            position VARCHAR(20),
            FOREIGN KEY (team_code) REFERENCES teams (team_code) ON DELETE SET NULL
        );
    """)
    logging.info("✅ Table 'players' créée.")

    # --- Table STATS ---
    cursor.execute("""
        CREATE TABLE stats (
            stat_id SERIAL PRIMARY KEY,
            player_id INTEGER REFERENCES players(player_id) ON DELETE CASCADE,
            games_played INTEGER,
            minutes REAL,
            points REAL,
            rebounds REAL,
            assists REAL,
            steals REAL,
            blocks REAL,
            turnovers REAL,
            fg_pct REAL,
            three_pct REAL,
            ft_pct REAL,
            usage_pct REAL,
            pace REAL,
            pie REAL,
            offensive_rating REAL,
            defensive_rating REAL,
            net_rating REAL,
            plus_minus REAL,
            double_doubles INTEGER,
            triple_doubles INTEGER
        );
    """)
    logging.info("✅ Table 'stats' créée.")

    # --- Table MATCHES ---
    cursor.execute("""
        CREATE TABLE matches (
            match_id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            home_team_code VARCHAR(10) REFERENCES teams(team_code),
            away_team_code VARCHAR(10) REFERENCES teams(team_code),
            home_score INTEGER,
            away_score INTEGER,
            winner_team_code VARCHAR(10) REFERENCES teams(team_code)
        );
    """)
    logging.info("✅ Table 'matches' créée.")

    # --- Table PLAYER_MATCH_STATS ---
    cursor.execute("""
        CREATE TABLE player_match_stats (
            id SERIAL PRIMARY KEY,
            match_id INTEGER REFERENCES matches(match_id) ON DELETE CASCADE,
            player_id INTEGER REFERENCES players(player_id) ON DELETE CASCADE,
            points INTEGER,
            rebounds INTEGER,
            assists INTEGER,
            steals INTEGER,
            blocks INTEGER,
            fg_pct REAL,
            three_pct REAL,
            ft_pct REAL,
            minutes_played REAL
        );
    """)
    logging.info("✅ Table 'player_match_stats' créée.")

    # --- Table REPORTS ---
    cursor.execute("""
        CREATE TABLE reports (
            report_id SERIAL PRIMARY KEY,
            report_type VARCHAR(50),
            description TEXT,
            team_code VARCHAR(10),
            player_id INTEGER REFERENCES players(player_id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_code) REFERENCES teams (team_code) ON DELETE SET NULL
        );
    """)
    logging.info("✅ Table 'reports' créée.")

    # --- Commit & fermeture ---
    conn.commit()
    cursor.close()
    conn.close()
    logging.info("🎯 Base PostgreSQL NBA initialisée avec succès.")

# --- Exécution ---
if __name__ == "__main__":
    create_nba_schema()

