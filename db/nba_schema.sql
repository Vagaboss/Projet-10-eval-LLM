-- ============================================
-- 🏀 NBA Database Schema
-- Compatible avec les données issues du fichier Excel (regular NBA.xlsx)
-- ============================================

PRAGMA foreign_keys = ON;

-- ======================
-- 1️⃣ Table des joueurs
-- ======================
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,          -- Nom du joueur
    team TEXT,                   -- Code à 3 lettres de l'équipe
    age INTEGER,                 -- Âge du joueur
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ======================
-- 2️⃣ Table des statistiques agrégées par joueur
-- ======================
CREATE TABLE IF NOT EXISTS stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,  -- Référence vers players
    gp INTEGER,                  -- Games Played (nombre de matchs joués)
    w INTEGER,                   -- Nombre de victoires
    l INTEGER,                   -- Nombre de défaites
    min FLOAT,                   -- Minutes jouées par match
    pts FLOAT,                   -- Points par match
    fgm FLOAT,                   -- Field Goals Made
    fga FLOAT,                   -- Field Goals Attempted
    fg_percent FLOAT,            -- Pourcentage de réussite aux tirs (FG%)
    three_pa FLOAT,              -- 3 Points Attempted
    three_p_percent FLOAT,       -- Pourcentage à 3 points (3P%)
    ftm FLOAT,                   -- Free Throws Made
    fta FLOAT,                   -- Free Throws Attempted
    ft_percent FLOAT,            -- Pourcentage aux lancers francs (FT%)
    oreb FLOAT,                  -- Rebonds offensifs
    dreb FLOAT,                  -- Rebonds défensifs
    reb FLOAT,                   -- Rebonds totaux
    ast FLOAT,                   -- Passes décisives
    tov FLOAT,                   -- Balles perdues
    stl FLOAT,                   -- Interceptions
    blk FLOAT,                   -- Contres
    pf FLOAT,                    -- Fautes personnelles
    fp FLOAT,                    -- Fantasy Points
    dd2 INTEGER,                 -- Double-Doubles
    td3 INTEGER,                 -- Triple-Doubles
    plus_minus FLOAT,            -- +/- (écart de score)
    offrtg FLOAT,                -- Offensive Rating
    defrtg FLOAT,                -- Defensive Rating
    netrtg FLOAT,                -- Net Rating
    ast_percent FLOAT,           -- AST%
    ast_to_ratio FLOAT,          -- AST/TO
    ast_ratio FLOAT,             -- AST RATIO
    oreb_percent FLOAT,          -- OREB%
    dreb_percent FLOAT,          -- DREB%
    reb_percent FLOAT,           -- REB%
    to_ratio FLOAT,              -- Turnover Ratio
    efg_percent FLOAT,           -- Effective Field Goal %
    ts_percent FLOAT,            -- True Shooting %
    usg_percent FLOAT,           -- Usage Rate %
    pace FLOAT,                  -- PACE (rythme de jeu)
    pie FLOAT,                   -- Player Impact Estimate
    poss FLOAT,                  -- Possessions totales
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);

-- ======================
-- 3️⃣ Table des matchs (préparée pour la future extension)
-- ======================
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE,
    season TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER,
    venue TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ======================
-- 4️⃣ Table des rapports textuels (générés ou saisis)
-- ======================
CREATE TABLE IF NOT EXISTS reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    summary TEXT,                      -- Contenu du rapport textuel
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
);

-- ======================
-- ✅ Index et optimisations
-- ======================
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team);
CREATE INDEX IF NOT EXISTS idx_stats_player_id ON stats(player_id);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_reports_match_id ON reports(match_id);
