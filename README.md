# Assistant RAG avec Mistral

- Lien Github : https://github.com/Vagaboss/Projet-10-eval-LLM

Ce projet vise à évaluer la performance d’un système RAG combinant recherche textuelle (corpus Reddit NBA) et exploitation chiffrée (base SQL issue du fichier Excel NBA). L’objectif est d’analyser la robustesse du pipeline, sa capacité à fusionner texte et données numériques, et à fournir des visualisations dynamiques pour appuyer l’analyse.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- 🤖 **Génération de réponses** avec les modèles Mistral (Small ou Large)
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.9+ 
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/Vagaboss/Projet-10-eval-LLM.git
cd Projet-10-eval-LLM
```

2. **Créer un environnement virtuel**

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement virtuel
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

## Structure du projet

```
.
├── MistralChat.py          # Application Streamlit principale
├── indexer.py              # Script pour indexer les documents
├── inputs/                 # Dossier pour les documents sources
├── vector_db/              # Dossier pour l'index FAISS et les chunks
├── db/                     # Base de données SQLite pour les interactions
└── utils/                  # Modules utilitaires
    ├── config.py           # Configuration de l'application
    ├── data_loader_.py     
    └── vector_store.py     # Gestion de l'index vectoriel

```
## Schéma architecture 
      [Sources initiales : Reddit PDF + Excel NBA]
                             │
                             ▼
             ┌─────────────────────────────────┐
             │   Étape 1 : Indexation RAG (FAISS) │
             │ - Extraction texte (data_loader)   │
             │ - Découpage et embeddings Mistral  │
             │ - Stockage FAISS + pickle          │
             └─────────────────────────────────┘
                             │
                             ▼
             ┌─────────────────────────────────┐
             │ Étape 2 : Création base SQL (PostgreSQL) │
             │ - Table players / stats / teams           │
             │ - Alimentation via Excel NBA              │
             └─────────────────────────────────┘
                             │
                             ▼
             ┌─────────────────────────────────┐
             │ Étape 3 : Application MistralChat │
             │ - Détection automatique (SQL / RAG) │
             │ - Fusion intelligente (SQL + texte) │
             │ - Génération de graphiques (PlotTool) │
             └─────────────────────────────────┘
                             │
                             ▼
             ┌─────────────────────────────────┐
             │ Étape 4 : Évaluation & Monitoring │
             │ - RAGAS (faithfulness, precision) │
             │ - Logfire (suivi des requêtes)   │
             └─────────────────────────────────┘

## Utilisation

### 1. Ajouter des documents

Placez vos documents dans le dossier `inputs/`. Les formats supportés sont :
- PDF
- TXT
- DOCX
- CSV
- JSON

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
python indexer.py
```

Ce script va :
1. Charger les documents depuis le dossier `inputs/`
2. Découper les documents en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

### 3. Lancer l'application

```bash
streamlit run MistralChat.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.


## Modules principaux

### `utils/vector_store.py`
Gère la base vectorielle FAISS et la recherche sémantique :  
- Découpage des documents en *chunks*  
- Génération des embeddings Mistral  
- Création et interrogation de l’index FAISS  
- Sauvegarde et chargement des données (`faiss_index.idx`, `document_chunks.pkl`)

### `utils/data_loader.py`
Responsable de l’extraction du texte à partir de fichiers bruts :  
- Supporte PDF, DOCX, TXT, CSV, Excel  
- Gère les fichiers scannés via **EasyOCR**  
- Retourne une liste normalisée de documents exploitables pour l’indexation 

### `utils/config.py`
Centralise tous les paramètres du projet :  
- Chargement du `.env`  
- Chemins (`inputs/`, `vector_db/`)  
- Paramètres techniques (taille de chunk, chevauchement, etc.)  
- Modèles utilisés (`mistral-small-latest`, `mistral-embed`)

---

## 🧠 Architecture du système


[Documents bruts]
      │

      ▼
 data_loader.py    →  extraction et parsing du texte
      │

      ▼
 indexer.py        →  orchestre l’indexation (embeddings + FAISS)
      │

      ▼
 vector_store.py   →  construit et interroge la base vectorielle
      │
      
      ▼
 MistralChat.py    →  interface Streamlit (RAG + génération de réponse)



## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- Nombre de documents par défaut
- Nom de la commune ou organisation

## 📊 Visualisation automatique (PlotTool)

Le PlotTool permet de tracer automatiquement des graphiques à partir de résultats SQL.
Types de graphiques gérés :

Diagramme en barres

Camembert

Histogramme

Courbe de tendance

Chaque graphique est sauvegardé dans un fichier temporaire (plot.png) et affiché dans Streamlit.

## ✅ Procédure de validation

	Vérification           =>	  Résultat attendu

- 1	Lancer python indexer.py	=> faiss_index.idx et document_chunks.pkl générés

- 2	Lancer PostgreSQL (nba_db) =>	Tables players, stats, teams créées

- 3	Exécuter streamlit run MistralChat.py => L’interface s’affiche

- 4	Tester une requête textuelle =>	Résultat issu du corpus Reddit

- 5	Tester une requête SQL =>	Tableau de résultats affiché

- 6	Tester une requête mixte (texte + chiffres) =>	Double réponse SQL         contexte textuel

- 7	Tester une requête graphique =>	Affichage d’un graphique dynamique


## 🔬 Check-list de validation finale

✅ Vérification	        =>         Résultat attendu

1️⃣	Exécution de indexer.py	=>     faiss_index.idx et document_chunks.pkl générés

2️⃣	Lancement de PostgreSQL	 =>    Base nba_db opérationnelle

3️⃣	Démarrage de MistralChat.py =>	Interface Streamlit fonctionnelle

4️⃣	Requête textuelle	=>          Réponse cohérente issue du corpus Reddit

5️⃣	Requête SQL	      =>         Résultat chiffré affiché sous forme de tableau

6️⃣	Requête mixte (SQL + texte)	=> Fusion correcte des deux sources

7️⃣	Requête avec graphique	=>     Image générée et affichée dans Streamlit

8️⃣	Évaluation RAGAS	  =>        Fichier results_rag_sql.json créé

9️⃣	Suivi Logfire	       =>        Journaux des requêtes visibles en temps réel

🔟	Test de robustesse	   =>      Réponses stables aux requêtes mixtes simples
