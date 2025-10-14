# indexer.py
import argparse
import logging
from typing import Optional
import logfire  # ✅ Import Logfire

from utils.config import INPUT_DIR
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.vector_store import VectorStoreManager

# --- Configuration du logging standard ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@logfire.instrument("Processus d'indexation complet")  # ✅ Trace principale
def run_indexing(input_directory: str, data_url: Optional[str] = None):
    """Exécute le processus complet d'indexation."""
    logging.info("--- Démarrage du processus d'indexation ---")

    # --- Étape 1: Téléchargement et extraction (optionnelle) ---
    with logfire.span("Téléchargement et extraction des données"):  # ✅ Trace Logfire
        if data_url:
            logging.info(f"Tentative de téléchargement depuis l'URL: {data_url}")
            success = download_and_extract_zip(data_url, input_directory)
            if not success:
                logging.error("Échec du téléchargement ou de l'extraction. Arrêt.")
                return
        else:
            logging.info(f"Aucune URL fournie. Utilisation des fichiers locaux dans: {input_directory}")

    # --- Étape 2: Chargement et parsing des fichiers ---
    with logfire.span("Chargement et parsing des fichiers"):  # ✅ Trace Logfire
        logging.info(f"Chargement et parsing des fichiers depuis: {input_directory}")
        documents = load_and_parse_files(input_directory)

        if not documents:
            logging.warning("Aucun document n'a été chargé ou parsé. Vérifiez le contenu du dossier d'entrée.")
            logging.info("--- Processus d'indexation terminé (aucun document traité) ---")
            return
        else:
            logging.info(f"{len(documents)} documents chargés avec succès.")

    # --- Étape 3: Création / mise à jour de l'index vectoriel ---
    with logfire.span("Création et sauvegarde de l'index vectoriel"):  # ✅ Trace Logfire
        logging.info("Initialisation du gestionnaire de Vector Store...")
        vector_store = VectorStoreManager()

        logging.info("Construction de l'index FAISS (cela peut prendre du temps)...")
        vector_store.build_index(documents)

        if vector_store.index:
            logging.info(f"Index FAISS créé avec {vector_store.index.ntotal} vecteurs.")
        else:
            logging.warning("L'index final n'a pas pu être créé ou est vide.")

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre de documents traités: {len(documents)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire contenant les fichiers sources (par défaut: {INPUT_DIR})"
    )
    parser.add_argument(
        "--data-url",
        type=str,
        default=None,
        help="URL optionnelle pour télécharger et extraire un fichier inputs.zip"
    )
    args = parser.parse_args()

    final_data_url = args.data_url
    run_indexing(input_directory=args.input_dir, data_url=final_data_url)
