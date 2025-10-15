# utils/vector_store.py
import logfire
import os
import pickle
import faiss
import numpy as np
import logging

from typing import List, Dict, Tuple, Optional
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # Utilisé pour le format attendu par le splitter
from utils.schemas import DocumentChunk
from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, any]] = []
        self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks si les fichiers existent."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}...")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Chargement des chunks depuis {DOCUMENT_CHUNKS_FILE}...")
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_documents_to_chunks(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Découpe les documents en chunks avec métadonnées."""
        logging.info(f"Découpage de {len(documents)} documents en chunks (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        all_chunks = []
        doc_counter = 0
        for doc in documents:
            langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            chunks = text_splitter.split_documents([langchain_doc])
            logging.info(f"  Document '{doc['metadata'].get('filename', 'N/A')}' découpé en {len(chunks)} chunks.")

            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "id": f"{doc_counter}_{i}",
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_id_in_doc": i,
                        "start_index": chunk.metadata.get("start_index", -1)
                    }
                }
                try:
                    validated_chunk = DocumentChunk(**chunk_dict)
                    all_chunks.append(validated_chunk.dict())
                except Exception as e:
                    logging.warning(
                        f"Chunk invalide ignoré ({chunk_dict['metadata'].get('source', 'inconnu')}) : {e}"
                    )

            doc_counter += 1

        logging.info(f"Total de {len(all_chunks)} chunks valides créés.")
        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, any]]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not MISTRAL_API_KEY:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})...")
        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
            try:
                response = self.mistral_client.embeddings(
                    model=EMBEDDING_MODEL,
                    input=texts_to_embed
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except MistralAPIException as e:
                logging.error(f"Erreur API Mistral lors de la génération d'embeddings (lot {batch_num}): {e}")
                logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
            except Exception as e:
                logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                else:
                    logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                    continue
                logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)

        if not all_embeddings:
            logging.error("Aucun embedding n'a pu être généré.")
            return None

        embeddings_array = np.array(all_embeddings).astype('float32')
        logging.info(f"Embeddings générés avec succès. Shape: {embeddings_array.shape}")
        return embeddings_array

    # 🧠 LOGFIRE : version corrigée
    @logfire.instrument("Construction complète de l'index FAISS")
    def build_index(self, documents: List[Dict[str, any]]):
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        # --- Étape 1 : Découpage des documents ---
        with logfire.span("Découpage des documents en chunks"):
            self.document_chunks = self._split_documents_to_chunks(documents)
            if not self.document_chunks:
                logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
                return
            else:
                logging.info(f"{len(self.document_chunks)} chunks créés à partir des documents.")
                logfire.info("✅ Chunks créés", extra={"nombre_chunks": len(self.document_chunks)})

        # --- Étape 2 : Génération des embeddings ---
        with logfire.span("Génération des embeddings via Mistral"):
            embeddings = self._generate_embeddings(self.document_chunks)

        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Problème de génération d'embeddings. Le nombre d'embeddings ne correspond pas au nombre de chunks.")
            self.document_chunks = []
            self.index = None
            if os.path.exists(FAISS_INDEX_FILE):
                os.remove(FAISS_INDEX_FILE)
            if os.path.exists(DOCUMENT_CHUNKS_FILE):
                os.remove(DOCUMENT_CHUNKS_FILE)
            logfire.error("❌ Erreur génération embeddings", extra={"nb_chunks": len(self.document_chunks)})
            return
        else:
            logfire.info("✅ Embeddings générés", extra={
                "dimension": embeddings.shape[1],
                "nb_embeddings": embeddings.shape[0]
            })

        # --- Étape 3 : Création de l’index FAISS ---
        with logfire.span("Création de l’index FAISS"):
            dimension = embeddings.shape[1]
            logging.info(f"Création de l'index Faiss optimisé pour la similarité cosinus avec dimension {dimension}...")

            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)

            logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")
            logfire.info("✅ Index FAISS créé", extra={
                "dimension": dimension,
                "nb_vecteurs": self.index.ntotal
            })

        # --- Étape 4 : Sauvegarde ---
        with logfire.span("Sauvegarde de l’index et des chunks"):
            self._save_index_and_chunks()
            logging.info("Index et chunks sauvegardés avec succès.")
            logfire.info("💾 Sauvegarde effectuée", extra={
                "fichier_index": FAISS_INDEX_FILE,
                "fichier_chunks": DOCUMENT_CHUNKS_FILE
            })

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks."""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[Dict[str, any]]:
        """Recherche les k chunks les plus pertinents pour une requête."""
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
            logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
            return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            response = self.mistral_client.embeddings(
                model=EMBEDDING_MODEL,
                input=[query_text]
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)

            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            results = []
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks):
                        chunk = self.document_chunks[idx]
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            continue

                        results.append({
                            "score": similarity,
                            "raw_score": raw_score,
                            "text": chunk["text"],
                            "metadata": chunk["metadata"]
                        })
                    else:
                        logging.warning(f"Index Faiss {idx} hors limites.")

            results.sort(key=lambda x: x["score"], reverse=True)
            if len(results) > k:
                results = results[:k]

            logging.info(f"{len(results)} chunks pertinents trouvés.")
            return results

        except MistralAPIException as e:
            logging.error(f"Erreur API Mistral lors de la génération de l'embedding : {e}")
            return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []
