# utils/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List


class Document(BaseModel):
    """Représente un document brut chargé depuis les fichiers."""
    page_content: str = Field(..., description="Texte brut extrait du document.")
    metadata: Dict[str, Any] = Field(..., description="Métadonnées du document (ex: source, nom de fichier).")

    @validator("page_content")
    def check_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Le contenu du document est vide.")
        return v


class DocumentChunk(BaseModel):
    """Représente un fragment de texte (chunk) prêt à être indexé."""
    id: str
    text: str = Field(..., min_length=10, description="Texte du chunk à indexer.")
    metadata: Dict[str, Any]

    @validator("text")
    def check_text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Chunk vide détecté.")
        print(f"✅ Chunk validé : {v[:50]}...")
        return v


class SearchResult(BaseModel):
    """Représente un résultat de recherche renvoyé par FAISS."""
    score: float
    text: str
    metadata: Dict[str, Any]


class RAGRequest(BaseModel):
    """Structure pour valider une question utilisateur."""
    question: str = Field(..., min_length=5, description="Question posée à l'assistant.")


class RAGResponse(BaseModel):
    """Structure pour valider la réponse générée."""
    answer: str
    context: List[SearchResult]
