"""A minimal vector store backed by Gemini embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

import google.generativeai as genai
import numpy as np

from .loader import Chunk

EmbeddingFunction = Callable[[Iterable[str]], List[List[float]]]


@dataclass
class EmbeddedChunk:
    """A stored embedding with its source metadata."""

    source: str
    index: int
    text: str
    embedding: list[float]


class VectorStore:
    """Lightweight store to persist embeddings and run similarity search."""

    def __init__(self, items: list[EmbeddedChunk]):
        self.items = items
        self.matrix = np.array([item.embedding for item in items])

    @classmethod
    def from_chunks(
        cls, chunks: Iterable[Chunk], embedder: EmbeddingFunction
    ) -> "VectorStore":
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder(texts)
        items = [
            EmbeddedChunk(
                source=str(chunk.source), index=chunk.index, text=chunk.text, embedding=emb
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        return cls(items)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [item.__dict__ for item in self.items]
        path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [EmbeddedChunk(**item) for item in data]
        return cls(items)

    def similarity_search(self, query_embedding: list[float], top_k: int = 4) -> List[EmbeddedChunk]:
        normalized_matrix = self._normalize(self.matrix)
        normalized_query = self._normalize(np.array([query_embedding]))
        scores = normalized_matrix @ normalized_query.T
        sorted_indices = np.argsort(scores, axis=0)[::-1].flatten()
        top_indices = sorted_indices[:top_k]
        return [self.items[i] for i in top_indices]

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(array, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return array / norm


def embed_with_gemini(api_key: str, model: str = "models/text-embedding-004") -> EmbeddingFunction:
    """Create an embedding function configured with the provided API key."""

    genai.configure(api_key=api_key)

    def embedder(texts: Iterable[str]) -> List[List[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            response = genai.embed_content(model=model, content=text)
            embeddings.append(response["embedding"]["values"])
        return embeddings

    return embedder


def embed_query(api_key: str, query: str, model: str = "models/text-embedding-004") -> list[float]:
    """Embed a single query string using Gemini embeddings."""

    genai.configure(api_key=api_key)
    response = genai.embed_content(model=model, content=query)
    return response["embedding"]["values"]
