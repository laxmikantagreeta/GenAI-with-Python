"""High-level orchestration for building and querying a documentation index."""

from pathlib import Path
from typing import List

import google.generativeai as genai

from .config import load_api_key
from .loader import chunk_documents, load_documents
from .vector_store import VectorStore, embed_query, embed_with_gemini


PROMPT_TEMPLATE = """You are a helpful tutor. Use the provided documentation snippets to answer the question.
Cite filenames when relevant and keep the explanation concise and friendly.

Question: {question}

Context:
{context}

Answer:
"""


def build_index(
    source_dir: str | Path,
    index_path: str | Path,
    chunk_size: int = 800,
    overlap: int = 200,
) -> Path:
    """Load docs, create embeddings, and persist a vector index."""

    documents = load_documents(source_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    api_key = load_api_key()
    embedder = embed_with_gemini(api_key)
    store = VectorStore.from_chunks(chunks, embedder)
    store.save(index_path)
    return Path(index_path)


def answer_question(question: str, index_path: str | Path, top_k: int = 4) -> str:
    """Answer a learner's question using the stored index and Gemini generation."""

    api_key = load_api_key()
    store = VectorStore.load(index_path)
    query_embedding = embed_query(api_key, question)
    top_chunks = store.similarity_search(query_embedding, top_k=top_k)

    context_blocks: List[str] = []
    for chunk in top_chunks:
        context_blocks.append(f"Source: {chunk.source} (section {chunk.index})\n{chunk.text}\n")

    prompt = PROMPT_TEMPLATE.format(question=question, context="\n".join(context_blocks))
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
