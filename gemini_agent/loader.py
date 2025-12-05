"""Document loading and chunking utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Document:
    """A basic text document with a source path and content."""

    path: Path
    content: str


@dataclass
class Chunk:
    """A chunk of text derived from a document."""

    source: Path
    text: str
    index: int


def load_documents(root: str | Path, extensions: Iterable[str] = (".md", ".txt")) -> List[Document]:
    """Load documents from the given directory.

    Args:
        root: Directory containing documentation files.
        extensions: File extensions to include.

    Returns:
        A list of loaded ``Document`` instances.
    """

    root_path = Path(root)
    documents: list[Document] = []
    for ext in extensions:
        for path in root_path.rglob(f"*{ext}"):
            content = path.read_text(encoding="utf-8")
            documents.append(Document(path=path, content=content))
    return documents


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for retrieval.

    Args:
        text: Raw document text.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to repeat between consecutive chunks.

    Returns:
        A list of chunk strings.
    """

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap to avoid infinite loops")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def chunk_documents(documents: Iterable[Document], chunk_size: int = 800, overlap: int = 200) -> List[Chunk]:
    """Convert documents into chunks for embedding.

    Args:
        documents: Iterable of ``Document`` instances.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to repeat between chunks.

    Returns:
        A list of ``Chunk`` objects.
    """

    chunks: list[Chunk] = []
    for doc in documents:
        for index, text in enumerate(chunk_text(doc.content, chunk_size, overlap)):
            chunks.append(Chunk(source=doc.path, text=text, index=index))
    return chunks
