"""Part 5: turn the handbook into a searchable vector store.

    python -m part05_rag.ingest            # writes data/handbook_index.json
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
HANDBOOK = ROOT / "data" / "handbook.md"
INDEX = ROOT / "data" / "handbook_index.json"


def get_embeddings():
    model = os.getenv("EMBEDDING_MODEL", "openai:text-embedding-3-small")
    # OpenAI's client normally downloads a tokenizer file to split over-long inputs. Our chunks are
    # small, so skip it: one less download, and it avoids certificate errors behind corporate proxies.
    extra = {"check_embedding_ctx_length": False} if model.startswith("openai:") else {}
    return init_embeddings(model, **extra)


def split_handbook(text: str):
    # 1. Split on headings first, so every chunk knows which section it came from.
    by_heading = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "doc"), ("##", "section"), ("###", "subsection")],
    ).split_text(text)
    # 2. Then cap the size, with a small overlap so sentences are not cut in half.
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(by_heading)
    for chunk in chunks:
        chunk.metadata["source"] = chunk.metadata.get("subsection") or chunk.metadata.get("section", "Handbook")
    return chunks


def build_index(embeddings=None) -> InMemoryVectorStore:
    chunks = split_handbook(HANDBOOK.read_text(encoding="utf-8"))
    store = InMemoryVectorStore(embeddings or get_embeddings())
    store.add_documents(chunks)
    return store


if __name__ == "__main__":
    store = build_index()
    store.dump(str(INDEX))
    print(f"Indexed {len(store.store)} chunks into {INDEX.name}")
