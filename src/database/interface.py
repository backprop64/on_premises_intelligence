from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from src.models import TextEmbedder, ImageCaptioner
from .vector_search import VectorIndex
from .sql import MetadataStore


# Lazy imports for optional dependencies – support multiple LangChain versions
# Newer versions moved modules to `langchain_community.document_loaders` and changed splitter name.
PyPDFLoader = None  # type: ignore
RecursiveCharacterTextSplitter = None  # type: ignore

try:
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from langchain.document_loaders import PyPDFLoader  # type: ignore
    except ImportError:
        pass

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
except ImportError:  # pragma: no cover
    RecursiveCharacterTextSplitter = None  # type: ignore

class IngestionInterface:
    """High-level helper that orchestrates file ingestion & storage."""

    def __init__(self) -> None:
        self.embedder = TextEmbedder()
        self.captioner: ImageCaptioner | None = None
        self.index = VectorIndex()
        self.store = MetadataStore()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load underlying components (stub)."""
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        self.embedder.load()
        if self.captioner is not None and hasattr(self.captioner, "load"):
            self.captioner.load()
        self.index.load()
        try:
            import faiss
            faiss.omp_set_num_threads(1)
        except ImportError:
            pass
        self.store.connect()

    def close(self) -> None:
        self.store.close()
        self.index.save()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_file(self, file_path: str | Path) -> None:
        """Process *file_path* and push data to FAISS + SQLite.

        The method supports:
        •  PDFs – loaded via *langchain* then chunked with overlap.
        •  Plain-text files – chunked naively / via text splitter.
        •  Images – captioned with BLIP2 wrapper and stored as a single chunk.
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()

        # Text-like files
        if suffix in {".pdf", ".txt", ".md", ".log"}:
            chunks, embeddings = self.embedder.embed_file(path)
            
            # Show token-aware logging
            total_chars = sum(len(chunk) for chunk in chunks)
            try:
                if hasattr(self.embedder, 'tokenizer') and self.embedder.tokenizer:
                    total_tokens = sum(len(self.embedder.tokenizer.encode(chunk, add_special_tokens=False)) for chunk in chunks)
                    print(f"[Ingest] {path.name}: {len(chunks)} chunks, {total_chars} chars, ~{total_tokens} tokens, embeddings shape {embeddings.shape}")
                else:
                    print(f"[Ingest] {path.name}: {len(chunks)} chunks, {total_chars} chars (tokenizer unavailable), embeddings shape {embeddings.shape}")
            except:
                print(f"[Ingest] {path.name}: {len(chunks)} chunks, {total_chars} chars (token count failed), embeddings shape {embeddings.shape}")
                
            ids = self.index.add_vectors(embeddings)
            for _id, chunk in zip(ids, chunks):
                self.store.add_record(id_=_id, chunk=chunk, source=str(path), filename=path.name)
            self.index.save()
            return

        # Image files
        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}:
            if self.captioner is None:
                # Use configured vision model size to avoid loading the default 2.2B model on CPU
                try:
                    from config import VISION_MODEL_SIZE
                except ImportError:
                    VISION_MODEL_SIZE = "256m"  # Fallback
                self.captioner = ImageCaptioner(size=VISION_MODEL_SIZE)
            caption = self.captioner.caption(path).rstrip() + " this is an image file"
            chunks = [caption]
            embeddings = self.embedder.embed(caption)
            print(f"[Ingest] {path.name}: 1 image caption embedded")
            ids = self.index.add_vectors(embeddings)
            for _id, chunk in zip(ids, chunks):
                self.store.add_record(id_=_id, chunk=chunk, source=str(path), filename=path.name)
            self.index.save()
            return

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # ------------------------------------------------------------------
    # CLI helper
    # ------------------------------------------------------------------
    def _cli(self, test_pdf: Path) -> None:  # pragma: no cover
        import random, textwrap

        self.load()

        print(f"\nIngesting {test_pdf}…")
        self.ingest_file(test_pdf)

        # Retrieve stored chunks to pick random query
        cur = self.store.conn.cursor()
        cur.execute("SELECT chunk FROM metadata WHERE filename = ?", (test_pdf.name,))
        all_chunks = [row[0] for row in cur.fetchall()]

        query = random.choice(all_chunks)
        print("\nQuery chunk:\n", textwrap.fill(query, 80), "\n")

        q_emb = self.embedder.embed(query)
        scores, ids = self.index.search(q_emb, k=5)

        print("Top-5 similar chunks:\n")
        for rank, (_score, _id) in enumerate(zip(scores[0], ids[0]), 1):
            rec = self.store.get_record(int(_id))
            chunk_txt = textwrap.fill(rec.get("chunk", ""), 80)
            print(f"[{rank}] score={_score:.4f}\n{chunk_txt}\n")

        self.close()

if __name__ == "__main__":  # pragma: no cover
    root = Path(__file__).resolve().parents[2]
    ingest = IngestionInterface()
    ingest._cli(root / "test_text.pdf")
 