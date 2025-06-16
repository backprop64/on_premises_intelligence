from pathlib import Path
from typing import Sequence, Union, Optional, TYPE_CHECKING, List, Tuple, Iterable

import numpy as np
import random
import config  # Import configuration

# Only import the heavy class for type checking to avoid runtime dependency
if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

# For PDF reading & smart chunking
try:
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore
except ImportError:
    PyPDFLoader = None  # type: ignore

# Fallback simple pdf text extraction via pypdf
try:
    import pypdf
except ImportError:
    pypdf = None  # type: ignore

# Add tokenizer import for token-based chunking
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore


# Use config values for chunking - no more hardcoded values!
# CHUNK_SIZE = 300  # tokens (was 512 characters) - NOW FROM CONFIG
# CHUNK_OVERLAP = 60  # tokens (was 50 characters) - NOW FROM CONFIG


class TextEmbedder:
    """Wrapper around the *all-MiniLM-L6-v2* sentence-transformer model.

    Falls back to zero-vectors when *sentence_transformers* isn't installed so
    the rest of the codebase can still execute. Now uses token-based chunking
    for more precise text segmentation.
    """

    def __init__(self, model_path: Optional[str | Path] = None, device: str = "cpu") -> None:
        self.model_path: Path = Path(model_path) if model_path else Path(config.EMBEDDING_MODEL)
        self.device: str = device
        self._model: Optional["SentenceTransformer"] = None
        self.tokenizer = None  # Will be initialized in load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Instantiate the underlying sentence-transformer and tokenizer (if available)."""
        try:
            from sentence_transformers import SentenceTransformer as _ST  # local import
        except ImportError:
            print("[TextEmbedder] sentence_transformers not installed – using zeros.")
            return

        if self._model is None:
            self._model = _ST(str(self.model_path), device=self.device)
        
        # Initialize tokenizer for token-based chunking
        if AutoTokenizer is not None:
            try:
                # Use the embedding model from config
                model_name = config.EMBEDDING_MODEL
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"[TextEmbedder] Loaded tokenizer for token-based chunking (config: {config.CHUNK_SIZE} tokens, {config.CHUNK_OVERLAP} overlap)")
            except Exception as e:
                print(f"[TextEmbedder] Could not load tokenizer: {e}. Falling back to character-based chunking.")
                self.tokenizer = None
        else:
            print("[TextEmbedder] transformers not available – using character-based chunking.")

    def embed(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """Return dense embeddings for *texts* as `np.ndarray`."""
        if isinstance(texts, str):
            texts = [texts]

        if self._model is None:
            # Ensure deterministic shape using config dimension.
            return np.zeros((len(texts), config.EMBEDDING_DIMENSION), dtype=np.float32)

        return self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str, *, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Chunk text into segments using token-based approach for better semantic boundaries."""
        # Use config values if not provided
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        if overlap is None:
            overlap = config.CHUNK_OVERLAP
            
        print(f"[TextEmbedder] Using chunk_size={chunk_size}, overlap={overlap} from config")
        
        if self.tokenizer is None:
            # Fallback to character-based chunking if tokenizer not available
            print("[TextEmbedder] No tokenizer available, using character-based chunking")
            chunks: List[str] = []
            start = 0
            while start < len(text):
                end = start + chunk_size * 4  # Rough estimate: 1 token ≈ 4 characters
                chunks.append(text[start:end])
                start += (chunk_size - overlap) * 4
            return chunks
        
        # Token-based chunking
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunks = []
            start = 0
            
            while start < len(tokens):
                end = start + chunk_size
                chunk_tokens = tokens[start:end]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(chunk_text)
                start += chunk_size - overlap
            
            print(f"[TextEmbedder] Created {len(chunks)} chunks from {len(tokens)} tokens")
            return chunks
            
        except Exception as e:
            print(f"[TextEmbedder] Error in token-based chunking: {e}. Falling back to character-based.")
            # Fallback to character-based chunking
            chunks: List[str] = []
            start = 0
            while start < len(text):
                end = start + chunk_size * 4  # Rough estimate
                chunks.append(text[start:end])
                start += (chunk_size - overlap) * 4
            return chunks

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        # Prefer lightweight pypdf to avoid multiprocessing overhead in LangChain
        if pypdf is not None:
            reader = pypdf.PdfReader(str(pdf_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if PyPDFLoader is not None:
            pages = PyPDFLoader(str(pdf_path)).load()
            return "\n".join(p.page_content for p in pages)
        raise RuntimeError("Install 'pypdf' or 'langchain_community' to extract PDF text")

    def embed_file(self, file_path: Union[str, Path]) -> Tuple[List[str], np.ndarray]:
        """Return (chunks, embeddings) for a text or PDF file."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._extract_pdf_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        chunks = self._chunk_text(text)
        embeddings = self.embed(chunks)
        return chunks, embeddings

    # ------------------------------------------------------------------
    # CLI helper
    # ------------------------------------------------------------------
    def _cli(self) -> None:  # pragma: no cover
        """Quick sanity check when invoked directly."""
        self.load()
        vec = self.embed(["Hello world", "How are you?"])
        print("Embeddings shape:", vec.shape)

        # file test
        pdf_path = Path("test_text.pdf")
        if pdf_path.exists():
            chunks, embs = self.embed_file(pdf_path)
            print(f"Total chunks from PDF: {len(chunks)} | embedding shape {embs.shape}")
            for idx, chunk in enumerate(random.sample(chunks, min(3, len(chunks))), 1):
                print(f"\n--- Chunk {idx} ---\n{chunk}\n")


if __name__ == "__main__":  # pragma: no cover
    TextEmbedder()._cli() 