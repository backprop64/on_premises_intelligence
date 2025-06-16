from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import config

# FAISS is optional for the stub stage so we import defensively
try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore

DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[2] / "db_storage" / "faiss.index"


class VectorIndex:
    """Minimal FAISS index wrapper used for similarity search."""

    def __init__(self, dim: int = None, index_path: str | Path = DEFAULT_INDEX_PATH):
        self.dim: int = dim if dim is not None else config.EMBEDDING_DIMENSION
        self.index_path: Path = Path(index_path)
        self.index = None  # lazy initialised faiss.IndexFlatIP

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def build_empty(self) -> None:
        if faiss is None:
            return
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))

    def load(self) -> None:
        if faiss is None:
            return
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.build_empty()

    def save(self) -> None:
        if faiss is None or self.index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_vectors(self, vectors: np.ndarray) -> List[int]:
        if faiss is None:
            return []
        if self.index is None:
            self.build_empty()
        ids = list(range(self.index.ntotal, self.index.ntotal + len(vectors)))
        self.index.add_with_ids(vectors.astype("float32"), np.asarray(ids))
        return ids

    def search(self, query: np.ndarray, *, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if faiss is None or self.index is None:
            return np.array([]), np.array([])

        # FAISS versions on macOS have been observed to crash when a search is
        # performed on an empty index (ntotal == 0). We defensively short-circuit
        # to an empty result in that case.
        if self.index.ntotal == 0:
            return np.zeros((1, 0), dtype="float32"), np.zeros((1, 0), dtype="int64")
        return self.index.search(query.astype("float32"), k)

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    def _cli(self) -> None:  # pragma: no cover
        import numpy.random as npr

        self.build_empty()
        vecs = npr.rand(10, self.dim).astype("float32")
        self.add_vectors(vecs)
        q = vecs[:1]
        scores, ids = self.search(q, k=3)
        print("Scores:", scores, "Ids:", ids)


if __name__ == "__main__":  # pragma: no cover
    VectorIndex()._cli() 