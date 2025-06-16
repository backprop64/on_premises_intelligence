from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "db_storage" / "metadata.db"


class MetadataStore:
    """Thread-safe wrapper around a SQLite metadata database."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path: Path = Path(db_path)
        self._local = threading.local()  # Thread-local storage for connections

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local.conn = sqlite3.connect(self.db_path)
            self._create_tables(self._local.conn)
        return self._local.conn

    def connect(self) -> None:
        """Initialize connection (for compatibility)."""
        self._get_connection()

    def close(self) -> None:
        """Close thread-local connection."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY,
                chunk TEXT,
                source TEXT,
                filename TEXT
            )
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # CRUD operations (minimal)
    # ------------------------------------------------------------------
    def add_record(self, *, id_: int, chunk: str, source: str, filename: str) -> None:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO metadata (id, chunk, source, filename) VALUES (?,?,?,?)",
            (id_, chunk, source, filename),
        )
        conn.commit()

    def get_record(self, id_: int) -> Dict[str, Any]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM metadata WHERE id = ?", (id_,))
        row = cur.fetchone()
        if row is None:
            return {}
        columns = [description[0] for description in cur.description]
        return dict(zip(columns, row))

    # ------------------------------------------------------------------
    # CLI helper
    # ------------------------------------------------------------------
    def _cli(self):  # pragma: no cover
        self.connect()
        self.add_record(id_=1, chunk="hello", source="test", filename="t.txt")
        print("Record:", self.get_record(1))
        self.close()


if __name__ == "__main__":  # pragma: no cover
    MetadataStore()._cli() 