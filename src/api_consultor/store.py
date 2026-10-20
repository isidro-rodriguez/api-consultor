"""Almacén SQLite: búsqueda vectorial (sqlite-vec) y por nombre exacto."""

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import sqlite_vec


@dataclass(frozen=True, slots=True)
class Chunk:
    qualname: str
    package: str
    version: str
    content: str


@dataclass(frozen=True, slots=True)
class SourceInfo:
    package: str
    version: str
    chunks: int


_COLUMNS = "qualname, package, version, content"


class Store:
    def __init__(self, path: Path, dim: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)
        self._db.executescript(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                qualname TEXT NOT NULL,
                package TEXT NOT NULL,
                version TEXT NOT NULL,
                content TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_package ON chunks(package);
            CREATE INDEX IF NOT EXISTS idx_chunks_qualname ON chunks(qualname);
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
                USING vec0(embedding float[{dim}]);
        """)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._db.close()

    def replace(
        self,
        package: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Sustituye todo lo indexado de `package` (reindexado idempotente)."""
        with self._db:  # una sola transacción
            old_ids = [
                (row[0],)
                for row in self._db.execute(
                    "SELECT id FROM chunks WHERE package = ?", (package,)
                )
            ]
            self._db.executemany("DELETE FROM vec_chunks WHERE rowid = ?", old_ids)
            self._db.execute("DELETE FROM chunks WHERE package = ?", (package,))
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                cur = self._db.execute(
                    f"INSERT INTO chunks({_COLUMNS}) VALUES (?, ?, ?, ?)",
                    (chunk.qualname, chunk.package, chunk.version, chunk.content),
                )
                self._db.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, sqlite_vec.serialize_float32(list(embedding))),
                )

    def sources(self) -> list[SourceInfo]:
        rows = self._db.execute(
            "SELECT package, version, COUNT(*) FROM chunks "
            "GROUP BY package, version ORDER BY package"
        )
        return [SourceInfo(*row) for row in rows]

    def qualnames(self) -> list[str]:
        return [row[0] for row in self._db.execute("SELECT qualname FROM chunks")]

    def by_name(self, names: Iterable[str], limit_per_name: int = 2) -> list[Chunk]:
        """Coincidencia exacta con el nombre cualificado o con su sufijo."""
        found: list[Chunk] = []
        for name in names:
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM chunks "
                "WHERE qualname = ? OR qualname GLOB ? LIMIT ?",
                (name, f"*.{name}", limit_per_name),
            )
            found.extend(Chunk(*row) for row in rows)
        return found

    def search(self, embedding: Sequence[float], k: int) -> list[Chunk]:
        rows = self._db.execute(
            f"""
            SELECT {", ".join("c." + col.strip() for col in _COLUMNS.split(","))}
            FROM (
                SELECT rowid, distance FROM vec_chunks
                WHERE embedding MATCH ? AND k = ?
            ) AS v
            JOIN chunks AS c ON c.id = v.rowid
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(list(embedding)), k),
        )
        return [Chunk(*row) for row in rows]
