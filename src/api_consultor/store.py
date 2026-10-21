"""Almacén SQLite: vectores (sqlite-vec), palabras (FTS5), nombres y metadatos."""

import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import sqlite_vec

# 1: sin metadatos (tablas chunks/vec_chunks/FTS); 2: meta + sources; 3: path y line
FORMAT_VERSION = 3


class IndexIntegrityError(Exception):
    """El índice no es compatible con la configuración actual."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True, slots=True)
class Chunk:
    qualname: str
    package: str
    version: str
    content: str
    path: str = ""  # fichero de origen editable ("" si no existe: zip, builtin)
    line: int = 0  # línea del símbolo o sección en `path` (0 = desconocida)


@dataclass(frozen=True, slots=True)
class SourceInfo:
    package: str
    version: str
    chunks: int
    kind: str = "package"  # "package" (API de un paquete) o "docs" (documentación)
    indexed_at: str = field(default="", compare=False)


_COLUMNS = "qualname, package, version, content, path, line"


class Store:
    """Índice local. Si se indica `model`, se valida y se registra."""

    def __init__(self, path: Path, dim: int, model: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._db = sqlite3.connect(path)
        try:
            self._open(dim, model)
        except BaseException:
            self._db.close()
            raise

    def _open(self, dim: int, model: str | None) -> None:
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)
        existing = {
            row[0]
            for row in self._db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self._check_stored_dimension(dim)
        self._db.executescript(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                qualname TEXT NOT NULL,
                package TEXT NOT NULL,
                version TEXT NOT NULL,
                content TEXT NOT NULL,
                path TEXT NOT NULL DEFAULT '',
                line INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_package ON chunks(package);
            CREATE INDEX IF NOT EXISTS idx_chunks_qualname ON chunks(qualname);
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
                USING vec0(embedding float[{dim}]);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(qualname, content,
                           tokenize = "unicode61 remove_diacritics 2 tokenchars '_'");
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                name TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                version TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
        """)
        with self._db:
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(chunks)")}
            if "path" not in columns:  # índice anterior a los enlaces al editor
                self._db.execute(
                    "ALTER TABLE chunks ADD COLUMN path TEXT NOT NULL DEFAULT ''"
                )
                self._db.execute(
                    "ALTER TABLE chunks ADD COLUMN line INTEGER NOT NULL DEFAULT 0"
                )
            if "chunks_fts" not in existing:  # índice anterior a la búsqueda léxica
                self._db.execute(
                    "INSERT INTO chunks_fts(rowid, qualname, content) "
                    "SELECT id, qualname, content FROM chunks"
                )
            if "sources" not in existing:  # índice anterior a los metadatos de fuente
                self._db.execute(
                    "INSERT INTO sources(name, kind, version, indexed_at) "
                    "SELECT package, "
                    "CASE WHEN SUM(instr(qualname, '#') > 0) > 0 "
                    "THEN 'docs' ELSE 'package' END, MIN(version), '' "
                    "FROM chunks GROUP BY package"
                )
        self._check_and_record_meta(dim, model)

    def _check_stored_dimension(self, dim: int) -> None:
        row = self._db.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_chunks'"
        ).fetchone()
        found = re.search(r"float\[(\d+)\]", row[0]) if row else None
        if found and int(found.group(1)) != dim:
            raise IndexIntegrityError(
                f"El índice guarda vectores de {found.group(1)} dimensiones y la "
                f"configuración usa {dim}.",
                "Restaura embed_model/embed_dim o reconstruye el índice "
                "(borra data/consultor.db y reindexa).",
            )

    def _check_and_record_meta(self, dim: int, model: str | None) -> None:
        stored = self.meta()
        version = int(stored.get("format_version", "0") or 0)
        if version > FORMAT_VERSION:
            raise IndexIntegrityError(
                f"El índice usa el formato {version}, más nuevo que el soportado "
                f"({FORMAT_VERSION}).",
                "Actualiza ApiConsultor.",
            )
        old_model = stored.get("embed_model")
        if model is not None and old_model not in (None, model):
            raise IndexIntegrityError(
                f"El índice se creó con el modelo de embeddings '{old_model}' y la "
                f"configuración usa '{model}'.",
                "Restaura embed_model o reconstruye el índice "
                "(borra data/consultor.db y reindexa).",
            )
        updates = {"format_version": str(FORMAT_VERSION), "embed_dim": str(dim)}
        if model is not None:
            updates["embed_model"] = model  # un índice anterior adopta el modelo actual
        if "created_at" not in stored:
            updates["created_at"] = _now()
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                updates.items(),
            )

    def meta(self) -> dict[str, str]:
        """Metadatos del índice: formato, modelo y dimensión de embeddings, creación."""
        return dict(self._db.execute("SELECT key, value FROM meta").fetchall())

    def check_vector(self, vector: Sequence[float]) -> None:
        """Falla si el vector no tiene la dimensión del índice."""
        if len(vector) != self.dim:
            raise IndexIntegrityError(
                f"El modelo de embeddings produce vectores de {len(vector)} "
                f"dimensiones y el índice espera {self.dim}.",
                "¿Cambió el modelo? Revisa embed_model/embed_dim o reconstruye.",
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def replace(
        self,
        package: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        kind: str = "package",
    ) -> None:
        """Sustituye todo lo indexado de `package` (reindexado idempotente)."""
        for embedding in embeddings:
            self.check_vector(embedding)  # antes de tocar nada
        with self._db:  # una sola transacción
            old_ids = [
                (row[0],)
                for row in self._db.execute(
                    "SELECT id FROM chunks WHERE package = ?", (package,)
                )
            ]
            self._db.executemany("DELETE FROM vec_chunks WHERE rowid = ?", old_ids)
            self._db.executemany("DELETE FROM chunks_fts WHERE rowid = ?", old_ids)
            self._db.execute("DELETE FROM chunks WHERE package = ?", (package,))
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                cur = self._db.execute(
                    f"INSERT INTO chunks({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        chunk.qualname,
                        chunk.package,
                        chunk.version,
                        chunk.content,
                        chunk.path,
                        chunk.line,
                    ),
                )
                self._db.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, sqlite_vec.serialize_float32(list(embedding))),
                )
                self._db.execute(
                    "INSERT INTO chunks_fts(rowid, qualname, content) VALUES (?, ?, ?)",
                    (cur.lastrowid, chunk.qualname, chunk.content),
                )
            if chunks:
                self._db.execute(
                    "INSERT OR REPLACE INTO sources(name, kind, version, indexed_at) "
                    "VALUES (?, ?, ?, ?)",
                    (package, kind, chunks[0].version, _now()),
                )
            else:
                self._db.execute("DELETE FROM sources WHERE name = ?", (package,))

    def sources(self) -> list[SourceInfo]:
        rows = self._db.execute(
            "SELECT s.name, s.version, COUNT(c.id), s.kind, s.indexed_at "
            "FROM sources AS s LEFT JOIN chunks AS c ON c.package = s.name "
            "GROUP BY s.name ORDER BY s.name"
        )
        return [SourceInfo(*row) for row in rows]

    def remove(self, package: str) -> int:
        """Elimina una fuente con todos sus fragmentos; devuelve cuántos borró."""
        row = self._db.execute(
            "SELECT COUNT(*) FROM chunks WHERE package = ?", (package,)
        ).fetchone()
        self.replace(package, [], [])
        return int(row[0])

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

    def by_short_name(self, names: Iterable[str], max_matches: int = 5) -> list[Chunk]:
        """Símbolos cuyo último componente coincide (sin distinguir mayúsculas).

        Un nombre que coincide con más de `max_matches` símbolos es demasiado genérico
        para identificar uno y se ignora.
        """
        found: list[Chunk] = []
        for name in dict.fromkeys(n.lower() for n in names):
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM chunks WHERE lower(qualname) GLOB ? LIMIT ?",
                (f"*.{name}", max_matches + 1),
            ).fetchall()
            if 0 < len(rows) <= max_matches:
                found.extend(Chunk(*row) for row in rows)
        return found

    def lexical(self, query: str, limit: int) -> list[Chunk]:
        """Búsqueda por palabras (FTS5, ranking BM25) con más peso en el nombre.

        `query` usa la sintaxis de FTS5; si no es válida no se devuelve nada.
        """
        try:
            rows = self._db.execute(
                f"""
                SELECT {", ".join("c." + col.strip() for col in _COLUMNS.split(","))}
                FROM chunks_fts
                JOIN chunks AS c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts, 5.0, 1.0)
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [Chunk(*row) for row in rows]

    def search(self, embedding: Sequence[float], k: int) -> list[Chunk]:
        return [chunk for chunk, _distance in self.search_scored(embedding, k)]

    def search_scored(
        self, embedding: Sequence[float], k: int
    ) -> list[tuple[Chunk, float]]:
        """Vecinos más cercanos con su distancia (menor = más parecido)."""
        self.check_vector(embedding)
        rows = self._db.execute(
            f"""
            SELECT {", ".join("c." + col.strip() for col in _COLUMNS.split(","))},
                   v.distance
            FROM (
                SELECT rowid, distance FROM vec_chunks
                WHERE embedding MATCH ? AND k = ?
            ) AS v
            JOIN chunks AS c ON c.id = v.rowid
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(list(embedding)), k),
        )
        return [(Chunk(*row[:6]), float(row[6])) for row in rows]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
