"""Almacén SQLite: vectores (sqlite-vec), palabras (FTS5), nombres y metadatos."""

import re
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import sqlite_vec

from .records import Chunk, SourceInfo

FORMAT_VERSION = 4
"""Formato del índice.

1 sin metadatos, 2 con `meta`/`sources`, 3 con `path`/`line`, 4 con rutas relativas
a la raíz de la fuente y `root`/`python` en `sources`.
"""


class IndexIntegrityError(Exception):
    """El índice no es compatible con la configuración actual.

    Args:
        message: Qué ocurre.
        hint: Cómo solucionarlo, si se sabe.

    Attributes:
        hint: Sugerencia para el usuario, o `None`.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


def _guess_root(name: str, kind: str, rows: list[tuple[int, str, str]]) -> str | None:
    """Deduce la raíz de una fuente de un índice anterior a partir de sus rutas.

    Para un paquete es la carpeta que contiene su directorio (o su módulo); para
    documentación, lo que queda al quitar la ruta relativa que recuerda el nombre.
    """
    for _row_id, qualname, raw in rows:
        path = Path(raw)
        if not path.is_absolute():
            continue
        if kind == "docs":
            relative = re.sub(r" \[\d+\]$", "", qualname.split("#", 1)[0])
            posix = path.as_posix()
            if posix.endswith("/" + relative):
                return posix[: -len(relative) - 1] or "/"
            continue
        top = name.split(".", 1)[0]
        for parent in path.parents:
            if parent.name == top:
                return str(parent.parent)
        if path.stem == top:
            return str(path.parent)
    return None


_COLUMNS = "qualname, package, version, content, path, line"


class Store:
    """Índice local: SQLite con vectores (sqlite-vec), palabras (FTS5) y metadatos.

    Se puede usar como gestor de contexto. Al abrirlo migra los índices de formatos
    anteriores y valida que la dimensión (y el modelo, si se indica) encajen.

    Args:
        path: Fichero de la base de datos (se crea con su carpeta si no existe).
        dim: Dimensión de los vectores de embeddings.
        model: Modelo de embeddings; si se indica, se valida y se registra.

    Attributes:
        dim: Dimensión de los vectores del índice.

    Raises:
        IndexIntegrityError: Si la dimensión, el modelo o el formato no son
            compatibles con el índice existente.
    """

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
                indexed_at TEXT NOT NULL,
                root TEXT NOT NULL DEFAULT '',
                python TEXT NOT NULL DEFAULT ''
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
            self._migrate_roots()
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

    def _migrate_roots(self) -> None:
        """Pasa un índice anterior al formato 4: raíz por fuente y rutas relativas."""
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(sources)")}
        if "root" in columns:
            return
        self._db.execute("ALTER TABLE sources ADD COLUMN root TEXT NOT NULL DEFAULT ''")
        self._db.execute(
            "ALTER TABLE sources ADD COLUMN python TEXT NOT NULL DEFAULT ''"
        )
        for name, kind in self._db.execute("SELECT name, kind FROM sources").fetchall():
            rows = self._db.execute(
                "SELECT id, qualname, path FROM chunks "
                "WHERE package = ? AND path != ''",
                (name,),
            ).fetchall()
            root = _guess_root(name, kind, rows)
            if root is None:
                continue  # las rutas absolutas siguen funcionando
            self._db.execute("UPDATE sources SET root = ? WHERE name = ?", (root, name))
            base = Path(root)
            self._db.executemany(
                "UPDATE chunks SET path = ? WHERE id = ?",
                [
                    (Path(path).relative_to(base).as_posix(), row_id)
                    for row_id, _qualname, path in rows
                    if Path(path).is_absolute() and Path(path).is_relative_to(base)
                ],
            )

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
                "(borra el fichero del índice, ruta en `api config`, y reindexa).",
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
                "(borra el fichero del índice, ruta en `api config`, y reindexa).",
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
        """Devuelve los metadatos: formato, modelo y dimensión, y fecha de creación."""
        return dict(self._db.execute("SELECT key, value FROM meta").fetchall())

    def check_vector(self, vector: Sequence[float]) -> None:
        """Comprueba que un vector tiene la dimensión del índice.

        Args:
            vector: Embedding a validar.

        Raises:
            IndexIntegrityError: Si la dimensión no coincide.
        """
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
        """Cierra la conexión con la base de datos."""
        self._db.close()

    def replace(
        self,
        package: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        kind: str = "package",
        *,
        root: str = "",
        python: str = "",
    ) -> None:
        """Sustituye todo lo indexado de una fuente (reindexado idempotente).

        Todo ocurre en una transacción: si falla, se conserva lo anterior. Sin
        fragmentos, la fuente desaparece del índice.

        Args:
            package: Nombre de la fuente.
            chunks: Fragmentos nuevos.
            embeddings: Un embedding por fragmento, en el mismo orden.
            kind: `"package"`, `"stdlib"` o `"docs"`.
            root: Carpeta a la que son relativas las rutas de los fragmentos.
            python: Intérprete del entorno indexado; vacío si es el de la herramienta.

        Raises:
            IndexIntegrityError: Si algún embedding no tiene la dimensión del índice.
            ValueError: Si hay distinto número de fragmentos y de embeddings.
        """
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
                    "INSERT OR REPLACE INTO sources"
                    "(name, kind, version, indexed_at, root, python) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (package, kind, chunks[0].version, _now(), root, python),
                )
            else:
                self._db.execute("DELETE FROM sources WHERE name = ?", (package,))

    def sources(self) -> list[SourceInfo]:
        """Lista las fuentes indexadas, por nombre, con su número de fragmentos."""
        rows = self._db.execute(
            "SELECT s.name, s.version, COUNT(c.id), s.kind, s.indexed_at, "
            "s.root, s.python "
            "FROM sources AS s LEFT JOIN chunks AS c ON c.package = s.name "
            "GROUP BY s.name ORDER BY s.name"
        )
        return [SourceInfo(*row) for row in rows]

    def paths(self, package: str) -> list[str]:
        """Lista los ficheros (rutas relativas) de los que salen los fragmentos.

        Args:
            package: Nombre de la fuente.

        Returns:
            Rutas distintas y ordenadas; vacío si la fuente no tiene ubicaciones.
        """
        rows = self._db.execute(
            "SELECT DISTINCT path FROM chunks WHERE package = ? AND path != '' "
            "ORDER BY path",
            (package,),
        )
        return [row[0] for row in rows]

    def relocate(self, package: str, root: str) -> bool:
        """Cambia la carpeta raíz de una fuente sin reindexar.

        Args:
            package: Nombre de la fuente.
            root: Nueva carpeta que contiene sus ficheros.

        Returns:
            `True` si la fuente existe.
        """
        with self._db:
            cur = self._db.execute(
                "UPDATE sources SET root = ? WHERE name = ?", (root, package)
            )
        return cur.rowcount > 0

    def remove(self, package: str) -> int:
        """Elimina una fuente con todos sus fragmentos.

        Args:
            package: Nombre de la fuente (si no existe, no hace nada).

        Returns:
            Cuántos fragmentos se borraron.
        """
        row = self._db.execute(
            "SELECT COUNT(*) FROM chunks WHERE package = ?", (package,)
        ).fetchone()
        self.replace(package, [], [])
        return int(row[0])

    def qualnames(self) -> list[str]:
        """Devuelve el nombre cualificado de todos los fragmentos del índice."""
        return [row[0] for row in self._db.execute("SELECT qualname FROM chunks")]

    def by_name(self, names: Iterable[str], limit_per_name: int = 2) -> list[Chunk]:
        """Busca por nombre cualificado exacto o por sufijo tras un punto.

        Args:
            names: Nombres a buscar, p. ej. `pathlib.Path.rglob` o `Path.rglob`.
            limit_per_name: Máximo de resultados por nombre.

        Returns:
            Los fragmentos encontrados, agrupados por nombre.
        """
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
        """Busca símbolos por su último componente, sin distinguir mayúsculas.

        Un nombre que coincide con más de `max_matches` símbolos es demasiado genérico
        para identificar uno y se ignora.

        Args:
            names: Nombres cortos, p. ej. `rglob`.
            max_matches: Máximo de coincidencias para que un nombre cuente.

        Returns:
            Los fragmentos de los nombres suficientemente específicos.
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
        """Busca por palabras (FTS5, ranking BM25) con más peso en el nombre.

        Args:
            query: Consulta en la sintaxis de FTS5.
            limit: Máximo de resultados.

        Returns:
            Fragmentos de mejor a peor; vacío si la consulta no es válida.
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
        """Busca los vecinos más cercanos a un embedding.

        Args:
            embedding: Vector de la consulta.
            k: Cuántos vecinos devolver.

        Returns:
            Los `k` fragmentos más cercanos, del más parecido al menos.

        Raises:
            IndexIntegrityError: Si el vector no tiene la dimensión del índice.
        """
        return [chunk for chunk, _distance in self.search_scored(embedding, k)]

    def search_scored(
        self, embedding: Sequence[float], k: int
    ) -> list[tuple[Chunk, float]]:
        """Busca los vecinos más cercanos junto con su distancia.

        Args:
            embedding: Vector de la consulta.
            k: Cuántos vecinos devolver.

        Returns:
            Pares (fragmento, distancia), de menor a mayor distancia.

        Raises:
            IndexIntegrityError: Si el vector no tiene la dimensión del índice.
        """
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
