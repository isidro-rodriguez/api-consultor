"""Almacén SQLite: vectores (sqlite-vec), palabras (FTS5), nombres y metadatos."""

import hashlib
import re
import sqlite3
from array import array
from collections.abc import Collection, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import sqlite_vec

from .records import Chunk, SourceInfo

FORMAT_VERSION = 6
"""Formato del índice.

1 sin metadatos, 2 con `meta`/`sources`, 3 con `path`/`line`, 4 con rutas relativas
a la raíz de la fuente y `root`/`python` en `sources`, 5 con `chunks.hash` (caché de
embeddings) y `sources.module` (paquete importable, distinto del nombre de la fuente),
6 con raíces en inglés en la búsqueda por palabras (`replace` encuentra `replacing`).
"""

_FTS_TOKENIZER = "porter unicode61 remove_diacritics 2 tokenchars '_'"
"""Tokenizador de FTS5: `porter` reduce las palabras inglesas a su raíz."""

_HASH_BATCH = 500
"""Hashes por consulta al buscar embeddings ya calculados (límite de parámetros)."""


def content_hash(content: str) -> str:
    """Identifica el texto de un fragmento: igual texto, mismo embedding.

    Args:
        content: Texto que se vectoriza.

    Returns:
        SHA-256 en hexadecimal.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def environment_label(python: str) -> str:
    """Etiqueta corta de un entorno para distinguir fuentes con el mismo paquete.

    Es la carpeta del proyecto que contiene el venv (`reCocina` para
    `reCocina/.venv/bin/python`), o la del propio venv si no tiene un nombre
    genérico; `herramienta` para el entorno de la aplicación.

    Args:
        python: Intérprete del entorno; vacío para el de la herramienta.

    Returns:
        La etiqueta.
    """
    if not python:
        return "herramienta"
    path = Path(python)
    venv = path.parent.parent if path.parent.name in {"bin", "Scripts"} else path.parent
    if venv.name.lstrip(".") in {"venv", "env"} and venv.parent.name:
        return venv.parent.name
    return venv.name or "entorno"


BUSY_TIMEOUT = 30.0
"""Segundos que se espera a que otro proceso suelte el índice antes de fallar."""


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
_PREFIXED = ", ".join("c." + column.strip() for column in _COLUMNS.split(","))


def _only(
    sources: Collection[str] | None, column: str = "package"
) -> tuple[str, list[str]]:
    """Condición SQL (con su `AND`) que limita la búsqueda a unas fuentes.

    Args:
        sources: Fuentes permitidas; `None` para no limitar.
        column: Columna con el nombre de la fuente.

    Returns:
        El fragmento SQL (vacío si no hay límite) y sus parámetros.
    """
    if sources is None:
        return "", []
    names = sorted(sources)
    return f" AND {column} IN ({', '.join('?' * len(names))})", names


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
        # Si otro proceso está escribiendo (p. ej. un `api index` en paralelo), espera
        # en vez de fallar con «database is locked».
        self._db = sqlite3.connect(path, timeout=BUSY_TIMEOUT)
        try:
            self._open(dim, model)
        except BaseException:
            self._db.close()
            raise

    def _open(self, dim: int, model: str | None) -> None:
        # WAL: se puede consultar mientras otro proceso indexa (lectores y escritor
        # no se bloquean). Persiste en el fichero; sin efecto en `:memory:`.
        self._db.execute("PRAGMA journal_mode = WAL")
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
                line INTEGER NOT NULL DEFAULT 0,
                hash TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_package ON chunks(package);
            CREATE INDEX IF NOT EXISTS idx_chunks_qualname ON chunks(qualname);
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
                USING vec0(embedding float[{dim}]);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(qualname, content, tokenize = "{_FTS_TOKENIZER}");
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
                python TEXT NOT NULL DEFAULT '',
                module TEXT NOT NULL DEFAULT ''
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
            self._migrate_hashes_and_modules()
            self._migrate_fts_stemming()
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash)")
        self._check_and_record_meta(dim, model)

    def _migrate_hashes_and_modules(self) -> None:
        """Migra al formato 5: hash por fragmento y paquete importable por fuente.

        Los hashes se calculan del texto ya guardado, así que los embeddings existentes
        sirven de caché desde el primer reindexado.
        """
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(chunks)")}
        if "hash" not in columns:
            self._db.execute(
                "ALTER TABLE chunks ADD COLUMN hash TEXT NOT NULL DEFAULT ''"
            )
        self._db.executemany(
            "UPDATE chunks SET hash = ? WHERE id = ?",
            [
                (content_hash(content), row_id)
                for row_id, content in self._db.execute(
                    "SELECT id, content FROM chunks WHERE hash = ''"
                ).fetchall()
            ],
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(sources)")}
        if "module" not in columns:
            self._db.execute(
                "ALTER TABLE sources ADD COLUMN module TEXT NOT NULL DEFAULT ''"
            )
            self._db.execute("UPDATE sources SET module = name WHERE kind != 'docs'")

    def _migrate_fts_stemming(self) -> None:
        """Migra al formato 6: rehace el índice de palabras con raíces (`porter`).

        Solo reconstruye `chunks_fts` a partir de `chunks`; los embeddings no cambian.
        """
        (sql,) = self._db.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'"
        ).fetchone()
        if "porter" in sql:
            return
        self._db.execute("DROP TABLE chunks_fts")
        self._db.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(qualname, content, "
            f'tokenize = "{_FTS_TOKENIZER}")'
        )
        self._db.execute(
            "INSERT INTO chunks_fts(rowid, qualname, content) "
            "SELECT id, qualname, content FROM chunks"
        )

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
        # Solo se escribe si algo cambia: abrir para consultar no debe pedir el
        # cerrojo de escritura (chocaría con un `api index` en marcha).
        if changed := [(k, v) for k, v in updates.items() if stored.get(k) != v]:
            with self._db:
                self._db.executemany(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", changed
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

    def source_name(self, module: str, python: str = "") -> str:
        """Nombre de fuente para un paquete indexado desde un entorno.

        Si ese paquete ya está indexado desde ese entorno, devuelve su nombre (el
        reindexado lo sustituye). Si no, usa el nombre del paquete y, cuando ya lo
        ocupa el mismo paquete de otro entorno, le añade la etiqueta del entorno
        (`zensical@reCocina`) para que no se pisen.

        Args:
            module: Nombre importable del paquete.
            python: Intérprete del entorno; vacío para el de la herramienta.

        Returns:
            El nombre de la fuente.
        """
        rows = self._db.execute(
            "SELECT name, python FROM sources "
            "WHERE kind != 'docs' AND (module = ? OR (module = '' AND name = ?))",
            (module, module),
        ).fetchall()
        for name, indexed_from in rows:
            if indexed_from == python:
                return str(name)
        taken = {row[0] for row in self._db.execute("SELECT name FROM sources")}
        if module not in taken:
            return module
        base = f"{module}@{environment_label(python)}"
        candidate, suffix = base, 2
        while candidate in taken:
            candidate, suffix = f"{base}-{suffix}", suffix + 1
        return candidate

    def cached_embeddings(self, contents: Iterable[str]) -> dict[str, list[float]]:
        """Busca embeddings ya calculados para unos textos.

        Un fragmento con el mismo texto (de cualquier fuente) tiene el mismo
        embedding: el índice usa un solo modelo, validado al abrirlo.

        Args:
            contents: Textos de los fragmentos.

        Returns:
            Embedding por hash de contenido (ver `content_hash`), solo de los que ya
            están en el índice.
        """
        hashes = list(dict.fromkeys(content_hash(text) for text in contents))
        found: dict[str, list[float]] = {}
        for start in range(0, len(hashes), _HASH_BATCH):
            batch = hashes[start : start + _HASH_BATCH]
            rows = self._db.execute(
                "SELECT c.hash, v.embedding FROM chunks AS c "
                "JOIN vec_chunks AS v ON v.rowid = c.id "
                f"WHERE c.hash IN ({', '.join('?' * len(batch))})",
                batch,
            )
            for digest, blob in rows:
                found.setdefault(digest, array("f", blob).tolist())
        return found

    def replace(
        self,
        package: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        kind: str = "package",
        *,
        root: str = "",
        python: str = "",
        module: str = "",
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
            module: Paquete importable si difiere del nombre de la fuente; por
                defecto, el nombre (salvo para documentación).

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
                    f"INSERT INTO chunks({_COLUMNS}, hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.qualname,
                        chunk.package,
                        chunk.version,
                        chunk.content,
                        chunk.path,
                        chunk.line,
                        content_hash(chunk.content),
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
                importable = module or ("" if kind == "docs" else package)
                self._db.execute(
                    "INSERT OR REPLACE INTO sources"
                    "(name, kind, version, indexed_at, root, python, module) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        package,
                        kind,
                        chunks[0].version,
                        _now(),
                        root,
                        python,
                        importable,
                    ),
                )
            else:
                self._db.execute("DELETE FROM sources WHERE name = ?", (package,))

    def sources(self) -> list[SourceInfo]:
        """Lista las fuentes indexadas, por nombre, con su número de fragmentos."""
        rows = self._db.execute(
            "SELECT s.name, s.version, COUNT(c.id), s.kind, s.indexed_at, "
            "s.root, s.python, s.module "
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

    def by_name(
        self,
        names: Iterable[str],
        limit_per_name: int = 2,
        sources: Collection[str] | None = None,
    ) -> list[Chunk]:
        """Busca por nombre cualificado exacto o por sufijo tras un punto.

        Args:
            names: Nombres a buscar, p. ej. `pathlib.Path.rglob` o `Path.rglob`.
            limit_per_name: Máximo de resultados por nombre.
            sources: Limita la búsqueda a estas fuentes; `None` para todas.

        Returns:
            Los fragmentos encontrados, agrupados por nombre.
        """
        where, params = _only(sources)
        found: list[Chunk] = []
        for name in names:
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM chunks "
                f"WHERE (qualname = ? OR qualname GLOB ?){where} LIMIT ?",
                (name, f"*.{name}", *params, limit_per_name),
            )
            found.extend(Chunk(*row) for row in rows)
        return found

    def by_short_name(
        self,
        names: Iterable[str],
        max_matches: int = 5,
        sources: Collection[str] | None = None,
    ) -> list[Chunk]:
        """Busca símbolos por su último componente, sin distinguir mayúsculas.

        Un nombre que coincide con más de `max_matches` símbolos es demasiado genérico
        para identificar uno y se ignora.

        Args:
            names: Nombres cortos, p. ej. `rglob`.
            max_matches: Máximo de coincidencias para que un nombre cuente.
            sources: Limita la búsqueda a estas fuentes; `None` para todas.

        Returns:
            Los fragmentos de los nombres suficientemente específicos.
        """
        where, params = _only(sources)
        found: list[Chunk] = []
        for name in dict.fromkeys(n.lower() for n in names):
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM chunks "
                f"WHERE lower(qualname) GLOB ?{where} LIMIT ?",
                (f"*.{name}", *params, max_matches + 1),
            ).fetchall()
            if 0 < len(rows) <= max_matches:
                found.extend(Chunk(*row) for row in rows)
        return found

    def frequent_terms(self, terms: Iterable[str], limit: int) -> set[str]:
        """Términos que aparecen en más de `limit` fragmentos (palabras comunes).

        Args:
            terms: Palabras a comprobar.
            limit: Máximo de fragmentos para no considerarla común.

        Returns:
            Las que superan el límite.
        """
        frequent: set[str] = set()
        for term in dict.fromkeys(terms):
            try:
                (count,) = self._db.execute(
                    "SELECT count(*) FROM (SELECT 1 FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? LIMIT ?)",
                    (f'"{term}"', limit + 1),
                ).fetchone()
            except sqlite3.OperationalError:  # término no válido para FTS5
                continue
            if count > limit:
                frequent.add(term)
        return frequent

    def mentions(
        self,
        owners: Iterable[str],
        word: str,
        name: str,
        sources: Collection[str] | None = None,
    ) -> bool:
        """Indica si el índice menciona un símbolo aunque no lo tenga como fragmento.

        Sirve para no tomar por inexistente algo sin docstring propio: `sys.argv` o
        `re.IGNORECASE` aparecen en el texto de su módulo.

        Args:
            owners: Símbolos dueños (p. ej. `json`, `pathlib.Path`) cuyo texto o el de
                sus miembros se revisa.
            word: Nombre corto que se busca en ese texto (`argv`).
            name: Nombre completo que basta encontrar en cualquier fragmento.
            sources: Limita la búsqueda a estas fuentes; `None` para todas.

        Returns:
            `True` si algún fragmento de los dueños contiene `word`, o alguno contiene
            `name` (distinguiendo mayúsculas).
        """
        where, params = _only(sources)
        families = sorted(set(owners))
        family = " OR ".join("qualname = ? OR qualname GLOB ?" for _ in families)
        family_params = [v for owner in families for v in (owner, f"{owner}.*")]
        condition = f"instr(content, ?) > 0 AND ({family})" if families else "0"
        row = self._db.execute(
            f"SELECT 1 FROM chunks WHERE (({condition}) OR instr(content, ?) > 0)"
            f"{where} LIMIT 1",
            (*([word, *family_params] if families else []), name, *params),
        ).fetchone()
        return row is not None

    def lexical(
        self, query: str, limit: int, sources: Collection[str] | None = None
    ) -> list[Chunk]:
        """Busca por palabras (FTS5, ranking BM25) con más peso en el nombre.

        Args:
            query: Consulta en la sintaxis de FTS5.
            limit: Máximo de resultados.
            sources: Limita la búsqueda a estas fuentes; `None` para todas.

        Returns:
            Fragmentos de mejor a peor; vacío si la consulta no es válida.
        """
        where, params = _only(sources, "c.package")
        try:
            rows = self._db.execute(
                f"""
                SELECT {_PREFIXED}
                FROM chunks_fts
                JOIN chunks AS c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?{where}
                ORDER BY bm25(chunks_fts, 5.0, 1.0)
                LIMIT ?
                """,
                (query, *params, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [Chunk(*row) for row in rows]

    def search(
        self,
        embedding: Sequence[float],
        k: int,
        sources: Collection[str] | None = None,
    ) -> list[Chunk]:
        """Busca los vecinos más cercanos a un embedding.

        Args:
            embedding: Vector de la consulta.
            k: Cuántos vecinos devolver.
            sources: Limita la búsqueda a estas fuentes; `None` para todas.

        Returns:
            Los `k` fragmentos más cercanos, del más parecido al menos.

        Raises:
            IndexIntegrityError: Si el vector no tiene la dimensión del índice.
        """
        return [chunk for chunk, _ in self.search_scored(embedding, k, sources)]

    def search_scored(
        self,
        embedding: Sequence[float],
        k: int,
        sources: Collection[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Busca los vecinos más cercanos junto con su distancia.

        Sin límite de fuentes usa el índice KNN de sqlite-vec. Con límite, calcula la
        misma distancia (L2) solo sobre los fragmentos de esas fuentes: el KNN no
        admite filtros y pedir más vecinos para filtrarlos después podría dejar fuera
        una fuente pequeña.

        Args:
            embedding: Vector de la consulta.
            k: Cuántos vecinos devolver.
            sources: Limita la búsqueda a estas fuentes; `None` para todas.

        Returns:
            Pares (fragmento, distancia), de menor a mayor distancia.

        Raises:
            IndexIntegrityError: Si el vector no tiene la dimensión del índice.
        """
        self.check_vector(embedding)
        vector = sqlite_vec.serialize_float32(list(embedding))
        if sources is None:
            rows = self._db.execute(
                f"""
                SELECT {_PREFIXED}, v.distance
                FROM (
                    SELECT rowid, distance FROM vec_chunks
                    WHERE embedding MATCH ? AND k = ?
                ) AS v
                JOIN chunks AS c ON c.id = v.rowid
                ORDER BY v.distance
                """,
                (vector, k),
            )
        else:
            where, params = _only(sources, "c.package")
            rows = self._db.execute(
                f"""
                SELECT {_PREFIXED}, vec_distance_l2(v.embedding, ?) AS distance
                FROM chunks AS c
                JOIN vec_chunks AS v ON v.rowid = c.id
                WHERE 1{where}
                ORDER BY distance
                LIMIT ?
                """,
                (vector, *params, k),
            )
        return [(Chunk(*row[:6]), float(row[6])) for row in rows]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
