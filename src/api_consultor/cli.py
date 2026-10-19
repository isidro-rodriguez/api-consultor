"""Interfaz de línea de comandos."""

import argparse
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from .config import SETTINGS
from .consultor import Consultor
from .docs import collect, extract_doc_chunks
from .indexer import extract_chunks
from .llm import embed
from .store import Chunk, Store

BATCH_SIZE = 32


def _save(store: Store, package: str, chunks: Sequence[Chunk]) -> None:
    if not chunks:
        print(f"{package}: nada indexable", file=sys.stderr)
        return
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        embeddings.extend(embed([chunk.content for chunk in batch]))
        print(f"\r{package}: {len(embeddings)}/{len(chunks)}", end="", flush=True)
    print()
    store.replace(package, chunks, embeddings)


def _index(packages: Sequence[str]) -> None:
    with Store(SETTINGS.db_path, SETTINGS.embed_dim) as store:
        for package in packages:
            try:
                chunks = extract_chunks(package)
            except ModuleNotFoundError:
                print(f"{package}: no está instalado en el entorno", file=sys.stderr)
                continue
            _save(store, package, chunks)


def _index_docs(directory: Path, exclude: Sequence[str]) -> None:
    if not directory.is_dir():
        print(f"{directory}: no existe el directorio", file=sys.stderr)
        return
    with Store(SETTINGS.db_path, SETTINGS.embed_dim) as store:
        for source, files in collect(directory).items():
            _save(store, source, extract_doc_chunks(source, files, exclude))


def _read_questions() -> Iterator[str]:
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line in {"salir", "exit"}:
            return
        if line:
            yield line


def _ask(question: str | None) -> None:
    questions = [question] if question else _read_questions()
    with Store(SETTINGS.db_path, SETTINGS.embed_dim) as store:
        consultor = Consultor(store)
        for text in questions:
            for token in consultor.ask(text):
                print(token, end="", flush=True)
            print()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="api-consultor", description="Consultor de APIs Python offline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="indexa paquetes instalados en el entorno")
    index.add_argument("packages", nargs="+")

    docs = sub.add_parser(
        "index-docs", help="indexa documentación local (.txt, .md, .rst, .zip)"
    )
    docs.add_argument("directory", nargs="?", type=Path, default=SETTINGS.docs_dir)
    docs.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        metavar="PATRÓN",
        help="patrón fnmatch de ruta a omitir (repetible), p. ej. 'whatsnew/*'",
    )

    ask = sub.add_parser("ask", help="consulta; sin pregunta, modo interactivo")
    ask.add_argument("question", nargs="?")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if args.command == "index":
        _index(args.packages)
    elif args.command == "index-docs":
        _index_docs(args.directory, args.exclude)
    else:
        _ask(args.question)
