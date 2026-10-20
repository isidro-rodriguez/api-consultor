# Changelog

Todos los cambios relevantes de este proyecto se documentan aquí.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el
proyecto usa [Versionado Semántico](https://semver.org/lang/es/).

## [Unreleased]

### Añadido

- Comando `api eval`: mide Recall@k y MRR de la recuperación con casos de referencia
  en TOML, con desglose por categoría y lista de fallos (`-v` para todos los casos).
- Casos de ejemplo sobre la biblioteca estándar en `evals/stdlib.toml`.
- Banco de regresión sintético (`tests/test_benchmark.py`) con suelos de calidad.
- `api eval` avisa cuando las fuentes esperadas no están en el índice, y `expected`
  admite alternativas con `|` (API indexada o página de documentación).

### Cambiado

- `Consultor.retrieve()` acepta un límite opcional `k` (por defecto, `retrieval_k`).
- Recuperación híbrida: a los nombres cualificados y la búsqueda vectorial se suman los
  nombres cortos de símbolos (`rglob`) y una búsqueda por palabras sobre el contenido
  (FTS5, BM25); las cuatro señales se fusionan con Reciprocal Rank Fusion sobre hasta
  `retrieval_candidates` candidatos por método.
- El índice se migra solo al abrirlo (se crea la tabla FTS5): no hace falta reindexar.
- `top_k` pasa a llamarse `retrieval_k` en `Settings`, para no confundirlo con el
  `top_k` de generación de Ollama, y se añade `retrieval_candidates`.

### Corregido

- La tabla de fallos de `api eval` colapsaba la columna del rango y recortaba en exceso
  la de fuentes esperadas.

## [0.1.0] - 2026-10-19

Primera versión etiquetada.

### Añadido

- Indexado de paquetes Python instalados: un fragmento por módulo, función, clase y
  método público con docstring, guardados en SQLite con `sqlite-vec`.
- Indexado de documentación local (`.txt`, `.md`, `.rst`; sueltos, en carpetas o en
  `.zip`) con fragmentación por secciones y exclusión por patrones.
- Recuperación combinada (nombre exacto y búsqueda vectorial) y generación con
  Qwen3 8B a través de Ollama (`Modelfile`).
- CLI con Typer: `index`, `index-docs`, `ask` (con modo interactivo), `sources` y
  `--version`.
- Salida con Rich: tema Monokai, Markdown en streaming con ancho máximo de 90
  caracteres, panel de fuentes, barras de progreso y panel de versión con banner
  figlet.
- Mensajes claros cuando Ollama no responde o falta un modelo.
- Suite de tests con pytest (cobertura mínima del 90 %), tipado estricto con `ty`,
  `ruff` y hooks de `pre-commit` y `pre-push`.
- Documentación del proyecto: `README.md`, `AGENTS.md` y este `CHANGELOG.md`.
