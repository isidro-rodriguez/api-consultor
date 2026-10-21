# AGENTS.md

Guía para agentes de código que trabajan en este repositorio. Sé conciso, verifica
antes de afirmar y avanza en cambios pequeños.

## Qué es

Consultor de APIs Python offline: indexa docstrings de paquetes instalados y
documentación local en SQLite (`sqlite-vec`) y responde con un LLM local vía Ollama
(RAG). Python 3.12+, gestionado con `uv`. CLI: `api` (Typer + Rich).

## Comandos

```bash
uv sync                                   # entorno + dependencias (.venv)
uv run pytest                             # tests + cobertura (mínimo 90 %)
uv run ruff check . --fix && uv run ruff format .
uv run ty check                           # tipado estricto (all = "error")
uv run pre-commit run --all-files         # todo lo anterior + higiene
uv run api --help
uv run api eval evals/stdlib.toml         # lo ejecuta el usuario (usa el índice real)
```

Definición de hecho: `ruff`, `ty` y `pytest` en verde. Pega la salida real; no
declares terminada una tarea que no hayas ejecutado.

## Mapa del código (`src/api_consultor/`)

| Módulo          | Responsabilidad                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `cli.py`        | Comandos Typer, orquestación y errores de Ollama. Sin lógica de dominio.                        |
| `indexer.py`    | Extrae chunks de paquetes instalados (`inspect`), por secciones sin truncar.                    |
| `records.py`    | `Chunk`, `SourceInfo`, `Extraction`: solo stdlib (los carga también `probe.py`).                |
| `external.py`   | Lado de la app para `--python`: resuelve el intérprete y lanza `probe.py`.                      |
| `probe.py`      | Guion stdlib que corre con el Python de otro entorno (extrae chunks/versiones; JSON).           |
| `docs.py`       | Lee `.txt`/`.md`/`.rst` (sueltos, carpetas, `.zip`) y los trocea por secciones.                 |
| `store.py`      | SQLite + `sqlite-vec` + FTS5 + metadatos (`meta`, `sources`); valida modelo y dimensión.        |
| `consultor.py`  | RAG: `retrieve()` (nombres + FTS5 + vectorial, fusión RRF) y `stream()`.                        |
| `versions.py`   | Versión instalada (distribución ≠ módulo) y estado de cada fuente indexada.                     |
| `userconfig.py` | `config.toml` del usuario (sobrescribe `Settings`, plantilla de enlaces) y `api config`.        |
| `evaluation.py` | Recall@k, MRR y carga de casos de `api eval`.                                                   |
| `llm.py`        | Capa fina sobre el cliente de Ollama (`embed`, `stream_chat`).                                  |
| `render.py`     | Toda la salida de consola: tema Monokai, Markdown, tablas, versión.                             |
| `config.py`     | `Settings` inmutable (modelos, rutas, `retrieval_k`, `retrieval_candidates`, ancho de consola). |

Nada imprime fuera de `render.py`. `src/api_consultor/Modelfile` (empaquetado) define el modelo `api-consultor`; `models.py` lo localiza y `api setup` lo crea.

## Convenciones

- Tipado estricto en `src/` y `tests/`: anota todo; evita `Any` y `# type: ignore`
  (si es inevitable, `# ty: ignore[regla]` con motivo).
- Estilo pythónico y PEP 8/257/484/695; `pathlib` en lugar de `os.path`; `dataclass`
  con `frozen=True, slots=True` para datos; textos de usuario en español.
- Docstrings estilo Google (`Args:`, `Returns:`, `Raises:`, `Yields:`, `Attributes:`) en
  todo lo público, constantes incluidas (cadena justo tras la asignación); ruff (`D`) lo
  exige. Los comandos Typer llevan una sola línea: es su texto de `--help`.
- Dependencias nuevas solo si están justificadas; añádelas con `uv add` (runtime) o
  `uv add --dev` (desarrollo). `uv.lock` se versiona.
- Commits convencionales (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
- Versionado SemVer: toda mejora visible para el usuario añade una línea a
  `## [Unreleased]` de `CHANGELOG.md`. Al publicar: `uv version --bump <nivel>`,
  mover lo de Unreleased a la nueva versión y `git tag -a vX.Y.Z`.

## Tests

- Cada cambio de comportamiento lleva tests; cada bug, un test de regresión.
- Los tests corren en un directorio temporal (fixture `isolated_cwd`) y **nunca**
  usan Ollama real (fixtures `stub_ollama` y `block_real_ollama`). Embeddings
  sintéticos con `hash_embedding`. Paquetes de prueba con `make_package`.
- Si una prueba necesita el modelo real, no es un test unitario: documéntalo aparte.
- Calidad de recuperación: `tests/test_benchmark.py` (corpus sintético, suelos de
  no regresión) y `api eval` (índice y embeddings reales, lo ejecuta el usuario). Un
  agente **no** ejecuta `api eval` contra el índice real: usa el banco sintético y
  sube los umbrales cuando una mejora los supere con holgura.

## Índice y migraciones

- El formato del índice está versionado (`FORMAT_VERSION` en `store.py`). Todo cambio de
  esquema debe migrar solo al abrir (sin reindexar) y llevar un test con un índice
  «antiguo» simulado (ver `test_a_legacy_index_gets_sources_inferred_from_its_chunks`).
- Las rutas de `chunks.path` son relativas a `sources.root`; `sources.python` guarda el
  intérprete de un paquete indexado con `--python`. `probe.py`, `indexer.py`, `docs.py`,
  `versions.py`, `external.py` y `records.py` deben seguir siendo stdlib-only y sin
  importar `store`/`cli`, porque la sonda los carga en el entorno ajeno.
- Nunca insertes ni consultes vectores sin validar su dimensión (`Store.check_vector`).
- Un cambio de `embed_model`/`embed_dim` invalida el índice: se detecta y se pide
  reconstruirlo; no lo hagas tú sobre el índice real.

## Datos y privacidad (reglas duras)

- **No leas, abras ni consultes la base real** (carpeta de datos del usuario; ver
  `api config`). Valida con bases temporales y
  datos sintéticos.
- No borres, reconstruyas ni reindexes el índice real. No toques `docs/` (material del
  usuario, ignorado por git).
- No ejecutes `git push --force`, `git reset --hard` ni reescribas historial sin pedirlo.

## Hardware objetivo

GPU de gama media (8 GB de VRAM) y unos 16 GB de RAM. Mantén compatible:

- Ollama con un LLM de ~8B cuantizado (Qwen3 8B) y contexto de 8k tokens.
- Embeddings ejecutables en CPU; nunca dos modelos grandes en GPU a la vez.
- Contexto enviado al modelo moderado: `retrieval_k` (5) fragmentos de ~2000 caracteres.

No propongas modelos mayores sin medir que el cuello de botella es la generación y no
la recuperación.

## Flujo de trabajo

1. Lee el código afectado antes de asumir que algo falta.
2. Cambios pequeños y verificables; una transformación grande cada vez.
3. Ejecuta los comandos de calidad tras cada fase; informa de resultados reales.
4. Conserva el comportamiento correcto existente; no rehagas partes sin un defecto
   concreto.
5. Si algo no se puede completar con la arquitectura actual, dilo y propón la
   alternativa más simple en lugar de simular resultados.
