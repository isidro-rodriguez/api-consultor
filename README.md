# ApiConsultor

Consultor de APIs Python **offline**. Indexa los docstrings de los paquetes instalados
y ficheros de documentación (`.txt`, `.md`, `.rst`, `.zip`) en una base SQLite local, y
responde preguntas con un LLM que corre en tu máquina mediante
[Ollama](https://ollama.com). Pensado para GPUs de gama media (8 GB de VRAM).

```
paquetes instalados / docs/ ──► fragmentos ──► embeddings (bge-m3) ──► SQLite + sqlite-vec
                                                                             │
pregunta ──► nombres + palabras (FTS5) + vectorial ──► fusión RRF ◄──────────┘
                     │
                     └──► Qwen3 8B (Ollama) ──► respuesta en Markdown + fuentes
```

## Requisitos

- Python 3.12+ y [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) con los modelos `qwen3:8b` y `bge-m3`

## Instalación

```bash
uv sync
ollama pull qwen3:8b
ollama pull bge-m3
ollama create api-consultor -f Modelfile
```

## Uso

```bash
uv run api index requests rich          # docstrings de paquetes instalados
uv run api index-docs                   # contenido de docs/ (zip, carpetas, ficheros)
uv run api index-docs -x 'whatsnew/*'   # omitiendo rutas (patrones fnmatch)
uv run api ask "¿cómo hago un POST con timeout?"
uv run api ask                          # modo interactivo
uv run api sources                      # qué hay indexado
uv run api eval evals/stdlib.toml       # mide la calidad de la recuperación
uv run api --version
```

- Un paquete solo se puede indexar si está instalado en el entorno
  (`uv add --dev <paquete>`).
- `docs/` admite, por ejemplo, el zip de texto de la documentación de Python
  (<https://docs.python.org/3/download.html>). Está ignorado por git.
- La base de datos vive en `data/consultor.db` (también ignorada).

## Evaluar la recuperación

`api eval` lanza preguntas de referencia contra el índice y mide si las fuentes
esperadas aparecen entre los primeros resultados:

- **R@k (Recall@k):** fracción de fuentes esperadas presentes en los `k` primeros.
- **MRR:** media de `1/posición` de la primera fuente correcta (1,0 es lo ideal).

Los casos son tablas `[[case]]` en un TOML (`question`, `expected`, `category`); ver
[evals/stdlib.toml](evals/stdlib.toml). `expected` admite nombres exactos, comodines `*`
y alternativas con `|` (`"json.dumps|library/json.txt#*"`). Si alguna fuente esperada
no está en el índice, `api eval` lo avisa: ese caso no puede acertar. Úsalo antes y después de cada cambio en la
recuperación o el chunking. Ejecútalo con Ollama en marcha: usa el índice real.

## Configuración

Modelos, ruta de la base, fragmentos enviados al modelo (`retrieval_k`), candidatos
por método (`retrieval_candidates`) y ancho de consola están en
`src/api_consultor/config.py`. Los parámetros y el prompt del modelo, en `Modelfile`.

## Desarrollo

```bash
uv sync
uv run pre-commit install    # hooks de pre-commit y pre-push
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
```

- **pre-commit:** higiene de ficheros, `uv.lock` al día, `ruff` y `ty`.
- **pre-push:** `ruff`, `ty` y `pytest` (cobertura mínima del 90 %).
- Los tests no usan Ollama ni tocan `data/` o `docs/`.
- Versionado SemVer, commits convencionales y cambios en [CHANGELOG.md](CHANGELOG.md).
  Guía para agentes de código en [AGENTS.md](AGENTS.md).

## Limitaciones conocidas

- La calidad de las respuestas depende de la documentación disponible: los símbolos sin
  docstring no se indexan.
- Los fragmentos de APIs Python se truncan a 2000 caracteres.
- Si cambia la versión de un paquete, hay que reindexarlo a mano: no se detecta.
- No hay umbral de relevancia: siempre se responde con lo recuperado.
