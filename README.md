# ApiConsultor

Consultor de APIs Python **offline**. Indexa los docstrings de los paquetes instalados
y ficheros de documentación (`.txt`, `.md`, `.rst`, `.zip`) en una base SQLite local, y
responde preguntas con un LLM que corre en tu máquina mediante
[Ollama](https://ollama.com). Pensado para GPUs de gama media (8 GB de VRAM).

```text
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
git clone https://github.com/isidro-rodriguez/api-consultor
cd api-consultor
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
uv run api ask --debug "..."            # muestra candidatos, señales y distancias
uv run api sources                      # qué hay indexado y si está al día
uv run api sources --remove requests   # elimina una fuente del índice
uv run api update                       # reindexa los paquetes que han cambiado de versión
uv run api eval evals/stdlib.toml       # mide la calidad de la recuperación
uv run api --version
```

- Un paquete solo se puede indexar si está instalado en el entorno
  (`uv add --dev <paquete>`).
- `docs/` admite, por ejemplo, el zip de texto de la documentación de Python
  (<https://docs.python.org/3/download.html>). Está ignorado por git.
- La base de datos vive en `data/consultor.db` (también ignorada).

## Integridad del índice

El índice (`data/consultor.db`) registra el modelo y la dimensión de los embeddings, el
formato y, por fuente, su tipo, versión y fecha de indexado.

- Si cambias `embed_model` o `embed_dim`, la CLI se niega a insertar o consultar vectores
  incompatibles y explica cómo reconstruir el índice (borrar `data/consultor.db`).
- Las respuestas citan los fragmentos como `[DOC-01]`; el panel «Fuentes» marca los
  citados con ✔ y avisa si el modelo cita ids inexistentes o no cita nada. Tras
  actualizar el `Modelfile`, recrea el modelo: `ollama create api-consultor -f Modelfile`.
- Al preguntar, avisa si un paquete indexado ya no coincide con la versión instalada y
  marca esas fuentes en el panel de resultados. `api update` reindexa solo esas.
- La versión se resuelve por la distribución instalada (`yaml` → PyYAML) y, para la
  biblioteca estándar, es la de Python. La documentación (`docs/`) no se compara.
- Un índice anterior se migra solo al abrirlo; se asume creado con el modelo configurado.

## Evaluar la recuperación

`api eval` lanza preguntas de referencia contra el índice y mide si las fuentes
esperadas aparecen entre los primeros resultados:

- **R@k (Recall@k):** fracción de fuentes esperadas presentes en los `k` primeros.
- **MRR:** media de `1/posición` de la primera fuente correcta (1,0 es lo ideal).

Los casos son tablas `[[case]]` en un TOML (`question`, `expected`, `category`); ver
[evals/stdlib.toml](evals/stdlib.toml). `expected` admite nombres exactos, comodines `*`
y alternativas con `|` (`"json.dumps|library/json.txt#*"`). Si alguna fuente esperada
no está en el índice, `api eval` lo avisa: ese caso no puede acertar. Úsalo antes y
después de cada cambio en la recuperación o el chunking. Ejecútalo con Ollama en
marcha: usa el índice real.

## Editor y enlaces

El panel «Fuentes» enlaza cada fragmento con su fichero (hiperenlace de terminal, si tu
terminal lo admite) y `api open NOMBRE` lo abre en tu editor, en la línea del símbolo o
sección. Los chunks de un `.zip` no tienen fichero. Tras actualizar, reindexa para
guardar las ubicaciones.

Editor por defecto: Windows `edit`, Linux `nano`, macOS TextEdit (`open -t`). Cámbialo en
`config.toml` (`api config` muestra su ruta y los valores efectivos):

```toml
[editor]
command = ["code", "--goto", "{file}:{line}"]   # lista; campos {file} {path} {uri} {line}
link = "vscode://file/{path}:{line}"            # por defecto "{uri}" (file://...)
```

Ruta: `%APPDATA%\api-consultor` (Windows), `~/.config/api-consultor` (Linux, o
`$XDG_CONFIG_HOME`), `~/Library/Application Support/api-consultor` (macOS).

## Configuración

Modelos, ruta de la base, fragmentos enviados al modelo (`retrieval_k`), candidatos
por método (`retrieval_candidates`), máximo de fragmentos por símbolo (`max_per_symbol`),
umbral de relevancia (`max_distance`, desactivado por defecto) y ancho de consola están en
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
- Un docstring largo se reparte en varios fragmentos (`nombre [2]`…): el modelo puede
  recibir solo uno de ellos.
- No hay umbral de relevancia: siempre se responde con lo recuperado.
