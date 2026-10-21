# ApiConsultor

[![CI](https://github.com/isidro-rodriguez/api-consultor/actions/workflows/ci.yml/badge.svg)](https://github.com/isidro-rodriguez/api-consultor/actions/workflows/ci.yml)

Consultor de APIs Python **offline**. Indexa los docstrings de los paquetes instalados
y ficheros de documentación (`.txt`, `.md`, `.rst`, `.zip`) en una base SQLite local, y
responde preguntas con un LLM que corre en tu máquina mediante
[Ollama](https://ollama.com). Pensado para GPUs de gama media (8 GB de VRAM).

```text
paquetes instalados / documentación ──► fragmentos ──► embeddings (bge-m3) ──► SQLite + sqlite-vec
                                                                             │
pregunta ──► nombres + palabras (FTS5) + vectorial ──► fusión RRF ◄──────────┘
                     │
                     └──► Qwen3 8B (Ollama) ──► respuesta en Markdown + fuentes
```

## Requisitos

- Python 3.12+ y [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) con los modelos `qwen3:8b` y `bge-m3`

## Instalación

Como herramienta global (el comando `api` queda disponible en cualquier carpeta):

```bash
uv tool install git+https://github.com/isidro-rodriguez/api-consultor
api setup        # descarga qwen3:8b y bge-m3 y crea el modelo api-consultor
```

Actualizar: `uv tool upgrade api-consultor` y, si cambió el `Modelfile`, `api setup`.

Para desarrollar:

```bash
git clone https://github.com/isidro-rodriguez/api-consultor
cd api-consultor
uv sync
uv run api setup
```

Los ejemplos de abajo usan `uv run api`; con la herramienta instalada basta `api`.

## Uso

```bash
uv run api index requests rich          # docstrings de paquetes de la herramienta
uv run api index requests -p .venv      # ... o de un entorno virtual de tu proyecto
uv run api index-docs                   # documentación del usuario (zip, carpetas, ficheros)
uv run api index-docs ~/mis-docs        # otra carpeta (ofrece copiarla a la del usuario)
uv run api index-docs -x 'whatsnew/*'   # omitiendo rutas (patrones fnmatch)
uv run api ask "¿cómo hago un POST con timeout?"
uv run api ask                          # modo interactivo
uv run api ask --debug "..."            # muestra candidatos, señales y distancias
uv run api sources                      # qué hay indexado y si está al día
uv run api sources --remove requests   # elimina una fuente del índice
uv run api sources --relocate notas D:/docs/notas  # una fuente movida de sitio
uv run api update                       # reindexa los paquetes que han cambiado de versión
uv run api eval evals/stdlib.toml       # mide la calidad de la recuperación
uv run api --version
```

- Un paquete solo se puede indexar si está instalado en el entorno que se indica con
  `--python` (carpeta del venv o su Python); sin él, en el de la propia herramienta. Cada
  fuente recuerda su entorno: `api update` y el aviso de versión obsoleta lo usan.
- La documentación local vive por defecto en `<carpeta de datos>/docs`. Al indexar otra
  carpeta, `api index-docs` pregunta si copiarla allí (por defecto sí; `--copy` /
  `--no-copy` evitan la pregunta), de modo que no se pierde si mueves o borras el
  original. Admite, por ejemplo, el zip de texto de la documentación de Python
  (<https://docs.python.org/3/download.html>).
- La base de datos vive en la carpeta de datos del usuario (`api config` muestra su ruta),
  no en el proyecto: `%LOCALAPPDATA%\\api-consultor` (Windows), `~/.local/share/api-consultor`
  (Linux) o `~/Library/Application Support/api-consultor` (macOS). Cámbiala con
  `[paths] db` (y `[paths] docs` para la documentación) en `config.toml`.
- El índice guarda las rutas de los ficheros relativas a la raíz de cada fuente. Si
  borras un venv o mueves una carpeta, `api sources` marca la fuente como «no encontrada»,
  el panel de `api ask` indica «fichero no disponible» (sin enlace) y
  `api sources --relocate FUENTE CARPETA` la recoloca sin reindexar.

## Integridad del índice

El índice registra el modelo y la dimensión de los embeddings, el
formato y, por fuente, su tipo, versión y fecha de indexado.

- Si cambias `embed_model` o `embed_dim`, la CLI se niega a insertar o consultar vectores
  incompatibles y explica cómo reconstruir el índice (borrar el fichero de la base).
- Las respuestas citan los fragmentos como `[DOC-01]`; el panel «Fuentes» marca los
  citados con ✔ y avisa si el modelo cita ids inexistentes o no cita nada. Tras
  actualizar el `Modelfile`, recrea el modelo con `api setup`.
- Al preguntar, avisa si un paquete indexado ya no coincide con la versión instalada y
  marca esas fuentes en el panel de resultados. `api update` reindexa solo esas.
- La versión se resuelve por la distribución instalada (`yaml` → PyYAML) y, para la
  biblioteca estándar, es la de Python. La documentación no se compara (solo se comprueba que su carpeta exista).
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

## Configuración

Los valores por defecto están en `src/api_consultor/config.py`; los tuyos van en un
`config.toml` opcional que sobrescribe solo lo que indiques. `api config` muestra los
valores efectivos (y cuáles son tuyos) y, si el fichero no existe, ofrece crearlo con
una plantilla comentada.

```toml
[models]      # chat, embed, embed_dim
[paths]       # db, docs
[retrieval]   # k, candidates, max_per_symbol, max_distance (sin definir = desactivado)
[display]     # width
[links]       # url
```

Ruta: `%APPDATA%\api-consultor` (Windows), `~/.config/api-consultor` (Linux, o
`$XDG_CONFIG_HOME`), `~/Library/Application Support/api-consultor` (macOS). Un fichero
inválido se rechaza con un mensaje claro antes de ejecutar nada. El prompt y los
parámetros del modelo siguen en `src/api_consultor/Modelfile`.

### Enlaces a las fuentes

El panel «Fuentes» enlaza cada fragmento con su fichero (hiperenlace de terminal: en la
mayoría de terminales, Ctrl+clic). Se abre con la aplicación que tu sistema asocie al
tipo de fichero; para usar un editor concreto define `[links] url`, con los campos
`{file}`, `{path}`, `{uri}` (por defecto) y `{line}`:

```toml
[links]
url = "vscode://file/{path}:{line}"
```

Los `.zip` de documentación se extraen (solo los textos) a `docs-cache/`, junto a la base
de datos, para que sus fragmentos también tengan fichero; `api sources --remove` borra
esa extracción. Tras actualizar, reindexa para guardar las ubicaciones.

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
- Los tests no usan Ollama ni tocan el índice o `docs/` reales.
- **CI** (GitHub Actions, `.github/workflows/ci.yml`): `ruff`, `ty` y `uv.lock` en Linux;
  tests con cobertura en Linux y Windows (Python 3.12 y 3.13); y construcción del wheel,
  instalado en un entorno limpio. Dependabot (`.github/dependabot.yml`) propone cada
  semana actualizaciones de dependencias (`uv`) y de las acciones.
- **Publicación** (`publish.yml`): al subir una etiqueta `vX.Y.Z` que coincide con la
  versión de `pyproject.toml`, se ejecuta el CI y se publica en PyPI con publicación de
  confianza (sin tokens). Antes de la primera vez, en PyPI: *Publishing → Add a pending
  publisher* (propietario `isidro-rodriguez`, repositorio `api-consultor`, flujo
  `publish.yml`, entorno `pypi`), y crea el entorno `pypi` en GitHub (Settings →
  Environments).
- Versionado SemVer, commits convencionales y cambios en [CHANGELOG.md](CHANGELOG.md).
  Guía para agentes de código en [AGENTS.md](AGENTS.md).

## Limitaciones conocidas

- La calidad de las respuestas depende de la documentación disponible: los símbolos sin
  docstring no se indexan.
- Un docstring largo se reparte en varios fragmentos (`nombre [2]`…): el modelo puede
  recibir solo uno de ellos.
- No hay umbral de relevancia: siempre se responde con lo recuperado.
