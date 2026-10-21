# ApiConsultor

[![PyPI](https://img.shields.io/pypi/v/api-consultor)](https://pypi.org/project/api-consultor/)
[![CI](https://github.com/isidro-rodriguez/api-consultor/actions/workflows/ci.yml/badge.svg)](https://github.com/isidro-rodriguez/api-consultor/actions/workflows/ci.yml)

Consultor **offline** de APIs Python y documentación local. Indexa docstrings y
ficheros (`.txt`, `.md`, `.rst`, `.zip`) en SQLite y responde con un LLM local vía
[Ollama](https://ollama.com) (`qwen3:8b` + `bge-m3`), citando las fuentes.

## Instalación

Requisitos: Python 3.12+, [uv](https://docs.astral.sh/uv/) y Ollama en marcha.

```bash
uv tool install api-consultor
api setup        # descarga los modelos y ofrece indexar la biblioteca estándar
```

Actualizar: `uv tool upgrade api-consultor` (y `api setup` si cambió el `Modelfile`).

## Uso

```bash
api index requests rich           # paquetes del entorno de la herramienta
api index pygame -p .venv         # ... o de un entorno virtual de tu proyecto
api index --stdlib                # biblioteca estándar
api index-docs [CARPETA]          # documentación local
api ask                           # entra en chat interactivo
api sources                       # qué hay indexado (--update, --remove, --relocate)
```

`api --help` y `api COMANDO --help` describen todas las opciones.

## Configuración

`api config` muestra los valores efectivos, las rutas de datos y, si no existe, ofrece
crear un `config.toml` comentado. Solo hace falta indicar lo que cambies:

```toml
[models]      # chat, embed, embed_dim
[paths]       # db, docs
[retrieval]   # k, candidates, max_per_symbol, max_distance
[display]     # width
[links]       # url, p. ej. "vscode://file/{path}:{line}"
```

## Documentación avanzada

- [Uso detallado](https://github.com/isidro-rodriguez/api-consultor/blob/main/docs/uso.md): todos los comandos, entornos `--python`, estados de
  las fuentes.
- [Configuración](https://github.com/isidro-rodriguez/api-consultor/blob/main/docs/configuracion.md): rutas, opciones de `config.toml`, enlaces a
  editores, `Modelfile`.
- [Cómo funciona](https://github.com/isidro-rodriguez/api-consultor/blob/main/docs/funcionamiento.md): indexado, recuperación híbrida, citas,
  integridad del índice y limitaciones.
- [Evaluar la recuperación](https://github.com/isidro-rodriguez/api-consultor/blob/main/docs/evaluacion.md): `api eval`, métricas y formato de los
  casos.
- [Desarrollo](https://github.com/isidro-rodriguez/api-consultor/blob/main/docs/desarrollo.md): calidad, CI y publicación en PyPI.

## Desarrollo

```bash
git clone https://github.com/isidro-rodriguez/api-consultor
cd api-consultor
uv sync && uv run pre-commit install
uv run pytest            # ruff, ty y pytest también corren en pre-commit / pre-push
```

Las versiones se publican en PyPI al subir una etiqueta `vX.Y.Z`. Detalles en
[AGENTS.md](https://github.com/isidro-rodriguez/api-consultor/blob/main/AGENTS.md) y
[CHANGELOG.md](https://github.com/isidro-rodriguez/api-consultor/blob/main/CHANGELOG.md).
