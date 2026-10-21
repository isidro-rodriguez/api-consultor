# Changelog

Todos los cambios relevantes de este proyecto se documentan aquí.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el
proyecto usa [Versionado Semántico](https://semver.org/lang/es/).

## [Unreleased]

### Cambiado

- La base de datos se guarda por defecto en la carpeta de datos del usuario, no en
  `data/` del proyecto. Para conservar la anterior: moverla allí o fijar
  `[paths] db = "data/consultor.db"` en `config.toml`.
- Docstrings en estilo Google en toda la API pública (constantes incluidas); ruff exige
  docstrings en lo público (`D`, convención Google).

### Corregido

- Títulos o nombres con corchetes (`[tool.ruff]`) rompían las tablas de Rich al
  interpretarse como estilos; ahora se imprimen literalmente.
- El contexto enviado al modelo escapa los atributos XML (comillas, `<`, `&`).
- `api sources --remove` contaba dos veces un nombre repetido en la confirmación.
- `api index` ya no se cae con paquetes que fallan al importarse o con nombres inválidos:
  avisa y sigue con el siguiente.
- Las rutas de `config.toml` admiten `~`.

### Añadido

- Los `.zip` de documentación se extraen a `docs-cache/` (junto a la base) al indexar, así sus
  fragmentos también tienen enlace; `api sources --remove` borra la extracción.
- `api sources --remove FUENTE` (`-r`, repetible, `-y` sin confirmar): elimina fuentes del
  índice; si alguna no existe, no borra nada.
- Enlaces a ficheros: los chunks guardan fichero y línea (formato de índice 3, migración
  automática; reindexa para obtener ubicaciones). El panel «Fuentes» enlaza cada fuente
  (Ctrl+clic) y lo indica con una pista.
- `config.toml` en el directorio de configuración del usuario: modelos, rutas, parámetros
  de recuperación, ancho y plantilla de enlace. `api config` muestra los valores efectivos
  y ofrece crear el fichero si no existe.
- Citas verificables: cada fragmento llega al modelo con un id (`DOC-01`), se le pide
  citarlos y `ask` comprueba la respuesta. El panel «Fuentes» muestra los ids, marca
  con ✔ los citados y avisa de citas inexistentes o de respuestas sin citas.
- `api ask --debug`: muestra candidatos, señales (N/S/L/V), distancias y tamaño del contexto.
- Diversidad por símbolo: como máximo `max_per_symbol` (2) fragmentos del mismo símbolo,
  con relleno si no hay variedad suficiente.
- Umbral de relevancia opcional (`max_distance`, desactivado): si ninguna coincidencia por
  nombre y la mejor distancia lo supera, `ask` se abstiene en lugar de inventar.
- Comando `api eval`: mide Recall@k y MRR de la recuperación con casos de referencia
  en TOML, con desglose por categoría y lista de fallos (`-v` para todos los casos).
- Casos de ejemplo sobre la biblioteca estándar en `evals/stdlib.toml`.
- Banco de regresión sintético (`tests/test_benchmark.py`) con suelos de calidad.
- `api eval` avisa cuando las fuentes esperadas no están en el índice, y `expected`
  admite alternativas con `|` (API indexada o página de documentación).
- Integridad del índice: metadatos (formato, modelo y dimensión de embeddings, fecha) y
  tabla de fuentes con tipo, versión y fecha de indexado. Se rechazan modelos o
  dimensiones incompatibles antes de insertar o consultar, con mensajes accionables.
- `api sources` muestra la versión indexada frente a la instalada y el estado de cada
  fuente; `api ask` avisa de las fuentes obsoletas y las marca en el panel de fuentes.
- Comando `api update`: reindexa los paquetes cuya versión instalada ha cambiado.

### Cambiado

- Los docstrings largos de las APIs ya no se truncan a 2000 caracteres: se reparten en
  varios fragmentos respetando las secciones (`Parameters`, `Returns`, `Examples`…) y
  los párrafos. Las continuaciones se etiquetan `nombre [2]` y repiten la firma. Hay que
  reindexar los paquetes (`api index <paquete>`) para beneficiarse.

- La versión de un paquete se resuelve por su distribución instalada (`yaml` → PyYAML),
  no por el nombre del módulo; la biblioteca estándar usa la versión de Python.
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
