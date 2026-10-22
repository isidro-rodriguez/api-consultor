# Changelog

Todos los cambios relevantes de este proyecto se documentan aquí.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el
proyecto usa [Versionado Semántico](https://semver.org/lang/es/).

## [Sin publicar]

### Corregido

- Un subpaquete cuyo `__init__` hace `raise SystemExit` abortaba todo el indexado
  (`pkgutil.walk_packages` no lo captura); ahora se omite con un aviso.
- La sonda de `--python` ya no puede quedarse colgada: se cancela a los 10 minutos y
  recibe la entrada estándar vacía (un `input()` al importar no la bloquea).
- `api index-docs` con una carpeta `foo/` y un `foo.zip` (o una carpeta `docs/` y
  ficheros sueltos) sustituía una fuente por otra en silencio; ahora da un error.
- Un `.zip` dañado de `api index-docs` da un error legible en vez de una traza.
- `api ask` o `api sources` durante un `api index` en marcha podían fallar con
  «database is locked»: abrir el índice ya no escribe si no hay nada que cambiar, el
  índice usa modo WAL (lectores y escritor no se bloquean) y la espera máxima por un
  cerrojo sube a 30 s.

### Añadido

- Las preguntas en español sin nombres de símbolos se traducen al inglés con el modelo
  de chat antes de buscar (los docstrings están en inglés): se busca con ambas
  versiones y se responde a la original. `[retrieval] translate = false` lo desactiva;
  `api ask --debug` muestra la traducción.
- `api ask` no responde si la pregunta nombra un símbolo cualificado que no existe
  (`pathlib.Path.teleport`, `json.compress`) y su dueño sí está indexado, aunque no
  haya `max_distance`; antes respondía con el símbolo más parecido.
- `api eval` mide la abstención: las preguntas con `expected = []` (sin respuesta en
  el índice) cuentan respuestas falsas y perdidas con el `max_distance` actual y se
  sugiere el umbral que mejor separa ambos grupos (callar una pregunta cuya fuente no
  se recuperó no cuenta como respuesta perdida). `evals/stdlib.toml` trae seis.
- Límites de tamaño en `api index-docs` (20 MiB por fichero, 512 MiB y 20 000
  ficheros por fuente) contra bombas zip y carpetas equivocadas.

### Cambiado

- Una palabra corriente («tree», «directory») ya no cuenta como nombre corto de un
  símbolo salvo que parezca un identificador o sea rara en el índice: antes ponía
  `rich.tree.Tree` por delante de `shutil.copytree`.
- `api eval` mide Recall@k y MRR sin aplicar `max_distance` (antes una abstención
  contaba como fallo de recuperación) y solo con las preguntas que tienen respuesta.
- El mensaje al modelo separa datos e instrucciones: fragmentos en `<docs>`, pregunta
  en `<question>` (con sus etiquetas escapadas) y reglas al final. El `Modelfile`
  trata la documentación recuperada como datos no confiables: ejecuta `api setup`
  para recrear el modelo `api-consultor`.
- El CI se puede lanzar a mano (`workflow_dispatch`) y `docs/desarrollo.md` pide
  verlo en verde en `main` antes de etiquetar.
- La documentación advierte de que indexar un paquete ejecuta su código y de qué API
  dinámica no se ve.

## [0.4.0] - 2026-10-22

### Añadido

- `api ask --source/-s FUENTE` (repetible) limita la búsqueda a esas fuentes; acepta
  el nombre de la fuente, el del paquete (todas sus fuentes) o `stdlib`.
- Caché de embeddings: al reindexar (`api sources -u`, `api index`, `api index-docs`)
  solo se calculan los fragmentos nuevos o cambiados; el resto reutiliza su vector.
- Si un paquete no está en el entorno de la herramienta pero sí en el venv activo o en
  `./.venv`, el aviso sugiere el comando exacto (`api index X -p .venv`).

### Cambiado

- Formato de índice 5 (`chunks.hash`, `sources.module`); un índice anterior se migra
  solo al abrirlo, sin reindexar.
- El mismo paquete indexado desde otro entorno se guarda como otra fuente
  (`zensical@reCocina`) en vez de sustituir la anterior. Tampoco sustituye ya una
  fuente de documentación con el mismo nombre.
- `render.py` pasa a ser el paquete `render/` (tema, ayuda de Typer, mensajes,
  respuesta y tablas); los parches de API privada de Typer quedan aislados en
  `render/typer_help.py` y la dependencia se acota a `typer<0.28`.
- README reducido a instalación (desde PyPI), uso, configuración y desarrollo; el
  detalle pasa a `docs/` (uso, configuración, funcionamiento, evaluación y desarrollo).

### Corregido

- `api index --python` fallaba con «La sonda no respondió correctamente (sin
  salida)» en paquetes que escriben por stdout al importarse (p. ej. el saludo de
  pygame / pygame-ce): la sonda desvía esa salida a stderr y solo emite el JSON.
- Un submódulo que hace `raise SystemExit` al importarse (como
  `pygame.examples.glcube` sin PyOpenGL) abortaba el indexado; ahora se omite
  con un aviso, igual que cualquier otro fallo de importación.
- Si la sonda no devuelve JSON y stderr está vacío, el error muestra la última
  línea de su salida en vez de «sin salida».
- Los avisos de `api index` salían pegados a la barra de progreso: la barra va ahora
  por la salida de error, como los avisos, y Rich los pinta por encima.
- `[display] width` de `config.toml` se aplica también a `--help`.
- Las etiquetas `<docs>`/`</docs>` dentro de un fragmento se escapan al enviarlo al
  modelo, para que no cierren su bloque de contexto.

### Eliminado

- `docs.import_docs` (sin uso desde que `api index-docs` indexa en su sitio).

## [0.3.0] - 2026-10-21

### Añadido

- `api sources --verbose` (`-v`) muestra, además de cada módulo de la biblioteca
  estándar, el entorno donde se indexó cada paquete.
- El `--help` de cada comando muestra un párrafo con más detalle; el listado de
  `api --help` mantiene una línea por comando.
- La ayuda y los avisos de uso de Typer salen en español (uso, paneles, `--help`,
  valores por defecto, obligatorio) y admiten el markup semántico del tema
  (`[info]`, `[muted]`, `[warning]`…).
- `-h` como alias de `--help`.

### Cambiado

- `api update` pasa a ser `api sources --update` (`-u`).
- `api sources --all` pasa a ser `--verbose` (`-v`).
- `api --version` usa `-v` (antes `-V`) y `api eval` solo admite `-k` (antes también
  `--k`).
- `api index-docs` indexa la carpeta indicada en su sitio, sin ofrecer copiarla a la
  del usuario.
- Errores y avisos se muestran en paneles «Error» y «Aviso» con el borde del tema
  (como los errores de uso de Typer), sin símbolos, y la pista va tras una línea en
  blanco.
- Las tablas de fuentes justifican las versiones a la derecha y todos los estados
  llevan símbolo separado del texto (`✔ al día`, `○ no instalada`, `· docs`…).
- El test de sondas inutilizables ya no depende de un guion con shebang: se ejecuta
  también en Windows.

### Eliminado

- `api index-docs --copy` / `--no-copy`.
- `api sources --yes` (`-y`): eliminar o reindexar fuentes siempre pide confirmación.

## [0.2.0] - 2026-10-21

### Añadido

- Metadatos para PyPI (licencia MIT, autor, clasificadores, URLs), `LICENSE` y
  `py.typed`; el CI valida el paquete con `twine check --strict`.
- CI de GitHub Actions: calidad (`ruff`, `ty`, `uv.lock`), tests en Linux y Windows con
  Python 3.12 y 3.13, y comprobación del wheel en un entorno limpio. Flujo de
  publicación en PyPI por etiqueta `vX.Y.Z` (publicación de confianza) y Dependabot
  semanal para dependencias y acciones.
- Instalable como herramienta global (`uv tool install`). El `Modelfile` viaja dentro
  del paquete y el nuevo comando `api setup` descarga los modelos y crea `api-consultor`
  (idempotente).
- `api index --python ENTORNO`: indexa paquetes instalados en el venv de un proyecto
  (no solo en el de la herramienta). Se hace con una sonda que corre con el Python de
  ese entorno; cada fuente recuerda su intérprete y `api update` y el aviso de versión
  obsoleta lo respetan.
- `api sources --relocate FUENTE CARPETA` recoloca una fuente movida sin reindexar.
- `api sources` marca como «no encontrada» una fuente cuyo venv o carpeta ya no existe;
  el panel de `api ask` indica «fichero no disponible» y no pone enlace.
- `api index-docs` ofrece copiar la documentación a la carpeta del usuario
  (`--copy` / `--no-copy` para no preguntar).
- `api index` también indexa las propiedades (`property` y `cached_property`) con
  docstring, con su fichero y línea.
- Los miembros heredados de clases base *privadas* del mismo paquete (p. ej.
  `pathlib.PurePath.suffix`, definido en `pathlib._abc.PurePathBase`) se indexan en la
  primera clase pública que los hereda. Los heredados de bases públicas no se duplican.
- Casos de `evals/stdlib.toml` sobre atributos y métodos heredados (`Path.suffix`…).
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

- La documentación local se busca por defecto en `<carpeta de datos>/docs` (antes
  `./docs`). Para el comportamiento anterior: `[paths] docs = "docs"` en `config.toml`.
- Formato de índice 4: las rutas de los fragmentos son relativas a la raíz de la fuente
  (`sources.root`). Un índice anterior se migra solo al abrirlo.
- La base de datos se guarda por defecto en la carpeta de datos del usuario, no en
  `data/` del proyecto. Para conservar la anterior: moverla allí o fijar
  `[paths] db = "data/consultor.db"` en `config.toml`.
- Docstrings en estilo Google en toda la API pública (constantes incluidas); ruff exige
  docstrings en lo público (`D`, convención Google).
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

- Títulos o nombres con corchetes (`[tool.ruff]`) rompían las tablas de Rich al
  interpretarse como estilos; ahora se imprimen literalmente.
- El contexto enviado al modelo escapa los atributos XML (comillas, `<`, `&`).
- `api sources --remove` contaba dos veces un nombre repetido en la confirmación.
- `api index` ya no se cae con paquetes que fallan al importarse o con nombres inválidos:
  avisa y sigue con el siguiente.
- Las rutas de `config.toml` admiten `~`.
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
