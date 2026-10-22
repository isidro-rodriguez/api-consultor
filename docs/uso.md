# Uso detallado

[← README](../README.md)

Todos los comandos aceptan `-h`/`--help`. `api -V` muestra la versión.

## `api setup`

Prepara Ollama: descarga `qwen3:8b` y `bge-m3` si faltan, crea el modelo de chat
`api-consultor` a partir del `Modelfile` incluido en el paquete y ofrece indexar la
biblioteca estándar. Es idempotente: repítelo tras actualizar la herramienta.

```bash
api setup               # pregunta si indexar la stdlib (por defecto, sí)
api setup --no-stdlib   # sin preguntar
```

## `api index`: paquetes Python

Importa cada paquete y guarda un fragmento por módulo, clase, función y propiedad
pública, con firma, docstring, fichero y línea. Reindexar un paquete sustituye la
versión anterior.

```bash
api index requests rich         # entorno de la propia herramienta
api index pygame -p .venv       # entorno virtual de un proyecto
api index pygame -p C:\ruta\python.exe
api index --stdlib              # biblioteca estándar (≈200 módulos)
api index --stdlib -p .venv     # la stdlib de otra versión de Python
```

- **Qué entorno se usa.** Un paquete solo se puede indexar si está instalado en el
  entorno indicado. Sin `-p/--python` es el de la herramienta (instalada con `uv tool`),
  que normalmente **no** contiene las dependencias de tus proyectos. Con `-p` se indica
  la carpeta del venv o directamente su ejecutable de Python.
- **Cómo funciona `--python`.** Se lanza una *sonda* (`probe.py`, solo biblioteca
  estándar) con el intérprete del entorno; importa el paquete allí y devuelve los
  fragmentos en JSON. El entorno objetivo necesita Python 3.12 o superior. Si tarda
  más de 10 minutos (un import que se queda colgado) se aborta con un error.
- **Indexar ejecuta código.** Para leer la API hay que importar el paquete, y eso
  ejecuta su código (con o sin `--python`; `python -I` no es un *sandbox*). Indexa
  solo paquetes en los que confíes, igual que al instalarlos.
- **Nombre de cada símbolo.** Lo que se reexporta en la raíz del paquete se indexa con
  ese nombre (`rich.Console`); lo demás, en el módulo que lo define
  (`rich.console.Console`), aunque otros módulos lo importen.
- **Qué se omite.** Nombres privados (`_algo`), símbolos sin docstring y submódulos que
  fallan al importarse, incluso con `SystemExit` (se avisa y se continúa).
- **Fragmentos largos.** Un docstring de más de ~2000 caracteres se divide por secciones
  en varios fragmentos (`nombre [2]`, `nombre [3]`…), sin perder texto.
- **Versión.** Se resuelve por la distribución instalada (`yaml` → PyYAML); la de la
  stdlib es la de Python. Cada fuente recuerda su entorno para comprobarla después.
- **Mismo paquete, varios entornos.** Indexarlo desde otro entorno crea otra fuente
  (`zensical@reCocina`) y no sustituye la anterior; ver
  [identidad de las fuentes](funcionamiento.md#identidad-de-las-fuentes).
- **Reindexar es barato.** Solo se calculan los embeddings de los fragmentos nuevos o
  cambiados ([caché](funcionamiento.md#caché-de-embeddings)).
- **Paquete no encontrado.** Si no está en el entorno de la herramienta pero sí en el
  venv activo o en `./.venv`, el aviso indica el comando exacto
  (`api index zensical -p .venv`).

## `api index-docs`: documentación local

Indexa `.txt`, `.md` y `.rst`, sueltos, en carpetas o dentro de `.zip`.

```bash
api index-docs                    # carpeta de documentación del usuario
api index-docs ~/mis-docs         # otra carpeta, indexada en su sitio
api index-docs -x 'whatsnew/*'    # omitiendo rutas (patrón fnmatch, repetible)
```

- Cada carpeta o `.zip` del directorio es una **fuente**; los ficheros sueltos forman
  la fuente `docs`. Dos entradas con el mismo nombre (`foo/` y `foo.zip`) son un error.
- Límites por fuente: 20 MiB por fichero y 512 MiB o 20 000 ficheros de texto en total
  (descomprimidos); lo que los supera se rechaza antes de leerlo.
- El texto se trocea por secciones (encabezados) y cada fragmento guarda su fichero y
  línea.
- Los `.zip` se extraen (solo los textos) a `docs-cache/`, junto a la base de datos,
  para que sus fragmentos también se puedan abrir.
- Admite, por ejemplo, el zip de texto plano de la documentación de Python
  (<https://docs.python.org/3/download.html>).

## `api ask`: preguntar

```bash
api ask "¿cómo hago un POST con timeout?"
api ask                  # modo interactivo (salir o Ctrl+D para terminar)
api ask -d "..."         # --debug: candidatos, señales, distancias y tamaño del contexto
api ask -s requests "¿cómo pongo un timeout?"   # solo en esa fuente (repetible)
api ask -s stdlib -s mis-notas "..."            # toda la stdlib y una carpeta de docs
```

`-s/--source` acepta el nombre de una fuente (`api sources`), el de un paquete (todas
sus fuentes, si está indexado desde varios entornos) o `stdlib`.

La respuesta se genera **solo** con los fragmentos recuperados y los cita como
`[DOC-01]`. Debajo aparece el panel «Fuentes»:

- ✔ marca los fragmentos citados; se avisa si el modelo cita ids inexistentes o no
  cita nada.
- Cada fuente enlaza a su fichero y línea (hipervínculo de terminal, normalmente
  Ctrl+clic). Ver [enlaces](configuracion.md#enlaces-a-las-fuentes).
- Si un paquete indexado ya no coincide con la versión instalada, se avisa y se marca.
- Si el fichero ya no existe, se indica «fichero no disponible».
- Con un umbral `max_distance` configurado, el comando se abstiene cuando nada es
  suficientemente parecido (ver [funcionamiento](funcionamiento.md#abstención)).

## `api sources`: administrar el índice

```bash
api sources                               # listado con estado
api sources -v                            # cada módulo de la stdlib y su entorno
api sources -u                            # --update: reindexa las obsoletas
api sources -r requests -r rich           # --remove (repetible); `-r stdlib` borra toda la stdlib
api sources --relocate notas D:/docs/notas   # la carpeta de una fuente cambió de sitio
```

Estados de cada fuente:

| Estado | Significado | Qué hacer |
| --- | --- | --- |
| al día | versión indexada = instalada | nada |
| obsoleta | el paquete cambió de versión | `api sources -u` |
| no encontrada | se borró el venv o se movió la carpeta | reinstalar, `--relocate` o `--remove` |

- `--update` reindexa cada paquete **en el entorno donde se indexó**.
- `--relocate` actualiza la raíz de la fuente sin reindexar: el índice guarda las rutas
  relativas a esa raíz.
- `--remove` también borra la extracción de un `.zip` en `docs-cache/`.
- Eliminar y reindexar piden confirmación.
- La documentación no tiene versión: solo se comprueba que su carpeta exista.

## `api config`

Muestra cada ajuste con su valor efectivo, marca los que cambia tu `config.toml` y las
rutas en uso. Si el fichero no existe, ofrece crearlo. Ver
[configuración](configuracion.md).

## `api eval`

Mide la calidad de la recuperación. Ver [evaluación](evaluacion.md).
