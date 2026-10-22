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
  fragmentos en JSON. El entorno objetivo necesita Python 3.12 o superior.
- **Qué se omite.** Nombres privados (`_algo`), símbolos sin docstring y submódulos que
  fallan al importarse (se avisa y se continúa).
- **Fragmentos largos.** Un docstring de más de ~2000 caracteres se divide por secciones
  en varios fragmentos (`nombre [2]`, `nombre [3]`…), sin perder texto.
- **Versión.** Se resuelve por la distribución instalada (`yaml` → PyYAML); la de la
  stdlib es la de Python. Cada fuente recuerda su entorno para comprobarla después.

## `api index-docs`: documentación local

Indexa `.txt`, `.md` y `.rst`, sueltos, en carpetas o dentro de `.zip`.

```bash
api index-docs                    # carpeta de documentación del usuario
api index-docs ~/mis-docs         # otra carpeta, indexada en su sitio
api index-docs -x 'whatsnew/*'    # omitiendo rutas (patrón fnmatch, repetible)
```

- Cada carpeta o `.zip` del directorio es una **fuente**; los ficheros sueltos forman
  la fuente `docs`.
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
```

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
