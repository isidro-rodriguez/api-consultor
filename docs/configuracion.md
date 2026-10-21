# Configuración

[← README](../README.md)

Los valores por defecto están en `src/api_consultor/config.py`. Los tuyos van en un
`config.toml` opcional que sobrescribe solo lo que indiques. `api config` muestra los
valores efectivos (marcando los tuyos) y, si el fichero no existe, ofrece crearlo con
una plantilla comentada.

## Ubicación de los ficheros

| | Windows | Linux | macOS |
|---|---|---|---|
| `config.toml` | `%APPDATA%\api-consultor` | `~/.config/api-consultor` (o `$XDG_CONFIG_HOME`) | `~/Library/Application Support/api-consultor` |
| Datos (índice, `docs/`, `docs-cache/`) | `%LOCALAPPDATA%\api-consultor` | `~/.local/share/api-consultor` (o `$XDG_DATA_HOME`) | `~/Library/Application Support/api-consultor` |

La variable de entorno `API_CONSULTOR_DATA` sustituye la carpeta de datos completa
(útil para tener índices separados o para pruebas).

## Opciones

```toml
[models]
chat = "api-consultor"     # modelo de chat de Ollama (el creado por `api setup`)
embed = "bge-m3"           # modelo de embeddings
embed_dim = 1024           # dimensión de sus vectores

[paths]
db = "…/consultor.db"      # base de datos del índice
docs = "…/docs"            # carpeta de documentación de `api index-docs`

[retrieval]
k = 5                      # fragmentos enviados al modelo
candidates = 30            # candidatos por método antes de fusionar
max_per_symbol = 2         # máximo de fragmentos de un mismo símbolo
# max_distance = 0.6       # umbral de abstención; sin definir = desactivado

[display]
width = 90                 # ancho máximo de la salida

[links]
url = "{uri}"              # plantilla de enlace de las fuentes
```

- Un fichero inválido (sección o clave desconocida, tipo incorrecto) se rechaza con un
  mensaje claro antes de ejecutar nada.
- **Cambiar `embed` o `embed_dim` invalida el índice**: la CLI se niega a mezclar
  vectores incompatibles y explica cómo reconstruirlo (borrar la base y reindexar).
- `max_distance` se calibra mirando las distancias de `api ask --debug` con preguntas
  que sí y que no deberían responderse.
- Subir `k` da más contexto al modelo a costa de memoria y velocidad; con 8 GB de VRAM
  y contexto de 8k tokens, 5 fragmentos de ~2000 caracteres es un buen equilibrio.

## Enlaces a las fuentes

El panel «Fuentes» enlaza cada fragmento con su fichero. Por defecto es un `file://`,
que se abre con la aplicación que tu sistema asocie al tipo de fichero. Para usar un
editor concreto, define una plantilla con estos campos:

| Campo | Contenido |
|---|---|
| `{file}` | ruta del fichero tal cual |
| `{path}` | ruta con `/` y codificada para URL |
| `{uri}` | URI `file://` (por defecto) |
| `{line}` | línea (mínimo 1) |

```toml
[links]
url = "vscode://file/{path}:{line}"        # VS Code
```

## Modelo de chat

El prompt de sistema y los parámetros del modelo (contexto, temperatura…) están en
`src/api_consultor/Modelfile`, que se incluye en el paquete. Tras modificarlo (o tras
actualizar a una versión que lo cambie), recrea el modelo con `api setup`.
