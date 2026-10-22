# Desarrollo

[← README](../README.md)

## Entorno

```bash
git clone https://github.com/isidro-rodriguez/api-consultor
cd api-consultor
uv sync                       # crea .venv con las dependencias de desarrollo
uv run pre-commit install     # hooks de pre-commit y pre-push
uv run api setup              # la CLI desde el código fuente
```

En desarrollo, los comandos se ejecutan con `uv run api …`.

## Calidad

```bash
uv run ruff check . --fix && uv run ruff format .
uv run ty check               # tipado estricto
uv run pytest                 # tests + cobertura (mínimo 90 %)
uv run pre-commit run --all-files
```

- **pre-commit:** higiene de ficheros, `uv.lock` al día, `ruff` y `ty`.
- **pre-push:** `ruff`, `ty` y `pytest`.
- Los tests no usan Ollama (se simula, con embeddings sintéticos) ni tocan el índice o
  la carpeta de datos reales: corren en directorios temporales.
- Convenciones de código, mapa de módulos y reglas para agentes:
  [AGENTS.md](https://github.com/isidro-rodriguez/api-consultor/blob/main/AGENTS.md).

## Integración continua

`.github/workflows/ci.yml`, en cada push a `main` y en cada pull request:

- `ruff`, `ty` y `uv.lock` en Linux.
- Tests con cobertura en Linux y Windows con Python 3.12, 3.13 y 3.14.
- Construcción del wheel y el sdist, e instalación del wheel en un entorno limpio.

También se puede lanzar a mano desde *Actions → CI → Run workflow*
(`workflow_dispatch`) o con `gh workflow run ci.yml --ref main`.

Dependabot propone cada semana actualizaciones de dependencias (`uv`) y de las acciones.

## Publicar una versión

1. Sube `version` en `pyproject.toml` (`uv version --bump patch|minor|major`).
2. Mueve lo de «Sin publicar» a la nueva versión en
   [CHANGELOG.md](https://github.com/isidro-rodriguez/api-consultor/blob/main/CHANGELOG.md).
3. Commit y `git push`; espera a que el **CI de `main` esté en verde en Linux y
   Windows** (`gh run watch` o la pestaña *Actions*).
4. Solo entonces etiqueta: `git tag -a vX.Y.Z -m vX.Y.Z && git push --tags`.

Etiquetar antes de ver el CI obliga a mover la etiqueta si falla (como en 0.4.0). Si
aun así ocurre y la versión **no** llegó a PyPI, corrige en `main`, borra la etiqueta
(`git push --delete origin vX.Y.Z`, `git tag -d vX.Y.Z`) y vuelve a crearla; si ya
está publicada, la corrección sale en la siguiente versión de parche.

`publish.yml` comprueba que la etiqueta coincide con la versión, ejecuta el CI y
publica en PyPI mediante **publicación de confianza** (OIDC, sin tokens guardados).

Configuración necesaria (una sola vez):

- **PyPI:** publicador de confianza del proyecto `api-consultor` con propietario
  `isidro-rodriguez`, repositorio `api-consultor`, flujo `publish.yml` y entorno
  `pypi`.
- **GitHub:** entorno `pypi` (Settings → Environments).

Si la publicación falla con `invalid-publisher`, algún dato del publicador de PyPI no
coincide con los del token (el registro del job los muestra). Corrígelo y relanza el
job: no hace falta otra etiqueta.

Versionado [SemVer](https://semver.org/lang/es/) y commits convencionales (`feat:`,
`fix:`, `docs:`…).
