# Evaluar la recuperación

[← README](../README.md)

`api eval` lanza preguntas de referencia contra el índice real y comprueba si las
fuentes esperadas aparecen entre los primeros resultados. No llama al modelo de chat,
pero sí a Ollama para los embeddings de las preguntas.

```bash
api eval evals/stdlib.toml          # muestra solo los casos que fallan
api eval evals/stdlib.toml -v       # todos los casos
api eval evals/stdlib.toml -k 5     # fragmentos recuperados por pregunta (por defecto 10)
```

## Métricas

- **R@k (Recall@k):** fracción de fuentes esperadas presentes entre los `k` primeros.
- **MRR (Mean Reciprocal Rank):** media de `1/posición` de la primera fuente correcta;
  1,0 es lo ideal.

Se calculan para el total y por categoría, solo con las preguntas que tienen respuesta.

## Abstención

Lo peor de un consultor no es no encontrar algo, sino responder con seguridad a una
pregunta que el índice no cubre (una función inexistente, un tema ajeno). Para medirlo,
marca esas preguntas con `expected = []`. `api eval` añade entonces una sección
**Abstención** que cuenta, para el `max_distance` configurado:

- **responde X/N sin respuesta:** respuestas falsas, la cifra que hay que bajar;
- **calla Y/M con respuesta:** preguntas buenas que el umbral dejaría sin responder.

Y propone un **umbral sugerido**: el que mejor separa ambos grupos (a igualdad,
el de menos respuestas falsas), con una línea lista para `config.toml`. Necesita
varias preguntas de cada tipo; con pocas, tómalo como orientación. La recuperación se
mide siempre sin abstención, así que R@k no cambia al probar umbrales.

Las preguntas que coinciden por nombre con algún símbolo indexado se responden con
cualquier umbral (aparecen como «nombre» en la tabla): el umbral no puede detenerlas.

## Formato de los casos

Un TOML con tablas `[[case]]`:

```toml
[[case]]
category = "exacto"                              # opcional; por defecto "general"
question = "¿Cómo uso json.dumps con indentación?"
expected = ["json.dumps|library/json.txt#*"]

[[case]]
category = "sin-respuesta"
question = "¿Qué hace pathlib.Path.teleport?"
expected = []                                    # el índice no lo responde
```

Cada elemento de `expected` admite:

- nombres exactos (`json.dumps`);
- comodines `*` (`library/json.txt#*`: cualquier sección de esa página);
- alternativas con `|` (basta con que acierte una), para aceptar la API indexada o su
  página de documentación.

Si alguna fuente esperada no está en el índice, `api eval` lo avisa: ese caso no puede
acertar. El ejemplo
[evals/stdlib.toml](https://github.com/isidro-rodriguez/api-consultor/blob/main/evals/stdlib.toml)
indica qué indexar antes de usarlo.

## Cuándo usarlo

Antes y después de cada cambio en la recuperación, el troceado o la configuración de
`[retrieval]`, para comprobar que la mejora es real. En desarrollo, los tests usan
además un banco sintético (`tests/test_benchmark.py`) con suelos de no regresión que
no necesita Ollama.
