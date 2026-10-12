# Datos

## Fuente

| Campo | Valor |
|---|---|
| Nombre | Multilingual Amazon Reviews Corpus (MARC), partición en español |
| Réplica utilizada | [`SetFit/amazon_reviews_multi_es`](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es), commit `16015418b488c9186fce74b058877ea939ca934d` |
| Dataset original | `amazon_reviews_multi` (Keung et al., 2020). Ya no se distribuye desde el Hub: su ficha está en [`defunct-datasets/amazon_reviews_multi`](https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi) |
| Responsable | Amazon (corpus original); SetFit (réplica de 4 columnas en el Hub) |
| Licencia | Licencia propia de Amazon: uso para **investigación académica**, no comercial; **prohíbe revender o republicar** el corpus y reidentificar a los autores |
| Fecha de consulta | 2026-10-11 |
| Idioma | Español (reseñas de amazon.es) |
| Particiones | train 200 000 · validation 5 000 · test 5 000 |
| Partición evaluada | `test` oficial completa: 5 000 reseñas |

## Por qué no se publica `dataset.csv`

La licencia del corpus prohíbe republicarlo. Por eso:

- `data/processed/dataset.csv` (con el texto de las reseñas) **se genera localmente** con `python src/prepare_data.py` y está excluido en `.gitignore` y en el ZIP de entrega.
- `data/processed/dataset_manifest.csv` **sí se publica**: contiene solo `review_id`, `stars` y `sentiment`, para verificar que se evalúan exactamente los mismos registros.
- `data/processed/data_profile.json` registra parámetros y estadísticas (distribuciones, duplicados, longitudes en tokens).

Para regenerar el archivo completo:

```bash
python src/prepare_data.py
```

## Diccionario de datos

| Campo | Tipo | Descripción | Ejemplo | Tratamiento |
|---|---|---|---|---|
| `id` (original) → `review_id` | texto | Identificador de la reseña en MARC | `es_0038754` | Renombrado; se verifica que sea único |
| `text` | texto | Cuerpo de la reseña escrito por el cliente | _(no se reproduce)_ | Sin limpieza; se trunca a 256 tokens solo en inferencia (afecta a < 0,3 %) |
| `label` (original) | entero 0–4 | Calificación del cliente menos 1 | `0` | Se transforma a `stars`; no se conserva |
| `label_text` (original) | texto | `label` como cadena | `"0"` | Solo para validar que coincide con `label`; no se conserva |
| `stars` | entero 1–5 | Estrellas otorgadas (`label + 1`) | `1` | Generado |
| `sentiment` | categórico | Clase objetivo: `negativo` (1–2★), `neutral` (3★), `positivo` (4–5★) | `negativo` | Generado; es la variable a predecir |
| `sentiment_id` | entero 0–2 | Código de `sentiment` | `0` | Generado (solo en `dataset.csv`) |
| `n_chars` | entero | Longitud en caracteres | `103` | Generado (análisis exploratorio) |
| `n_words` | entero | Longitud en palabras | `21` | Generado (análisis exploratorio) |
| `pred_<modelo>` | categórico | Predicción normalizada de cada modelo | `positivo` | Generado en `results/*/predictions.csv` |

## Preparación (resumen de `data_profile.json`)

- Sin valores faltantes ni textos vacíos; IDs únicos.
- 6 filas con texto duplicado (1 par con etiquetas distintas). **Se conservan** para no alterar la partición oficial; su efecto máximo sobre las métricas es de 0,12 puntos porcentuales.
- Estrellas: 1 000 por nivel. Sentimiento: negativo 2 000 (40 %), neutral 1 000 (20 %), positivo 2 000 (40 %).
- Longitud media ≈ 40 tokens; p95 ≈ 100 tokens.
- No se realiza muestreo: se evalúa la partición completa, con semilla 42 para la muestra de 200 reseñas de la prueba de latencia en línea.

## Limitaciones y posibles sesgos

- Reseñas de amazon.es: predomina el español peninsular; puede no representar el español de México u otras variantes.
- Solo productos de Amazon; otros dominios (servicios, banca, restaurantes) tienen vocabulario distinto.
- La agrupación 3★ = neutral es una convención: muchas reseñas de 3★ contienen opiniones mixtas, y las de 2★ y 4★ pueden ser ambiguas.
- Las estrellas son la autoevaluación del cliente, no una anotación de sentimiento por expertos: el texto y la calificación pueden no coincidir.
- Es posible que algún modelo haya visto reseñas similares (o del mismo corpus) durante su entrenamiento; los autores no lo documentan.
