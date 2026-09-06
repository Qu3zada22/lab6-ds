# Laboratorio 6 — Redes de participación en YouTube

Análisis reproducible de los ejercicios 1–10: calidad de datos, exploración,
red bipartita autor-video, proyecciones, comunidades, centralidad y sentimiento
en español. El notebook existente es la fuente del análisis y del informe PDF;
las relaciones representan participación observada, no conversaciones entre autores.

## Ejecución completa

Desde la raíz de este repositorio, con [uv](https://docs.astral.sh/uv/) instalado:

```bash
uv sync --frozen --python 3.11
uv run --frozen lab6-ds
uv run --frozen python -m unittest discover -s tests -v
```

El segundo comando ejecuta el notebook desde un kernel limpio, comprueba sus
invariantes, guarda las salidas y genera `results/Informe_Laboratorio6.pdf`.
Una ejecución fallida detiene el proceso: no se presenta un informe nuevo como
si el análisis hubiera terminado. Los archivos de una ejecución anterior pueden
seguir presentes; compruebe el estado de salida del comando.

Para regenerar únicamente el PDF de un notebook ejecutado y sin cambios de fuente:

```bash
uv run --frozen lab6-ds --report-only
```

Para explorar el notebook interactivamente: `uv run --frozen jupyter lab`.
Use el entorno del proyecto como kernel y ejecute las celdas en orden desde la raíz.

## Entorno y descargas

- Entorno verificado: Linux, Python 3.11, pandas 3.0.5 y NetworkX 3.6.1.
- `uv.lock` fija dependencias; `pyproject.toml` declara las directas.
  `requirements.txt` es una exportación del mismo lock, no otra selección manual.
  PyTorch utiliza el índice CPU `https://download.pytorch.org/whl/cpu`.
  Se recomienda `uv sync`: el formato requirements no conserva el aislamiento
  del índice por paquete y lo expresa como un índice adicional global.
- spaCy `es_core_news_sm` 3.8.0 se instala desde su distribución oficial de GitHub.
- Sentimiento: pysentimiento 0.7.3, Transformers 4.57.3, PyTorch 2.8.0 CPU;
  [RoBERTuito](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis),
  revisión `a2cc0f67ebd705c55191e25a05ba23d885fcc09b`.
  La primera inferencia descarga aproximadamente 435 MB de pesos desde Hugging
  Face; reserve espacio adicional para el entorno y la caché (varios GB).
  Se necesita acceso a PyPI, GitHub, el índice CPU y Hugging Face durante la
  instalación/primera ejecución. Después se reutiliza la caché local estándar
  de Hugging Face (`HF_HOME` permite cambiarla).
- No se requiere GPU, LaTeX, navegador ni API de pago. ReportLab genera el PDF
  desde las salidas reales del notebook, con gráficos, tablas e interpretación,
  sin incluir bloques de código. Los comentarios se procesan localmente.
- El análisis con modelo ya descargado tardó alrededor de 40 segundos en el
  entorno de verificación; el tiempo depende del equipo y la conexión.

## Archivos de entrega

| Archivo | Contenido |
| --- | --- |
| `Laboratorio6_Redes_Sociales_YouTube.ipynb` | Ejercicios 1–10, código, resultados y narrativa |
| `results/Informe_Laboratorio6.pdf` | Informe reproducible; tablas extensas resumidas para legibilidad |
| `results/nodes.csv`, `results/edges.csv` | Todos los nodos y aristas de la bipartita, tipos y atributos |
| `results/author_projection_edges.csv`, `results/video_projection_edges.csv` | Proyecciones y pesos por vecinos compartidos |
| `results/author_metrics.csv`, `results/video_metrics.csv`, `results/bridge_evidence.csv` | Centralidad, diversidad, articulaciones y evidencia |
| `results/comments_sentiment.csv` | Texto original preservado, texto limpio, entradas del modelo, etiquetas, probabilidades y auditoría |
| `results/video_sentiment.csv`, `results/channel_sentiment.csv`, `results/community_profiles.csv` | Comparaciones con tamaños y perfiles internos de comunidades |
| `results/video_coverage.csv`, `results/topology.csv` | Cobertura de los 293 videos y métricas de las tres redes |
| `results/word_frequencies.csv`, `results/bigram_frequencies.csv` | Frecuencias principales de contenido |
| `results/manifest.json` | SHA-256 de insumos, revisión del modelo y validaciones |
| `src/lab6_ds/analysis.py`, `src/lab6_ds/report.py` | Funciones verificables y ejecución/exportación del notebook |
| `tests/` | Pruebas unitarias e integración de los archivos exportados |

Los CSV usan identificadores estables, no nombres como llaves. En `nodes.csv`,
`tipo` distingue `autor` y `video`; los atributos no aplicables quedan vacíos.
El peso bipartito cuenta comentarios, mientras los pesos de proyección cuentan
videos o autores compartidos. Grado y fuerza no son distancias: la intermediación
y cercanía usan caminos no ponderados. `component_increase` verifica la eliminación
de cada articulación; `projection_articulation` distingue videos que articulan
audiencias de videos cuya eliminación solo aísla comentaristas.

## Decisiones y límites

La red observada tiene 332 autores y 19 videos (351 nodos, 343 aristas). Los otros
274 videos están en la tabla de cobertura: ausencia de comentarios recolectados
no equivale a ausencia real de participación. Hay 17 comunidades Louvain, con
modularidad 0.7774; esta cifra no valida grupos sociales por un umbral universal.
Los perfiles de las tres mayores usan 158, 48 y 34 comentarios **internos**,
no todos los comentarios de los videos asignados. Entradas y salidas se cuentan
por separado.

Sentimiento se calcula sobre el original, con el preprocesamiento oficial en una
columna separada, y se audita el truncamiento a 128 tokens. NEG/NEU/POS son
predicciones de tono, no aprobación del canal ni etiquetas humanas. El criterio
`n ≥ 10` para gráficas y la bandera de confianza `< 0.60` son exploratorios;
no prueban significación ni calibración. No hay validación manual local ni
generalización a YouTube o a la población de Guatemala.

Los identificadores y comentarios de los CSV proporcionados se conservan para
auditoría académica. Evite reutilizarlos para perfilar personas o atribuirles
posiciones políticas; revise permisos y licencias antes de otra difusión.

## Enlaces requeridos

- Repositorio: https://github.com/Qu3zada22/lab6-ds
- Espacio colaborativo del grupo: [Informe del Laboratorio 6](https://github.com/Qu3zada22/lab6-ds/blob/main/results/Informe_Laboratorio6.pdf).
  No se encontró en el repositorio y no se ha inventado un enlace.

Enunciado: `docs/Laboratorio_6_Analisis_de_redes_sociales_YouTube_2026.md`.
