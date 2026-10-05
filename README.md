# Laboratorio 8 — Deep Learning y Sistemas Inteligentes

Implementación del Laboratorio 8 del curso **CC3092 — Deep Learning y Sistemas Inteligentes**. El laboratorio estudia un Mini-GPT a nivel de carácter con mecanismos de muestreo controlado, y compara embeddings contextuales de BERT con embeddings estáticos de Word2Vec.

## Contenido

El notebook está organizado en los siguientes bloques:

1. Mini-GPT a nivel de carácter (Task 1):
   - Pipeline de datos sobre `tiny_shakespeare`: vocabulario de 65 caracteres, split 90/10 respetando el orden, batches con objetivo desplazado.
   - Decoder-only Transformer entrenado 10 épocas. Perplejidad de validación ≈ 7.65 (rango esperado 3.5–8.0).
2. Muestreo controlado (Task 2):
   - Greedy, top-k con temperatura y top-p con temperatura, implementados desde cero.
   - Función `generate()` y las 25 muestras requeridas (5 configuraciones × 5 muestras de 200 caracteres desde `"ROMEO:"`).
   - Análisis escrito que cita las muestras generadas.
3. Embeddings contextuales vs. estáticos (Task 3):
   - Embeddings de `bert-base-multilingual-cased` para `banco`, `copa` y `vela` en 7 oraciones.
   - Similitud coseno entre sentidos: entre 0.5667 y 0.7999 en BERT.
   - Comparación contra Word2Vec preentrenado en español (SBWC): 1.0000 en todos los pares.
   - PCA 2D de los embeddings de BERT.
4. Preguntas de análisis (Task 4): máscara causal en Language Modeling y el paradigma de preentrenar y adaptar.

## Requisitos

- Python 3.10 o superior.
- Dependencias enumeradas en `requirements.txt`, incluyendo PyTorch,
  Transformers, scikit-learn, gensim y ReportLab.

## Instalación

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución

Con el entorno virtual activado, inicie Jupyter:

```powershell
jupyter notebook
```

Abra `lab8.ipynb`, seleccione el kernel de la `.venv` y ejecute las celdas en orden.

También puede reproducir todas las salidas y construir el informe final con:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_notebook.py
.\.venv\Scripts\python.exe .\scripts\build_report.py
```

El entrenamiento del Mini-GPT corre en CPU (≈ 17–20 min para las 10 épocas) y no requiere GPU. La primera ejecución de la sección de BERT descarga `bert-base-multilingual-cased` desde Hugging Face (≈ 700 MB).


## Estructura

```text
.
├── lab8.ipynb
├── data/
│   └── input.txt
├── loss_curve.png
├── bert_pca.png
├── output/
│   └── pdf/
│       └── CC3092_Laboratorio_8_Entrega_Final.pdf
├── scripts/
│   ├── build_report.py
│   ├── patch_notebook.py
│   ├── run_notebook.py
│   └── sync_notebook_analysis.py
├── requirements.txt
├── README.md
└── .gitignore
```
