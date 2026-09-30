# Laboratorio 5 — Deep Learning y Sistemas Inteligentes

Implementación del Laboratorio 5 del curso **CC3092 — Deep Learning y Sistemas Inteligentes**. El laboratorio estudia funciones de pérdida, optimización y regularización mediante implementaciones manuales con tensores de PyTorch.

## Contenido

El notebook está organizado en los siguientes bloques:

1. Funciones de pérdida para clasificación:
   - Entropía cruzada categórica.
   - Entropía cruzada binaria.
   - Gradiente manual de la entropía cruzada.
   - Comparación entre CE estándar y CE ponderada.
2. Funciones de pérdida para regresión:
   - MSE.
   - MAE.
   - Huber loss.
   - Comparación experimental con datos atípicos.
3. Optimizadores: SGD, momentum y Adam.
4. Regularización: L2, dropout y early stopping.
5. Preguntas de análisis sobre los resultados experimentales.


## Requisitos

- Python 3.10 o superior.
- PyTorch.
- NumPy.
- Matplotlib.
- Jupyter Notebook o JupyterLab.

## Instalación

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib notebook ipykernel
```

## Ejecución

Con el entorno virtual activado, inicie Jupyter:

```powershell
jupyter notebook
```

Abra `S8 - Lab5_Semana8_Estudiante.ipynb`, seleccione el kernel de la `.venv` y ejecute las celdas en orden.

## Estructura

```text
.
├── S8 - Lab5_Semana8_Estudiante.ipynb
├── regresion_perdidas.png
├── README.md
└── .gitignore
```

`regresion_perdidas.png` contiene la comparación de convergencia y predicciones de MSE, MAE y Huber sobre el dataset con outliers.

