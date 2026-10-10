# Laboratorio 9 — Deep Learning y Sistemas Inteligentes

Implementación del Laboratorio 9 del curso **CC3092 — Deep Learning y Sistemas Inteligentes**. El laboratorio implementa desde cero, con tensores de PyTorch, dos familias de modelos generativos sobre Fashion-MNIST: el autoencoder variacional (VAE) y el proceso forward de un modelo de difusión. Esta versión contiene la **entrega parcial (Tasks 1 y 2)**.

## Contenido

El notebook `lab9.ipynb` está organizado así:

1. VAE (Task 1):
   - Datos: Fashion-MNIST en [0, 1], con los últimos 5 000 ejemplos de entrenamiento como validación.
   - VAE convolucional con reparametrización, KL cerrada y pérdida $\mathcal L_\beta$ escritas a mano, entrenado para β ∈ {0.1, 1, 10}.
   - Figuras: curvas de validación, dispersión de μ, cuadrícula 15×15, muestras del prior e interpolaciones.
   - Análisis:
     - KL en bits frente a log₂10.
     - Verificación Monte Carlo de la KL (error relativo 0.096 %, 1.6 errores estándar).
     - Regiones vacías del espacio latente.
     - Nitidez de las muestras y relación σ_x² = β/2.
2. Proceso forward de difusión (Task 2):
   - Calendario lineal (T = 1000, β de 1e-4 a 0.02) y forma cerrada `q_sample`.
   - Verificación paso a paso con 5 000 copias durante 300 pasos.
   - SNR(t): el primer t con SNR < 1 es 260, y el primer t con ᾱ_t < 0.01 es 674.
   - Calendario propio: coseno de Nichol & Dhariwal (2021) con piso ᾱ_min = 1e-4.
   - Análisis:
     - Aproximación de ᾱ_T a mano.
     - Pasos útiles por calendario.
     - Varianza global de x_t.

Todas las respuestas de análisis están en celdas markdown, junto al código que produce sus números. Semilla: **42**.

## Requisitos

- Python 3.10 o superior.
- Dependencias en `requirements.txt` (PyTorch, torchvision, NumPy, Matplotlib, Jupyter).
- Opcional: GPU NVIDIA con CUDA. El notebook también corre en CPU, pero más lento.

## Instalación

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# PyTorch con CUDA (ajuste cu130 a la versión de su driver; omita --index-url para la versión CPU)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m ipykernel install --user --name lab9 --display-name "Python (.venv Lab9)"
```

## Ejecución

Con el entorno virtual activado:

```powershell
jupyter notebook
```

Abra `lab9.ipynb`, seleccione el kernel **Python (.venv Lab9)** y ejecute las celdas en orden. Para ejecutarlo de principio a fin sin abrir el navegador:

```powershell
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=lab9 lab9.ipynb
```

La primera ejecución descarga Fashion-MNIST en `data/`. En una RTX 4050, todo el notebook tarda unos 4 minutos, de los cuales unos 65 s son por cada VAE.

## Estructura

```text
.
├── lab9.ipynb               # notebook con código, figuras y respuestas
├── cc3045-laboratorio-9.md  # enunciado
├── figuras/                 # figuras exportadas por el notebook (para el informe PDF)
├── requirements.txt
├── README.md
├── data/                    # Fashion-MNIST (descargado, ignorado por git)
└── checkpoints/             # state_dict de los VAE (ignorado por git; se reutilizan en la Task 4)
```
