# Importar bibliotecas necesarias
# Importar bibliotecas necesarias
import pandas as pd  # Pandas es una biblioteca para manipulación y análisis de datos. Proporciona estructuras de datos como DataFrame para manejar datos de manera eficiente.
import numpy as np  # NumPy es una biblioteca para realizar cálculos numéricos eficientes en Python, con estructuras de datos como arrays para operaciones matemáticas y estadísticas.
from sklearn.model_selection import train_test_split  # Esta parte de Scikit-learn ofrece herramientas para dividir datos en conjuntos de entrenamiento y prueba.
from sklearn.linear_model import LogisticRegression  # Esta parte de Scikit-learn proporciona el modelo de regresión logística, útil para tareas de clasificación.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Estas funciones de Scikit-learn se utilizan para evaluar modelos de aprendizaje automático.
import matplotlib.pyplot as plt  # Matplotlib es una biblioteca de visualización de datos que permite crear gráficos y visualizar resultados, como la matriz de confusión.


# Ruta al archivo CSV
ruta_csv = "C:\\Users\\wilme\\Pictures\\actividad 3\\transporte_masivo.csv"

# Cargar datos desde un archivo CSV
datos = pd.read_csv(ruta_csv)