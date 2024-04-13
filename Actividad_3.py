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

# Preprocesamiento de datos
# Las características son distancia, tiempo y costo del viaje
X = datos[['distancia', 'tiempo', 'costo']]  # Características

y = datos['transporte']  # Etiqueta (tipo de transporte masivo)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)


# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
print('Matriz de confusión:')
print(confusion_matrix(y_test, y_pred))

print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred))

print('\nPrecisión del modelo:', accuracy_score(y_test, y_pred))

# Visualización de resultados (opcional)
# Graficar la matriz de confusión
plt.figure()
plt.imshow(confusion_matrix(y_test, y_pred), cmap='Blues')
plt.title('Matriz de confusión')
plt.colorbar()
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas reales')
plt.show()

tiempo_promedio_por_transporte = datos.groupby('transporte')['tiempo'].mean()

# Mostrar el tiempo promedio de viaje para cada tipo de transporte
print("\n\nTiempo promedio de viaje por tipo de transporte:")
print(tiempo_promedio_por_transporte)

# Determinar el tipo de transporte más rápido
transporte_mas_rapido = tiempo_promedio_por_transporte.idxmin()
print("\nEl medio de transporte más rápido es:", transporte_mas_rapido)