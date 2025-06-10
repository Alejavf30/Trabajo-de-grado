import serial
import csv
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from tensorflow.keras.models import load_model
import os

# Configuración Raspberry Pi 3 para lectura por el puerto serial
puerto_serial = '/dev/ttyUSB0'  # Cambiar según el puerto serial que use el dispositivo
baud_rate = 115200 #Cambiar según la tasa que use el dispositivo

NUM_MEDICIONES = 50 #Cambiar según el número de mediciones que se tengan 
nodos = ['Nodo_1', 'Nodo_2', 'Nodo_3', 'Nodo_4']
regex = re.compile(r'Name:\s*(Nodo_\d),.?rssi:\s(-?\d+)')

# Creación archivo CSV
csv_filename = 'mediciones.csv'
header = ['Nodo 1', 'Nodo 2', 'Nodo 3', 'Nodo 4']

# Función para aplicar el filtro Kalman
def apply_kalman_filter(rssi_values):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([rssi_values[0]])
    kf.F = np.array([[1]])
    kf.H = np.array([[1]])
    kf.P *= 1000.
    kf.R = 8
    kf.Q = 1e-3

    filtered_values = []
    for value in rssi_values:
        kf.predict()
        kf.update(np.array([value]))
        filtered_values.append(kf.x[0])
    return np.array(filtered_values)

# Toma de datos de la ESP32
mediciones = []
medicion_actual = {}

with serial.Serial(puerto_serial, baud_rate, timeout=1) as ser:

    while len(mediciones) < NUM_MEDICIONES:
        linea = ser.readline().decode(errors='ignore').strip()

        if linea.startswith("Iniciando medición"):
            if len(medicion_actual) == 4:
                mediciones.append(medicion_actual)
            medicion_actual = {}

        # Para extraer el nodo y el RSSI de la salida de la ESP32
        match = regex.search(linea)
        if match:
            nodo = match.group(1)
            rssi = int(match.group(2))
            medicion_actual[nodo] = rssi

if len(medicion_actual) == 4:
    mediciones.append(medicion_actual)

# Para guardar el archivo CSV
with open(csv_filename, mode='w', newline='') as archivo_csv:
    writer = csv.DictWriter(archivo_csv, fieldnames=header)
    writer.writeheader()
    for fila in mediciones:
        writer.writerow({
            'Nodo 1': fila.get('Nodo_1', np.nan),
            'Nodo 2': fila.get('Nodo_2', np.nan),
            'Nodo 3': fila.get('Nodo_3', np.nan),
            'Nodo 4': fila.get('Nodo_4', np.nan)
        })

# Procesar la predicción con el modelo LSTM
df = pd.read_csv(csv_filename)

# Aplicar el filtro Kalman
for column in ['Nodo 1', 'Nodo 2', 'Nodo 3', 'Nodo 4']:
    df[column] = apply_kalman_filter(df[column].values)

# Calcular la media de las mediciones para que ingresen 4 datos al modelo
mean_rssi = df[['Nodo 1', 'Nodo 2', 'Nodo 3', 'Nodo 4']].mean().values.reshape(1, -1)

# Escalar los datos
scaler_filename = 'scaler.pkl' # Cambiar por el nombre del scaler 
if not os.path.exists(scaler_filename):
    print(f"Error: el archivo '{scaler_filename}' no se encuentra.")
    exit(1)
scaler = joblib.load(scaler_filename)
rssi_values_scaled = scaler.transform(mean_rssi)
rssi_values_scaled = rssi_values_scaled.reshape((1, 1, 4))

# Cargar el modelo anteriormente entrenado y realizar la predicción
model_filename = 'lstm_model.keras' # Cambiar por el nombre del modelo entrenado
if not os.path.exists(model_filename):
    print(f"Error: el archivo '{model_filename}' no se encuentra.")
    exit(1)
lstm_model = load_model(model_filename)
prediccion = lstm_model.predict(rssi_values_scaled)

# Guardar las coordenadas predichas en un archivo .txt
output_filename = 'prediccion_coordenadas.txt'
with open(output_filename, 'w') as f:
    f.write(f"Predicción de coordenadas (X, Y):\n")
    f.write(f"X: {prediccion[0][0]:.2f}\n")
    f.write(f"Y: {prediccion[0][1]:.2f}\n")

