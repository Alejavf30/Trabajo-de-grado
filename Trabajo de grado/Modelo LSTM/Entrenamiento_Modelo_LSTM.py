import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sys
import os

# Leer archivo .csv
if len(sys.argv) < 2:
    print("Uso: python entrenamiento_lstm.py <nombre_dataset.csv>")
    sys.exit(1)

csv_filename = sys.argv[1]
if not os.path.exists(csv_filename):
    print(f"Error: el archivo '{csv_filename}' no se encuentra en la carpeta.")
    sys.exit(1)

df = pd.read_csv(csv_filename)

# Aplicar el filtro Kalman por punto
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

filtered_dataframes = []
for point, group in df.groupby('Punto'):
    group = group.copy()
    for column in ['Nodo 1', 'Nodo 2', 'Nodo 3', 'Nodo 4']:
        group[column] = apply_kalman_filter(group[column].values)
    filtered_dataframes.append(group)

df_filtered = pd.concat(filtered_dataframes, ignore_index=True)

# Preparar los datos para el modelo LSTM
X = df_filtered[['Nodo 1', 'Nodo 2', 'Nodo 3', 'Nodo 4']].values
y = df_filtered[['X', 'Y']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# Redimensionar los datos  para LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Crear el modelo LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(256, input_shape=(1, X_train.shape[2]), return_sequences=True, kernel_regularizer=l2(0.001)))
lstm_model.add(Dropout(0.3))
lstm_model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
lstm_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
lstm_model.add(Dense(2))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo LSTM
history = lstm_model.fit(X_train, y_train, epochs=300, validation_split=0.2, verbose=1)

# Evaluar el modelo LSTM con el RMSE
y_pred = lstm_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE en el conjunto de prueba: {rmse:.4f}')

# Guardar el modelo y el scaler
lstm_model.save('lstm_model.keras')
print("Modelo LSTM guardado como 'lstm_model.keras'.")
print("Scaler guardado como 'scaler.pkl'.")

# Graficar la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.show()