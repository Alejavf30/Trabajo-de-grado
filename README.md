# Trabajo-de-grado
Repositorio del trabajo de grado para el desarrollo de un IPS basado en BLE con ML-IL y fingerprinting. La estructura del repositorio está organizada en tres carpetas principales: 

1. Códigos nodos (ESP32): Contiene los scripts desarrollados en Arduino IDE para programar los nodos fijos y el nodo móvil, configurados en modo advertising y scanner. Estos nodos se encargan de la transmisión y captura de la señal RSSI necesaria para la localización.
   
2. Modelo LSTM: Carpeta que almacena el código en Python utilizado para entrenar el modelo LSTM y evaluarlo con la métrica del RMSE.
   
3. Código Raspberry Pi 3: Incluye el script en Python necesario para la recepción de datos vía puerto serial, preprocesamiento mediante filtro de Kalman y escalado, así como la carga del modelo LSTM para realizar la estimación de coordenadas. También contiene los archivos necesarios del scaler y el modelo entrenado del presente trabajo de grado, bajo condiciones de BS y no BS.
 
