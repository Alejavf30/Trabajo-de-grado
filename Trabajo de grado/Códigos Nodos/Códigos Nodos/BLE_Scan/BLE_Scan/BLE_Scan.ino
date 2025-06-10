#include <BLEDevice.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>

int scanTime = 2; // duración del escaneo en segundos
BLEScan* pBLEScan;

class MyAdvertisedDeviceCallbacks : public BLEAdvertisedDeviceCallbacks {
    void onResult(BLEAdvertisedDevice advertisedDevice) {
        // Solo mostrar nodos que se llamen exactamente "Nodo 1", ..., "Nodo 4"
        std::string name = advertisedDevice.getName();
        if (name == "Nodo 1" || name == "Nodo 2" || name == "Nodo 3" || name == "Nodo 4") {
            int rssi = advertisedDevice.getRSSI();
            Serial.printf("Dispositivo: %s | RSSI: %d dBm\n", name.c_str(), rssi);
        }
    }
};

void setup() {
    Serial.begin(115200);
    Serial.println("Iniciando escaneo BLE...");
    
    BLEDevice::init("");
    pBLEScan = BLEDevice::getScan();
    pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
    pBLEScan->setActiveScan(true);  // Activo para obtener nombre y más info
    pBLEScan->setInterval(100);
    pBLEScan->setWindow(99); // debe ser menor o igual al intervalo
}

void loop() {
    BLEScanResults foundDevices = pBLEScan->start(scanTime, false);
    pBLEScan->clearResults(); // Limpia memoria entre escaneos
    delay(500); // Pequeña pausa entre escaneos
}
