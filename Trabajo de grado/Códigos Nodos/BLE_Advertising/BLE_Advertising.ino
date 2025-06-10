#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <esp_bt.h>

#define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// CAMBIAR ESTE NOMBRE EN CADA DISPOSITIVO
#define NOMBRE_DEL_NODO "Nodo 1"  // Cambiar a "Nodo 2", "Nodo 3", "Nodo 4" según el dispositivo

void setup()
{
    Serial.begin(115200);
    Serial.println("Iniciando advertising BLE...");

    // Inicializar BLE y asignar nombre
    BLEDevice::init(NOMBRE_DEL_NODO);

    // Aumentar potencia de transmisión (puedes ajustar si lo deseas)
    esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, ESP_PWR_LVL_P9);

    // Crear el servidor BLE
    BLEServer *pServer = BLEDevice::createServer();

    // Crear un servicio
    BLEService *pService = pServer->createService(SERVICE_UUID);

    // Crear una característica (aunque no la estés usando ahora)
    BLECharacteristic *pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_READ |
        BLECharacteristic::PROPERTY_WRITE);

    pCharacteristic->setValue("ESP32 Nodo BLE");

    // Iniciar servicio
    pService->start();

    // Comenzar advertising
    BLEAdvertising *pAdvertising = pServer->getAdvertising();
    pAdvertising->start();

    Serial.printf("Nodo %s anunciándose\n", NOMBRE_DEL_NODO);
}

void loop()
{
    delay(1000); // Nada más que hacer
}
