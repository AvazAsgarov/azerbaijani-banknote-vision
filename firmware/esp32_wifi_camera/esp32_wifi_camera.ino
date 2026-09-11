#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiMulti.h>
#include "board_config.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// =======================================================================
// OVERRIDE BROWNOUT DETECTOR (Prevents USB voltage drop reset loop)
// =======================================================================
extern "C" void esp_brownout_init(void) {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
}

#include <ESPmDNS.h>

WiFiMulti wifiMulti;
void startCameraServer();
void setupLedFlash();

void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    delay(1000);
    Serial.println();
    Serial.println("==================================================");
    Serial.println(" AZN Vision: XIAO ESP32-S3 Wi-Fi Camera Streamer ");
    Serial.println("==================================================");
    Serial.printf("[SYSTEM] Free Heap: %d bytes, Free PSRAM: %d bytes, PSRAM Found: %s\n",
                  ESP.getFreeHeap(), ESP.getFreePsram(), psramFound() ? "YES" : "NO");

    wifiMulti.addAP("WPRVT", "123qweasdzxc,./@");
    wifiMulti.addAP("ALHN-F4DF", "2WC5rWCSwC");
    wifiMulti.addAP("Guest", "12345678");

    Serial.println("[WIFI] Connecting to Wi-Fi (WPRVT / ALHN-F4DF / Guest)...");
    while (wifiMulti.run() != WL_CONNECTED) {
        delay(300);
        Serial.print(".");
    }
    Serial.println();
    Serial.printf("[WIFI] Connected to %s! IP: %s\n", WiFi.SSID().c_str(), WiFi.localIP().toString().c_str());
    WiFi.setSleep(false);

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.frame_size = FRAMESIZE_QVGA; // 320x240 for high-speed streaming
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Camera init failed: 0x%x\n", err);
        return;
    }

    sensor_t * s = esp_camera_sensor_get();
    if (s != NULL) {
        if (s->id.PID == OV3660_PID) {
            s->set_vflip(s, 1);
            s->set_brightness(s, 1);
            s->set_saturation(s, -2);
        }
        s->set_framesize(s, FRAMESIZE_QVGA); // 320x240
    }

    startCameraServer();

    if (MDNS.begin("aznvision")) {
        MDNS.addService("http", "tcp", 80);
        MDNS.addService("mjpeg", "tcp", 81);
        Serial.println("[mDNS] Responder started: http://aznvision.local:81/stream");
    }

    Serial.println("==================================================");
    Serial.println(">>> WI-FI CAMERA SERVER READY! <<<");
    Serial.print("SSID: ");
    Serial.println(WiFi.SSID());
    Serial.print("Web UI / Capture URL: http://");
    Serial.println(WiFi.localIP());
    Serial.print("High-Speed Stream URL: http://");
    Serial.print(WiFi.localIP());
    Serial.println(":81/stream");
    Serial.println("==================================================");
}

void loop() {
    delay(10000);
}
