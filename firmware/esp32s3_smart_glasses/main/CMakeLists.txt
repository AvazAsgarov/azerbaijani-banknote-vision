# Component CMakeLists.txt for Smart Glasses Firmware
idf_component_register(
    SRCS
        "main.cpp"
        "camera_pipeline.cpp"
        "model_runner.cpp"
        "safety_guard.cpp"
    INCLUDE_DIRS
        "."
    REQUIRES
        esp32-camera
        esp-tflite-micro
        nvs_flash
        bt
)
