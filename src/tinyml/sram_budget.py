"""SRAM budget calculator for ESP32-S3 YOLO-FastestV2 INT8 deployment."""
from dataclasses import dataclass

ESP32S3_SRAM_TOTAL_KB = 512
ESP32S3_SRAM_RESERVED_KB = 128

@dataclass
class SRAMProfile:
    model_weights_kb: float
    runtime_buffers_kb: float
    stack_heap_kb: float
    total_kb: float
    available_kb: float = ESP32S3_SRAM_TOTAL_KB - ESP32S3_SRAM_RESERVED_KB
    fits: bool = False
    def __post_init__(self):
        self.fits = self.total_kb <= self.available_kb

YOLO_FASTESTV2_PROFILE = SRAMProfile(model_weights_kb=289.8, runtime_buffers_kb=64.0, stack_heap_kb=20.0, total_kb=289.8+64.0+20.0)

def check_budget(profile):
    return {"fits": profile.fits, "total_kb": profile.total_kb,
            "available_kb": profile.available_kb, "headroom_kb": profile.available_kb - profile.total_kb}
