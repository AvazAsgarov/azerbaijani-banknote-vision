"""TFLite FlatBuffer schema validator for YOLO-FastestV2 INT8 model."""
from pathlib import Path
from dataclasses import dataclass

TFLITE_MAGIC = b"\x18\x00\x00\x00"
INT8_TYPE_CODE = 2

@dataclass
class FlatBufferReport:
    path: str; size_bytes: int; magic_valid: bool; has_int8_tensors: bool; ok: bool

def validate_tflite_flatbuffer(path: Path) -> FlatBufferReport:
    data = path.read_bytes(); size = len(data)
    magic_valid = size >= 8 and data[4:8] == TFLITE_MAGIC
    has_int8 = INT8_TYPE_CODE.to_bytes(4, "little") in data
    return FlatBufferReport(path=str(path), size_bytes=size, magic_valid=magic_valid,
                            has_int8_tensors=has_int8, ok=(magic_valid and size > 1024))
