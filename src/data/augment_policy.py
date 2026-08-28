"""Augmentation policy registry for AZN-Vision Experiment 2 ablation."""

POLICIES = {
    "raw":         {"fliplr": 0.0, "flipud": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0, "mosaic": 0.0},
    "geometric":   {"fliplr": 0.5, "flipud": 0.1, "degrees": 10.0, "translate": 0.1, "scale": 0.3, "mosaic": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0},
    "photometric": {"fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "mosaic": 0.0},
    "composite":   {"fliplr": 0.5, "flipud": 0.1, "degrees": 10.0, "translate": 0.1, "scale": 0.3, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "mosaic": 1.0},
}

def get_policy(name: str) -> dict:
    if name not in POLICIES:
        raise ValueError(f"Unknown policy '{name}'. Choose from: {list(POLICIES.keys())}")
    return dict(POLICIES[name])
