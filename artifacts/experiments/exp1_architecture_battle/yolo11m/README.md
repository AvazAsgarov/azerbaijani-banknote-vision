# Model Evaluation Arm: YOLO11m (Champion Architecture)

**Model Identifier:** `yolo11m`  
**Architecture:** YOLOv11m (Pointwise Spatial Attention CNN)  
**Status:** **OFFICIAL CHAMPION**  
**Parameter Count:** 20.06M  
**Checkpoint Path:** `weights/best.pt` (40.5 MB)  

---

## Performance Summary (Zero-Leakage Test Split)
* **Test mAP@0.50:** **78.54%** (#1 rank)
* **Test mAP@0.50:0.95:** **61.67%** (#1 rank)
* **Test mAP@0.75:** **59.92%** (#1 rank)
* **A100 GPU Latency:** **4.76 ms** (Throughput: 210.0 FPS)
* **Training Time:** 38.73 minutes (100 epochs on A100 GPU)

## Denomination-Level AP@0.50 Breakdown
* **001 AZN:** 88.49%
* **005 AZN:** 87.48%
* **010 AZN:** 79.82%
* **020 AZN:** 41.00%
* **050 AZN:** 93.83%
* **100 AZN:** 99.46%
* **200 AZN:** 59.67%

## Architectural Advantages
* Incorporates the **C2PSA (Cross-Stage Partial with Pointwise Spatial Attention)** module in the neck, enhancing feature representation for subtle numismatic watermarks and security threads.
* Preserves convolutional inductive biases in early stages, resulting in crisp boundary localization without transformer boundary softening.
* Highest accuracy, lowest parameter footprint, and optimal real-time throughput among all tested detectors.
