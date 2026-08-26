# Model Evaluation Arm: YOLOv8m

**Model Identifier:** `yolov8m`  
**Architecture:** YOLOv8m (Anchor-Free Convolutional Neural Network)  
**Status:** **Runner-Up**  
**Parameter Count:** 25.86M (+28.9% over YOLO11m)  
**Checkpoint Path:** `weights/best.pt` (52.0 MB)  

---

## Performance Summary (Zero-Leakage Test Split)
* **Test mAP@0.50:** **75.58%** (-2.96% vs YOLO11m)
* **Test mAP@0.50:0.95:** **57.83%** (-3.84% vs YOLO11m)
* **Test mAP@0.75:** **56.80%** (-3.12% vs YOLO11m)
* **A100 GPU Latency:** **4.39 ms** (Throughput: 227.7 FPS)
* **Training Time:** 23.87 minutes (100 epochs on A100 GPU)

## Denomination-Level AP@0.50 Breakdown
* **001 AZN:** 81.24%
* **005 AZN:** 85.81%
* **010 AZN:** 78.76%
* **020 AZN:** 44.80%
* **050 AZN:** 85.43%
* **100 AZN:** 95.59%
* **200 AZN:** 57.45%

## Architectural Analysis
* Standard anchor-free decoupled head with C2f backbone blocks.
* Fast and robust, but lacks spatial attention on small guilloche features, leading to lower overall precision than YOLO11m despite having 5.8M more parameters.
