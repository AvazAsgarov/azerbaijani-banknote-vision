# Model Evaluation Arm: RT-DETR-L

**Model Identifier:** `rtdetr_l`  
**Architecture:** RT-DETR-L (Real-Time Vision Transformer)  
**Status:** **Hypothesis Refuted / Degraded Baseline**  
**Parameter Count:** 32.82M (+63.6% over YOLO11m)  
**Checkpoint Path:** `weights/best.pt` (66.2 MB)  

---

## Performance Summary (Zero-Leakage Test Split)
* **Test mAP@0.50:** **69.65%** (-8.89% vs YOLO11m)
* **Test mAP@0.50:0.95:** **53.68%** (-7.99% vs YOLO11m)
* **Test mAP@0.75:** **51.86%** (-8.06% vs YOLO11m)
* **A100 GPU Latency:** **10.94 ms** (Throughput: 91.4 FPS - 2.3x slower)
* **Training Time:** 68.76 minutes (100 epochs on A100 GPU)

## Denomination-Level AP@0.50 Breakdown
* **001 AZN:** 82.93%
* **005 AZN:** 80.27%
* **010 AZN:** 78.34%
* **020 AZN:** 33.73%
* **050 AZN:** 67.79% (-26.04% vs YOLO11m)
* **100 AZN:** 95.22%
* **200 AZN:** 49.29%

## Architectural Post-Mortem: Why the Transformer Failed
* **Sample Inefficiency:** Transformers require vast training corpora to learn spatial locality. With 1,760 banknote images, the deformable cross-attention layers overfit to training background patterns.
* **Boundary Softening:** Sparse sampling points drifted into textured table backgrounds, degrading high-IoU box localization (mAP75 dropped to 51.86%).
* **Heavy Latency:** Hybrid encoder and cross-attention decoders added substantial computational complexity, running at only 91 FPS compared to YOLO11m's 210 FPS.
