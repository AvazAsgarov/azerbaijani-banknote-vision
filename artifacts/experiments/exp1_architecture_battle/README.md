# Experiment 1: Neural Architecture Battle

**Canonical Identifier:** `exp1_architecture_battle`  
**Status:** **COMPLETED (100 Epochs A100 GPU)**  
**Target Domain:** Azerbaijani Banknote Detection (7 Denominations: 1, 5, 10, 20, 50, 100, 200 AZN)  
**Execution Environment:** Dedicated NVIDIA A100-SXM4-40GB GPU Cluster  

---

## 1. Overview & Scientific Objective
Experiment 1 executed a formal tournament comparing four modeling paradigms on Azerbaijani banknote recognition under identical optimization conditions:
1. **DINOv2 ViT-L/14:** Frozen Foundation Vision Transformer (Linear Probe baseline, 304.3M params).
2. **YOLOv8m:** Modern anchor-free pure convolutional network (25.86M params).
3. **YOLO11m:** State-of-the-art CNN with Pointwise Spatial Attention (C2PSA, 20.06M params).
4. **RT-DETR-L:** End-to-end real-time Vision Transformer (32.82M params).

---

## 2. Quantitative Benchmark Results (Zero-Leakage Test Split)

| Architecture | Paradigm | Parameters (M) | Test $\text{mAP}_{50}$ | Test $\text{mAP}_{50-95}$ | Test $\text{mAP}_{75}$ | A100 Latency | Throughput | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8m** | **Anchor-Free CNN** | 25.86 | **95.96%** | **93.07%** | **94.48%** | 8.87 ms | 112.8 FPS | **18.87 min** |
| RT-DETR-L | Vision Transformer | 32.82 | 95.46% | 92.60% | 94.42% | 11.91 ms | 84.0 FPS | 140.47 min |
| **YOLO11m** | **Spatial Attention CNN** | **20.06** | 93.63% | 90.09% | 91.82% | **7.18 ms** | **139.2 FPS** | 61.33 min |
| DINOv2 ViT-L/14 | Frozen ViT Backbone | 304.30 | 42.56%* | N/A | N/A | 24.10 ms | 41.5 FPS | N/A |

*\*Linear probe accuracy on patch embeddings.*

---

## 3. Scientific Hypothesis Verdict: REJECTED
* **Hypothesis ($H_1$):** Vision Transformers will significantly outperform CNNs due to global attention.
* **Verdict:** **REJECTED ($H_0$ CONFIRMED)**. YOLOv8m achieved the highest accuracy (95.96% vs 95.46%), trained 7.4x faster (18.9 min vs 140.5 min), and YOLO11m delivered the fastest inference (7.18 ms, 139.2 FPS).

---

## 4. Directory Layout

```
artifacts/experiments/exp1_architecture_battle/
├── README.md                           <- Master experiment documentation
├── exp1_diagnostic_report.md           <- Scientific post-mortem & root-cause analysis
├── exp1_architecture_comparison.csv    <- Authentic comparative tabular metrics
├── exp1_architecture_comparison.json   <- Machine-readable serialized benchmark data
├── figures/                            <- 300 DPI publication-grade comparison plots
│   ├── README.md                       <- Catalog of diagnostic figures
│   ├── exp1_map_comparison.png         <- mAP50 & mAP50-95 bar chart across architectures
│   ├── exp1_per_class_ap.png           <- Denomination-level AP50 breakdown
│   └── pareto_latency_vs_map.png       <- Accuracy vs Latency Pareto frontier
├── yolo11m/                            <- YOLO11m weights, curves, logs, evaluations
├── yolov8m/                            <- YOLOv8m weights, curves, logs, evaluations
└── rtdetr_l/                           <- RT-DETR-L weights, curves, logs, evaluations
```
