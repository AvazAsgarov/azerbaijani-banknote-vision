# Oral Defense Slide Deck & Technical Presentation (`presentation/`)

This directory houses the official LaTeX Beamer presentation materials and compiled deliverables for the **AZN-Vision** project defense at the **National Artificial Intelligence Center (AI Academy)**.

---

## Deliverables

* **[`presentation.pdf`](presentation.pdf)**: Official 20-minute technical slide deck formatted in 16:9 widescreen presentation mode, incorporating empirical benchmark curves, architectural diagrams, and deployment demonstrations.

---

## Presentation Metadata

* **Project Title:** AZN-Vision: Assistive Banknote Intelligence & Edge Vision System
* **Course:** Deep Learning (DLE-AI-202) · Cohort I 2026
* **Group:** M001 | **Team:** Qarabağ Zəfəri
* **Authors:**
  - **Avaz Asgarov** ([@AvazAsgarov](https://github.com/AvazAsgarov)) — Lead Architect & Systems Engineer
  - **Gulnar Babazade** ([@gulnarbabazade](https://github.com/gulnarbabazade)) — Dataset & Benchmark Specialist
  - **Hasan Mammadov** ([@Hasawr](https://github.com/Hasawr)) — Photometrics & Compression Specialist
  - **Kazim Mammadli** ([@KazimMammadli](https://github.com/KazimMammadli)) — TinyML & Augmentation Specialist
  - **Nicat Alaskarli** ([@NicatAlaskarli](https://github.com/NicatAlaskarli)) — XAI & Hardware Specialist

---

## LaTeX Source Structure

* **`main.tex`**: Master Beamer entry point defining custom brand typography, color palettes (Dark Navy `#0D0F1A`, Accent Lime `#D4F938`, Tech Blue `#007BFF`), and modular frame inclusions.
* **`sections/01_dataset.tex`**: Zero-leakage environment-stratified dataset design, 7 banknote denominations, and Fleiss' Kappa ($0.962$) inter-annotator calibration.
* **`sections/02_experiments.tex`**: Empirical results across 6 deep learning studies:
  - *Exp 1:* Architecture Battle (YOLOv8m vs YOLO11m vs RT-DETR-L).
  - *Exp 2:* Data Augmentation Impact Ablation.
  - *Exp 3:* Color Space & Chromatic Shortcut Learning.
  - *Exp 4:* Multiscale Resolution Scaling Dynamics ($320 \rightarrow 640 \rightarrow 1280$).
  - *Exp 5:* Model Compression, INT8 PTQ & L1 Filter Pruning.
  - *Exp 6:* Explainable AI (XAI) EigenCAM Numismatic Attention Heatmaps.
* **`sections/03_tinyml_hardware.tex`**: Microcontroller deployment on Seeed Studio XIAO ESP32-S3 Sense (289.8 KB SRAM footprint, 4-tier safety guardrail).
* **`sections/04_mobile_app.tex`**: Assistive cross-platform mobile companion app (Expo SDK 57, React Native, Azerbaijani voice synthesizer).
* **`sections/05_conclusion.tex`**: Quantitative summary, societal accessibility impact, and future edge deployment roadmap.
* **`assets/`**: High-resolution vector and raster assets (`ai_academy_logo.png`).
