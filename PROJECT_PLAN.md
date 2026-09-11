# PROJECT PLAN: AI-Based Early Detection of Forest Fire and Smoke

## Executive Summary
This project implements a research-grade, hybrid spatial-temporal deep learning system for early detection of forest fires and smoke using satellite (NASA MODIS/VIIRS/FIRMS) and ground-based camera inputs.

---

## Non-Negotiable Rules & Guidelines
1. Real working full-stack system (FastAPI backend + PyTorch models + React/Vite/Tailwind UI + SQLite).
2. Absolute transparency: No fabricated datasets, fake accuracy metrics, fake satellite records, or unearned model performance numbers.
3. Explicit separation of Spatial (single-frame visual feature extraction) and Temporal (multi-frame sequence evolution) modelling.
4. Support for CPU inference by default, with CUDA auto-detection when available.
5. Fully modular architecture with 8 separate model pipelines (3 baselines + 5 hybrids).

---

## Phase Checklist & Roadmap

### Phase 1: Infrastructure & Core Setup
- [ ] Create repository directory tree (`forest-fire-ai/`).
- [ ] Define Python dependencies (`requirements.txt`) & configuration files (`config.yaml`, `.env.example`).
- [ ] Implement SQLite database tables (`FireDetection`, `SatelliteEvent`, `Alert`, `Camera`).
- [ ] Setup FastAPI server shell with health checks.
- [ ] Initialize React + Vite + Tailwind CSS frontend with dark disaster dashboard theme.

### Phase 2: Data Preprocessing & Sequence Engineering
- [ ] Build `preprocessing/image_processor.py` for image normalization, resizing, augmentations, contrast enhancement.
- [ ] Build `preprocessing/sequence_generator.py` for temporal sliding window sequences ($t-7, \dots, t$).
- [ ] Enforce data split safety (split by video/location/time period to avoid frame data leakage).

### Phase 3: NASA Satellite Data Ingestion
- [ ] Implement `data/satellite_ingestion.py` supporting NASA FIRMS API / CSV ingestion.
- [ ] Implement fallback `MODE=MOCK` engine for offline local execution with clearly tagged `MOCK DATA` flags.

### Phase 4: Baseline Models Development
- [ ] `baseline_cnn/` (Spatial CNN Classifier).
- [ ] `baseline_yolo/` (Object Detector for Bounding Boxes, Class, Confidence).
- [ ] `baseline_efficientnet/` (Transfer Learning Spatial Backbone).

### Phase 5: Hybrid Spatial-Temporal AI Models
- [ ] Model 1: `yolo_efficientnet_lstm/` (YOLO ROI -> EfficientNet Features -> LSTM Sequence Decoder).
- [ ] Model 2: `cnn_bilstm_attention/` (CNN Sequence -> BiLSTM -> Attention Mechanism with Weight Maps).
- [ ] Model 3: `swin_cnn/` (Swin Transformer Global + CNN Local -> Feature Fusion Head).
- [ ] Model 4: `yolo_vit_lstm/` (YOLO ROI -> ViT Spatial Tokens -> LSTM Temporal Representation).
- [ ] Model 5: `convlstm_attention/` (2D ConvLSTM Spatial-Temporal Tensor -> 3D Attention Mask).

### Phase 6: Multi-Model Fusion & Risk Engine
- [ ] Build `fusion/fusion_engine.py` weighted risk score calculator.
- [ ] Build `severity/severity_engine.py` estimating fire area, smoke density, persistence, FRP.
- [ ] Build `geolocation/geo_engine.py` mapping camera/satellite bounding box coordinates to GIS map layers.

### Phase 7: Alert Engine
- [ ] Build `alerts/alert_engine.py` evaluating persistence rules, satellite verification, and generating alert payloads.
- [ ] Implement browser alert, dashboard alert, and optional webhook dispatchers (Email/Telegram).

### Phase 8: Web Dashboard Implementation
- [ ] Dashboard Page: Key metric counters, active alerts, interactive Leaflet map, risk heatmap.
- [ ] Detection Page: Image/Video/Webcam analyzer with visual bounding boxes & persistence timeline.
- [ ] Satellite Page: NASA FIRMS hotspot table & GIS layer filter.
- [ ] Temporal Page: Multi-frame confidence progression charts & growth velocity meters.
- [ ] Model Comparison Page: Comparative metrics matrix across all 8 models.
- [ ] Alerts Page: Historical event log with status management.

### Phase 9: Benchmark, Testing & Explainability
- [ ] Implement `evaluation/benchmark.py` calculating mAP, IoU, F1, latency, parameter count.
- [ ] Implement Grad-CAM & Attention weight visualizer modules.
- [ ] Write unit & integration test suite (`tests/`).

### Phase 10: Complete Documentation Suite
- [ ] `README.md`
- [ ] `PROJECT_PLAN.md` & `ARCHITECTURE.md`
- [ ] `docs/architecture.md`, `docs/datasets.md`, `docs/models.md`, `docs/methodology.md`, `docs/evaluation.md`, `docs/limitations.md`, `docs/research_gap.md`.
