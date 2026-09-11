# ARCHITECTURE SPECIFICATION

## AI-Based Early Detection of Forest Fire and Smoke from Satellite / Camera Images Using Hybrid Spatial-Temporal Deep Learning

---

## 1. System Overview & Core Data Flow

```text
                  +-----------------------------------+
                  |  NASA MODIS / VIIRS / FIRMS API  |
                  +-----------------+-----------------+
                                    |
                                    v
+------------------+     +-------------------+
| Camera / Image / | --> | Satellite Data    |
| Video Stream     |     | Ingestion Engine  |
+--------+---------+     +---------+---------+
         |                         |
         v                         v
+------------------+     +-------------------+
| Frame Extraction |     | Spatial-Temporal  |
| & Preprocessing  |     | Data Alignment    |
+--------+---------+     +---------+---------+
         |                         |
         +------------+------------+
                      |
                      v
      +---------------+---------------+
      |  Spatial Feature Extraction   |
      |  (YOLO / CNN / Swin / ViT)   |
      +---------------+---------------+
                      |
                      v
      +---------------+---------------+
      |  Temporal Sequence Modeling   |
      |  (LSTM / BiLSTM / ConvLSTM)   |
      +---------------+---------------+
                      |
                      v
      +---------------+---------------+
      |   Multi-Model Fusion Engine   |
      | (Weights + Persistence + FRP) |
      +---------------+---------------+
                      |
                      v
      +---------------+---------------+
      | Severity & Geolocation Engine |
      +---------------+---------------+
                      |
                      v
      +---------------+---------------+
      | Early Warning Alert Engine    |
      +---------------+---------------+
                      |
                      v
      +---------------+---------------+
      | React + Vite Dark Dashboard   |
      +-------------------------------+
```

---

## 2. Spatial vs. Temporal Deep Learning Paradigm

- **Spatial Modeling**: Processes individual static frames $I_t \in \mathbb{R}^{H \times W \times C}$ to extract high-level feature maps, bounding boxes, visual signatures (flames, smoke plumes, haze), and patch tokens.
- **Temporal Sequence Modeling**: Processes contiguous tensor sequences $[X_{t-K}, X_{t-K+1}, \dots, X_t]$ across sliding time windows of length $K$. Evaluates persistence, temporal spread velocity, and transient artifact filtering to prevent single-frame false positives.

---

## 3. Modular Model Architectures

The framework encapsulates 8 distinct model pipelines:

```text
models/
├── baseline_cnn/               # 2D CNN Classifier
├── baseline_yolo/              # YOLO Object Detector
├── baseline_efficientnet/      # EfficientNet Backbone
├── yolo_efficientnet_lstm/     # YOLO ROI -> EfficientNet Features -> LSTM
├── cnn_bilstm_attention/       # CNN -> BiLSTM -> Bahdanau Attention
├── swin_cnn/                   # Swin Transformer (Global) + CNN (Local) Fusion
├── yolo_vit_lstm/              # YOLO ROI -> Vision Transformer -> LSTM
└── convlstm_attention/         # 2D ConvLSTM + 3D Spatial-Temporal Attention
```

Each module contains:
- `model.py`: PyTorch Neural Network definition.
- `config.yaml`: Model hyper-parameters, window sizes, learning rate, channels.
- `train.py`: Isolated training loop with loss reporting.
- `evaluate.py`: Evaluation pipeline returning standard research metrics.
- `inference.py`: Standardized inferencing interface returning predictions, probabilities, and attention maps.
- `README.md`: Technical documentation of mathematical mechanisms.

---

## 4. Multi-Model Fusion & Risk Calculation

The fusion engine calculates a unified `risk_score` $R \in [0, 100]$:

$$R = w_1 \cdot P_{\text{YOLO}} + w_2 \cdot P_{\text{LSTM}} + w_3 \cdot P_{\text{Swin-CNN}} + w_4 \cdot P_{\text{ConvLSTM}} + w_5 \cdot S_{\text{persistence}} + w_6 \cdot S_{\text{satellite}}$$

Where:
- $P_{\text{model}}$ is the normalized probability output from each model pipeline.
- $S_{\text{persistence}}$ represents temporal detection stability across consecutive frames.
- $S_{\text{satellite}}$ represents proximity and FRP (Fire Radiative Power) confirmation from NASA FIRMS.
- Risk categories are mapped as:
  - $R < 25$: **LOW**
  - $25 \le R < 50$: **MEDIUM**
  - $50 \le R < 75$: **HIGH**
  - $R \ge 75$: **CRITICAL**

---

## 5. Severity Estimation Engine

Combines four distinct physical and visual indicators:
1. **Bounding Box / Segment Area Ratio**: Normalized visual ROI scale relative to total image canvas.
2. **Smoke Density / Opacity**: Spatial feature map intensity extracted from smoke classification heads.
3. **Temporal Growth Rate**: $\Delta A / \Delta t$ bounding box area velocity over consecutive sequence windows.
4. **Satellite Radiative Intensity**: FRP value in Megawatts (MW) when satellite cross-verification is available.

---

## 6. Geolocation & Interactive GIS Mapping

- **Satellite Hotspots**: Real-time geolocation using WGS84 EPSG:4326 lat/lon coordinates from MODIS/VIIRS sensors.
- **Ground Camera Geofencing**: Configurable camera GPS positions $(lat, lon, alt, fov, azimuth)$ mapped onto interactive Leaflet / OpenStreetMap layers with calculated threat radius overlays.

---

## 7. Web Dashboard & API Layer

- **Backend**: FastAPI REST framework with CORS enabled, asynchronous processing handlers, and SQLite/SQLAlchemy ORM layer.
- **Frontend**: React + Vite SPA built with Tailwind CSS, Leaflet JS, Recharts, Lucide Icons, featuring a high-contrast dark theme optimized for AI disaster control centers.
