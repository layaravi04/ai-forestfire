# System Architecture & Pipeline Design

## Overview
The system integrates ground camera sensors and NASA MODIS/VIIRS satellite active fire telemetry into a unified spatial-temporal deep learning network.

```text
NASA Satellite Telemetry (MODIS/VIIRS)
         +
Ground Camera Stream / Video / Images
         ↓
Preprocessing & CLAHE Contrast Enhancement
         ↓
Spatial Deep Learning (CNN / YOLO / Swin / ViT)
         ↓
Temporal Sequence Modeling (LSTM / BiLSTM / ConvLSTM)
         ↓
Multi-Model Risk Fusion Engine
         ↓
Severity Estimation & GIS Geolocation
         ↓
Early Warning Alert System
         ↓
React + Vite Disaster Monitoring Control Dashboard
```

## Spatial vs. Temporal Deep Learning
- **Spatial Deep Learning**: Analyzes single visual frames to isolate localized bounding boxes, smoke opacity, and patch tokens.
- **Temporal Sequence Modeling**: Processes contiguous sliding frame windows ($t-7, \dots, t$) to track flame growth rate, plume movement, and persistence over time.
