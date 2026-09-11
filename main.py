import os
import io
import datetime
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from PIL import Image
import numpy as np

from backend.database.session import init_db, get_db
from backend.database.models import FireDetection, SatelliteEvent, Alert, Camera
from inference.unified_predictor import UnifiedPredictor
from data.satellite_ingestion import NASASatelliteIngestion
from alerts.alert_engine import EarlyWarningAlertEngine

# Initialize SQLite Database Tables
init_db()

app = FastAPI(
    title="AI Forest Fire & Smoke Early Detection System",
    description="Hybrid Spatial-Temporal Deep Learning Multi-Model Platform",
    version="1.0.0"
)

# Enable CORS for Frontend Development (Vite port 5173 / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor & Engines
predictor = UnifiedPredictor()
satellite_engine = NASASatelliteIngestion()
alert_engine = EarlyWarningAlertEngine()

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "system": "Forest Fire AI Early Detection System",
        "mode": os.getenv("MODE", "MOCK"),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    latitude: float = Query(37.7749),
    longitude: float = Query(-122.4194),
    db: Session = Depends(get_db)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)

        # Unified Multi-Model Inference
        res = predictor.predict_single_image(img_np, camera_lat=latitude, camera_lon=longitude)

        # Save Detection to SQLite
        detection_record = FireDetection(
            timestamp=datetime.datetime.utcnow(),
            latitude=latitude,
            longitude=longitude,
            source="camera_image",
            fire_confidence=res["fire_confidence"],
            smoke_confidence=res["smoke_confidence"],
            severity=res["severity"],
            risk_score=res["risk_score"],
            risk_level=res["risk_level"],
            model="Hybrid Spatial-Temporal Fusion",
            media_reference=file.filename
        )
        db.add(detection_record)
        db.commit()
        db.refresh(detection_record)

        # Trigger Alert Check
        alert_payload = alert_engine.evaluate_alert_condition(
            fusion_result={
                "fire_probability": res["fire_confidence"],
                "smoke_probability": res["smoke_confidence"],
                "risk_level": res["risk_level"]
            },
            location_name="Station Lookout Alpha",
            lat=latitude, lon=longitude
        )

        if alert_payload:
            new_alert = Alert(
                timestamp=datetime.datetime.utcnow(),
                location=alert_payload["location"],
                latitude=latitude,
                longitude=longitude,
                severity=alert_payload["severity"],
                risk=alert_payload["risk"],
                message=alert_payload["message"],
                status="ACTIVE"
            )
            db.add(new_alert)
            db.commit()

        res["detection_id"] = detection_record.id
        res["alert_triggered"] = alert_payload is not None
        res["alert"] = alert_payload
        return res

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image detection failed: {str(e)}")

@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    latitude: float = Query(37.7749),
    longitude: float = Query(-122.4194),
    db: Session = Depends(get_db)
):
    # Simulated video sequence processing across 8 frames
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_img[:, :, 0] = 200 # Add fire red hue
    dummy_img[:, :, 1] = 100
    
    res = predictor.predict_single_image(dummy_img, camera_lat=latitude, camera_lon=longitude)
    res["persistence_frames"] = 8
    res["is_persistent"] = True

    detection_record = FireDetection(
        timestamp=datetime.datetime.utcnow(),
        latitude=latitude,
        longitude=longitude,
        source="video_stream",
        fire_confidence=res["fire_confidence"],
        smoke_confidence=res["smoke_confidence"],
        severity=res["severity"],
        risk_score=res["risk_score"],
        risk_level=res["risk_level"],
        model="YOLO+EfficientNet+LSTM / ConvLSTM",
        media_reference=file.filename
    )
    db.add(detection_record)
    db.commit()
    return res

@app.post("/detect/sequence")
async def detect_sequence(
    files: list[UploadFile] = File(...),
    latitude: float = Query(37.7749),
    longitude: float = Query(-122.4194)
):
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    res = predictor.predict_single_image(dummy_img, camera_lat=latitude, camera_lon=longitude)
    res["sequence_length"] = len(files)
    return res

@app.get("/fires")
def get_fires(limit: int = 50, db: Session = Depends(get_db)):
    records = db.query(FireDetection).order_by(FireDetection.timestamp.desc()).limit(limit).all()
    return records

@app.get("/fires/{fire_id}")
def get_fire_by_id(fire_id: int, db: Session = Depends(get_db)):
    record = db.query(FireDetection).filter(FireDetection.id == fire_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Fire record not found")
    return record

@app.get("/alerts")
def get_alerts(limit: int = 50, db: Session = Depends(get_db)):
    records = db.query(Alert).order_by(Alert.timestamp.desc()).limit(limit).all()
    return records

@app.get("/satellite/hotspots")
def get_satellite_hotspots(days: int = 1):
    hotspots = satellite_engine.get_active_hotspots(days=days)
    return {
        "mode": os.getenv("MODE", "MOCK"),
        "count": len(hotspots),
        "hotspots": hotspots
    }

@app.get("/risk-map")
def get_risk_map(db: Session = Depends(get_db)):
    fires = db.query(FireDetection).order_by(FireDetection.timestamp.desc()).limit(30).all()
    hotspots = satellite_engine.get_active_hotspots()
    cameras = db.query(Camera).all()
    
    return {
        "fires": fires,
        "satellite_hotspots": hotspots,
        "cameras": cameras
    }

@app.get("/models")
def get_registered_models():
    return [
        {"id": "baseline_cnn", "name": "Baseline CNN", "type": "Baseline Spatial Classifier", "params": "1.2M"},
        {"id": "baseline_yolo", "name": "Baseline YOLO", "type": "Baseline Object Detector", "params": "3.1M"},
        {"id": "baseline_efficientnet", "name": "Baseline EfficientNet", "type": "Baseline Transfer Feature Extractor", "params": "5.3M"},
        {"id": "yolo_efficientnet_lstm", "name": "YOLO + EfficientNet + LSTM", "type": "Hybrid 1 (Spatial-Temporal ROI)", "params": "8.7M"},
        {"id": "cnn_bilstm_attention", "name": "CNN + BiLSTM + Attention", "type": "Hybrid 2 (Bidirectional Temporal Attention)", "params": "6.4M"},
        {"id": "swin_cnn", "name": "Swin Transformer + CNN", "type": "Hybrid 3 (Global + Local Feature Fusion)", "params": "28.5M"},
        {"id": "yolo_vit_lstm", "name": "YOLO + ViT + LSTM", "type": "Hybrid 4 (Vision Transformer Spatial + LSTM Temporal)", "params": "89.2M"},
        {"id": "convlstm_attention", "name": "ConvLSTM + Attention", "type": "Hybrid 5 (2D Spatial-Grid Temporal + 3D Attention)", "params": "11.8M"}
    ]

@app.get("/model-comparison")
def get_model_comparison():
    """
    Returns benchmark table comparing baselines vs five hybrid architectures.
    Only reports authentic evaluation metrics derived from testing runs.
    """
    from models.baseline_cnn.evaluate import evaluate_baseline_cnn
    from models.baseline_yolo.evaluate import evaluate_baseline_yolo
    from models.baseline_efficientnet.evaluate import evaluate_baseline_efficientnet
    from models.yolo_efficientnet_lstm.evaluate import evaluate_yolo_efficientnet_lstm
    from models.cnn_bilstm_attention.evaluate import evaluate_cnn_bilstm_attention
    from models.swin_cnn.evaluate import evaluate_swin_cnn
    from models.yolo_vit_lstm.evaluate import evaluate_yolo_vit_lstm
    from models.convlstm_attention.evaluate import evaluate_convlstm_attention

    return [
        evaluate_baseline_cnn(),
        evaluate_baseline_yolo(),
        evaluate_baseline_efficientnet(),
        evaluate_yolo_efficientnet_lstm(),
        evaluate_cnn_bilstm_attention(),
        evaluate_swin_cnn(),
        evaluate_yolo_vit_lstm(),
        evaluate_convlstm_attention()
    ]

@app.post("/camera/register")
def register_camera(name: str, latitude: float, longitude: float, db: Session = Depends(get_db)):
    cam = Camera(name=name, latitude=latitude, longitude=longitude, status="ONLINE")
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam
