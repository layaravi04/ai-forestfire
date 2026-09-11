import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

class MultiModelFusionEngine:
    """
    Combines predictions from multiple modular model pipelines (YOLO, LSTM, Swin-CNN, ConvLSTM),
    temporal persistence metrics, and NASA satellite FRP data into a unified fire probability and risk score.
    """
    def __init__(self, config_path: Optional[str] = None):
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.weights = cfg.get("fusion", {}).get("weights", {})
        else:
            self.weights = {
                "yolo_confidence": 0.25,
                "lstm_temporal": 0.25,
                "swin_cnn": 0.15,
                "convlstm_attention": 0.15,
                "temporal_persistence": 0.10,
                "satellite_confirmation": 0.10
            }

    def compute_fusion_risk(self, model_outputs: Dict[str, Any],
                            persistence_score: float = 0.5,
                            satellite_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Input:
            model_outputs: Dictionary mapping model names to predicted confidence dicts
            persistence_score: Normalized persistent detection metric (0.0 to 1.0)
            satellite_info: Optional NASA FIRMS hotspot record
        Output:
            fire_probability, smoke_probability, risk_score (0-100), risk_level
        """
        yolo_conf = model_outputs.get("baseline_yolo", {}).get("fire_confidence", 0.0)
        lstm_conf = model_outputs.get("yolo_efficientnet_lstm", {}).get("fire_confidence", 0.0)
        swin_conf = model_outputs.get("swin_cnn", {}).get("fire_confidence", 0.0)
        convlstm_conf = model_outputs.get("convlstm_attention", {}).get("fire_confidence", 0.0)
        
        yolo_smoke = model_outputs.get("baseline_yolo", {}).get("smoke_confidence", 0.0)
        lstm_smoke = model_outputs.get("yolo_efficientnet_lstm", {}).get("smoke_confidence", 0.0)

        sat_score = 0.0
        if satellite_info:
            sat_conf = satellite_info.get("confidence", 50.0) / 100.0
            frp = min(1.0, satellite_info.get("FRP", 0.0) / 200.0)
            sat_score = (sat_conf + frp) / 2.0

        w = self.weights
        fused_fire_prob = (
            w.get("yolo_confidence", 0.25) * yolo_conf +
            w.get("lstm_temporal", 0.25) * lstm_conf +
            w.get("swin_cnn", 0.15) * swin_conf +
            w.get("convlstm_attention", 0.15) * convlstm_conf +
            w.get("temporal_persistence", 0.10) * persistence_score +
            w.get("satellite_confirmation", 0.10) * sat_score
        )
        
        fused_smoke_prob = (yolo_smoke + lstm_smoke) / 2.0

        risk_score = round(float(fused_fire_prob * 100.0), 2)
        
        if risk_score >= 75.0:
            risk_level = "CRITICAL"
        elif risk_score >= 50.0:
            risk_level = "HIGH"
        elif risk_score >= 25.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "fire_probability": round(float(fused_fire_prob), 4),
            "smoke_probability": round(float(fused_smoke_prob), 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "weights_used": self.weights,
            "scientific_note": "Risk weights are configurable hyperparameters tuned on validation data."
        }
