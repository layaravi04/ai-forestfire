import pytest
from fusion.fusion_engine import MultiModelFusionEngine
from severity.severity_engine import SeverityEstimator

def test_fusion_engine():
    engine = MultiModelFusionEngine()
    dummy_outputs = {
        "baseline_yolo": {"fire_confidence": 0.85, "smoke_confidence": 0.70},
        "yolo_efficientnet_lstm": {"fire_confidence": 0.90},
        "swin_cnn": {"fire_confidence": 0.92},
        "convlstm_attention": {"fire_confidence": 0.94}
    }
    res = engine.compute_fusion_risk(dummy_outputs, persistence_score=0.8)
    assert "risk_score" in res
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def test_severity_estimator():
    estimator = SeverityEstimator()
    boxes = [{"class": "fire", "bbox": [10, 10, 200, 200]}]
    res = estimator.estimate_severity(boxes, smoke_confidence=0.8, temporal_growth_rate=0.5)
    assert res["severity_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
