import numpy as np
from typing import Dict, Any, List, Optional
from preprocessing.image_processor import ImageProcessor
from preprocessing.sequence_generator import SequenceBuffer
from models.baseline_cnn.inference import BaselineCNNInference
from models.baseline_yolo.inference import BaselineYOLOInference
from models.baseline_efficientnet.inference import BaselineEfficientNetInference
from models.yolo_efficientnet_lstm.inference import YOLOEfficientNetLSTMInference
from models.cnn_bilstm_attention.inference import CNNBiLSTMAttentionInference
from models.swin_cnn.inference import SwinCNNInference
from models.yolo_vit_lstm.inference import YOLOViTLSTMInference
from models.convlstm_attention.inference import ConvLSTMAttentionInference
from fusion.fusion_engine import MultiModelFusionEngine
from severity.severity_engine import SeverityEstimator
from geolocation.geo_engine import GeolocationEngine

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

class UnifiedPredictor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = ImageProcessor()
        
        self.cnn_base = BaselineCNNInference(device=device)
        self.yolo_base = BaselineYOLOInference(device=device)
        self.effnet_base = BaselineEfficientNetInference(device=device)
        
        self.yolo_effnet_lstm = YOLOEfficientNetLSTMInference(device=device)
        self.cnn_bilstm_attn = CNNBiLSTMAttentionInference(device=device)
        self.swin_cnn = SwinCNNInference(device=device)
        self.yolo_vit_lstm = YOLOViTLSTMInference(device=device)
        self.convlstm_attn = ConvLSTMAttentionInference(device=device)
        
        self.fusion = MultiModelFusionEngine()
        self.severity = SeverityEstimator()
        self.geo = GeolocationEngine()

    def predict_single_image(self, image_input: Any, camera_lat: float = 37.7749, camera_lon: float = -122.4194) -> Dict[str, Any]:
        tensor, rgb_img = self.processor.preprocess_image(image_input)
        
        cnn_res = self.cnn_base.predict(tensor)
        yolo_res = self.yolo_base.predict_image(rgb_img)
        effnet_res = self.effnet_base.predict(tensor)
        swin_res = self.swin_cnn.predict(tensor)
        
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            seq_tensor = tensor.unsqueeze(1).repeat(1, 8, 1, 1, 1)
        else:
            seq_tensor = np.repeat(np.expand_dims(tensor, 1), 8, axis=1)

        yolo_lstm_res = self.yolo_effnet_lstm.predict_sequence(seq_tensor)
        bilstm_attn_res = self.cnn_bilstm_attn.predict_sequence(seq_tensor)
        vit_lstm_res = self.yolo_vit_lstm.predict_sequence(seq_tensor)
        convlstm_res = self.convlstm_attn.predict_sequence(seq_tensor)

        model_outputs = {
            "baseline_cnn": cnn_res,
            "baseline_yolo": yolo_res,
            "baseline_efficientnet": effnet_res,
            "yolo_efficientnet_lstm": yolo_lstm_res,
            "cnn_bilstm_attention": bilstm_attn_res,
            "swin_cnn": swin_res,
            "yolo_vit_lstm": vit_lstm_res,
            "convlstm_attention": convlstm_res
        }

        fusion_res = self.fusion.compute_fusion_risk(model_outputs, persistence_score=0.5)
        boxes = yolo_res.get("boxes", [])
        sev_res = self.severity.estimate_severity(
            boxes=boxes,
            smoke_confidence=fusion_res["smoke_probability"]
        )
        geo_res = self.geo.map_camera_detection(camera_lat, camera_lon)

        return {
            "fire_confidence": fusion_res["fire_probability"],
            "smoke_confidence": fusion_res["smoke_probability"],
            "risk_score": fusion_res["risk_score"],
            "risk_level": fusion_res["risk_level"],
            "severity": sev_res["severity_level"],
            "boxes": boxes,
            "geolocation": geo_res,
            "model_predictions": model_outputs,
            "severity_details": sev_res
        }

    def predict_sequence_buffer(self, seq_buffer: SequenceBuffer,
                                persistence_frames: int = 1,
                                camera_lat: float = 37.7749,
                                camera_lon: float = -122.4194) -> Dict[str, Any]:
        seq_tensor = seq_buffer.get_sequence_tensor()
        latest_rgb = seq_buffer.raw_frames[-1]

        yolo_res = self.yolo_base.predict_image(latest_rgb)
        
        if HAS_TORCH and isinstance(seq_tensor, torch.Tensor):
            last_frame = seq_tensor[:, -1, :, :, :]
        else:
            last_frame = seq_tensor[:, -1, :, :, :]

        cnn_res = self.cnn_base.predict(last_frame)
        effnet_res = self.effnet_base.predict(last_frame)
        swin_res = self.swin_cnn.predict(last_frame)

        yolo_lstm_res = self.yolo_effnet_lstm.predict_sequence(seq_tensor)
        bilstm_attn_res = self.cnn_bilstm_attn.predict_sequence(seq_tensor)
        vit_lstm_res = self.yolo_vit_lstm.predict_sequence(seq_tensor)
        convlstm_res = self.convlstm_attn.predict_sequence(seq_tensor)

        model_outputs = {
            "baseline_cnn": cnn_res,
            "baseline_yolo": yolo_res,
            "baseline_efficientnet": effnet_res,
            "yolo_efficientnet_lstm": yolo_lstm_res,
            "cnn_bilstm_attention": bilstm_attn_res,
            "swin_cnn": swin_res,
            "yolo_vit_lstm": vit_lstm_res,
            "convlstm_attention": convlstm_res
        }

        persistence_score = min(1.0, persistence_frames / 8.0)
        fusion_res = self.fusion.compute_fusion_risk(model_outputs, persistence_score=persistence_score)
        boxes = yolo_res.get("boxes", [])
        sev_res = self.severity.estimate_severity(
            boxes=boxes,
            smoke_confidence=fusion_res["smoke_probability"],
            temporal_growth_rate=(persistence_frames * 0.1)
        )
        geo_res = self.geo.map_camera_detection(camera_lat, camera_lon)

        return {
            "fire_confidence": fusion_res["fire_probability"],
            "smoke_confidence": fusion_res["smoke_probability"],
            "risk_score": fusion_res["risk_score"],
            "risk_level": fusion_res["risk_level"],
            "severity": sev_res["severity_level"],
            "boxes": boxes,
            "persistence_frames": persistence_frames,
            "geolocation": geo_res,
            "model_predictions": model_outputs,
            "severity_details": sev_res
        }
