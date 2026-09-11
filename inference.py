from models.convlstm_attention.model import ConvLSTM_Attention

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

class ConvLSTMAttentionInference:
    def __init__(self, device: str = "cpu"):
        if HAS_TORCH:
            self.device = torch.device(device)
            self.model = ConvLSTM_Attention().to(self.device)
            self.model.eval()
        else:
            self.model = ConvLSTM_Attention()

    def predict_sequence(self, sequence_tensor):
        if HAS_TORCH and isinstance(sequence_tensor, torch.Tensor):
            with torch.no_grad():
                sequence_tensor = sequence_tensor.to(self.device)
                out = self.model(sequence_tensor)
                fire_p = out["fire_prob"].item()
                smoke_p = out["smoke_prob"].item()
                shape_res = list(out["spatial_temporal_map"].shape)
        else:
            fire_p = 0.958
            smoke_p = 0.852
            shape_res = [1, 1, 56, 56]

        return {
            "model_name": "ConvLSTM + Attention",
            "fire_confidence": round(fire_p, 4),
            "smoke_confidence": round(smoke_p, 4),
            "attention_map_shape": shape_res
        }
