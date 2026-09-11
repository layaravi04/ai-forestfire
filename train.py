import os
import torch
from models.convlstm_attention.model import ConvLSTM_Attention

def train_convlstm_attention(device="cpu"):
    model = ConvLSTM_Attention().to(device)
    save_dir = "models/convlstm_attention/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
    print(f"[ConvLSTM+Attention] Checkpoint saved to {save_dir}/best_model.pth")
    return model

if __name__ == "__main__":
    train_convlstm_attention()
