import pytest
import numpy as np
from models.baseline_cnn.model import BaselineCNN
from models.yolo_efficientnet_lstm.model import YOLO_EfficientNet_LSTM
from models.cnn_bilstm_attention.model import CNN_BiLSTM_Attention
from models.swin_cnn.model import Swin_CNN_Fusion
from models.yolo_vit_lstm.model import YOLO_ViT_LSTM
from models.convlstm_attention.model import ConvLSTM_Attention

def test_baseline_cnn():
    model = BaselineCNN()
    x = np.random.randn(2, 3, 224, 224)
    out = model.forward(x)
    assert "fire_prob" in out and "smoke_prob" in out

def test_yolo_efficientnet_lstm():
    model = YOLO_EfficientNet_LSTM()
    seq_x = np.random.randn(2, 8, 3, 224, 224)
    out = model.forward(seq_x)
    assert "fire_prob" in out

def test_cnn_bilstm_attention():
    model = CNN_BiLSTM_Attention()
    seq_x = np.random.randn(2, 8, 3, 224, 224)
    out = model.forward(seq_x)
    assert "attention_weights" in out

def test_swin_cnn():
    model = Swin_CNN_Fusion()
    x = np.random.randn(2, 3, 224, 224)
    out = model.forward(x)
    assert out["fused_dim"] == 1024

def test_yolo_vit_lstm():
    model = YOLO_ViT_LSTM()
    seq_x = np.random.randn(2, 8, 3, 224, 224)
    out = model.forward(seq_x)
    assert "spatial_representation" in out

def test_convlstm_attention():
    model = ConvLSTM_Attention()
    seq_x = np.random.randn(2, 8, 3, 224, 224)
    out = model.forward(seq_x)
    assert "fire_prob" in out
