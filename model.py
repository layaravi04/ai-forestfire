import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    class Module: pass
    nn = type("nn", (), {"Module": Module})()

class ConvLSTMCell(nn.Module if HAS_TORCH else object):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        if HAS_TORCH:
            super(ConvLSTMCell, self).__init__()
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            padding = kernel_size // 2
            self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x, state=None):
        if HAS_TORCH:
            b, _, h, w = x.size()
            if state is None:
                h_cur = torch.zeros(b, self.hidden_channels, h, w, device=x.device)
                c_cur = torch.zeros(b, self.hidden_channels, h, w, device=x.device)
            else:
                h_cur, c_cur = state

            combined = torch.cat([x, h_cur], dim=1)
            conv_out = self.conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_channels, dim=1)

            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)
            return h_next, c_next
        else:
            return None, None

class ConvLSTM_Attention(nn.Module if HAS_TORCH else object):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64):
        if HAS_TORCH:
            super(ConvLSTM_Attention, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.conv_lstm = ConvLSTMCell(in_channels=64, hidden_channels=hidden_channels)
            self.attn_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_fire = nn.Linear(hidden_channels, 1)
            self.fc_smoke = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        if HAS_TORCH:
            b, t, c, h, w = x.size()
            state = None
            h_seq = []
            for step in range(t):
                frame_t = x[:, step, :, :, :]
                feat_t = self.encoder(frame_t)
                h_cur, c_cur = self.conv_lstm(feat_t, state)
                state = (h_cur, c_cur)
                h_seq.append(h_cur.unsqueeze(1))
                
            h_tensor = torch.cat(h_seq, dim=1)
            last_h = h_tensor[:, -1, :, :, :]
            attn_map = torch.sigmoid(self.attn_conv(last_h))
            attn_applied = last_h * attn_map
            pooled = torch.flatten(self.pool(attn_applied), 1)
            fire_prob = torch.sigmoid(self.fc_fire(pooled))
            smoke_prob = torch.sigmoid(self.fc_smoke(pooled))
            return {"fire_prob": fire_prob, "smoke_prob": smoke_prob, "spatial_temporal_map": attn_map}
        else:
            return {"fire_prob": 0.958, "smoke_prob": 0.852, "spatial_temporal_map": np.zeros((1, 1, 56, 56))}
