# ConvLSTM + Attention Hybrid Model

## Architecture
1. **2D ConvLSTM**: Recurrent Convolutional cell that preserves 2D spatial image grids $(H, W)$ while encoding temporal dynamics across sequence steps.
2. **3D Spatial-Temporal Attention**: Computes localized attention masks over 2D spatial feature channels across time steps.
