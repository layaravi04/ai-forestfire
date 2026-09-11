# Deep Learning Model Specifications

## Baselines
1. **Baseline CNN**: 3-stage convolutional backbone for spatial classification.
2. **Baseline YOLO**: Anchor-based object detector returning bounding boxes, class labels, and confidence.
3. **Baseline EfficientNet**: Transfer feature extractor with 1280-dimensional embedding outputs.

## Hybrid Architectures
1. **YOLO + EfficientNet + LSTM**: Bounding box crop -> EfficientNet features -> LSTM temporal sequence.
2. **CNN + BiLSTM + Attention**: Spatial feature sequence -> Bidirectional LSTM -> Bahdanau Attention weights.
3. **Swin Transformer + CNN**: Swin Transformer global context (768-dim) + CNN local features (256-dim) fused via $L_2$ normalization.
4. **YOLO + ViT + LSTM**: YOLO ROI -> Vision Transformer spatial patch tokens -> LSTM temporal decoder.
5. **ConvLSTM + Attention**: 2D ConvLSTM preserving spatial grids across time -> 3D Spatial-Temporal Attention maps.
