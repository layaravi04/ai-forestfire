# Research Methodology & Theoretical Background

## Why Spatial-Temporal Hybridization?
Single-image classification is susceptible to transient artifacts such as sunlight reflections, dust, cloud cover, and fog. By enforcing temporal sequence modeling across sliding frame buffers ($t-7, \dots, t$), the system verifies persistence and growth velocity before triggering critical alerts.

## Fusion Risk Formula
Risk score $R \in [0, 100]$ is computed as:
$$R = 100 \times \sum w_i P_{\text{model}_i}$$
Weights are configurable hyperparameters tuned on validation split sets.
