import pandas as pd
from models.baseline_cnn.evaluate import evaluate_baseline_cnn
from models.baseline_yolo.evaluate import evaluate_baseline_yolo
from models.baseline_efficientnet.evaluate import evaluate_baseline_efficientnet
from models.yolo_efficientnet_lstm.evaluate import evaluate_yolo_efficientnet_lstm
from models.cnn_bilstm_attention.evaluate import evaluate_cnn_bilstm_attention
from models.swin_cnn.evaluate import evaluate_swin_cnn
from models.yolo_vit_lstm.evaluate import evaluate_yolo_vit_lstm
from models.convlstm_attention.evaluate import evaluate_convlstm_attention

def run_all_benchmarks():
    print("=" * 70)
    print("RUNNING ALL FOREST FIRE AI BENCHMARK EVALUATIONS")
    print("=" * 70)

    results = [
        evaluate_baseline_cnn(),
        evaluate_baseline_yolo(),
        evaluate_baseline_efficientnet(),
        evaluate_yolo_efficientnet_lstm(),
        evaluate_cnn_bilstm_attention(),
        evaluate_swin_cnn(),
        evaluate_yolo_vit_lstm(),
        evaluate_convlstm_attention()
    ]

    df = pd.DataFrame(results)
    print("\nBENCHMARK RESULTS MATRIX:")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    run_all_benchmarks()
