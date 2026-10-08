import argparse
import yaml
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def run_prediction(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    predict_cfg = cfg["predict"]
    model_cfg = cfg["train"]

    print(f"Loading model from {model_cfg['model_path']}...")
    model = joblib.load(model_cfg["model_path"])

    print(f"Loading inference data from {predict_cfg['input_path']}...")
    test_df = pd.read_csv(predict_cfg["input_path"])
    
    if "target" in test_df.columns:
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]
    else:
        X_test = test_df
        y_test = None

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    result_df = pd.DataFrame({"prediction": preds, "probability": probs})
    result_df.to_csv(predict_cfg["predictions_path"], index=False)
    print(f"Predictions saved to {predict_cfg['predictions_path']} (count: {len(result_df)})")

    if y_test is not None:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)
        print(f"Test Evaluation Metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_prediction(args.config)
