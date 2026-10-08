import argparse
import json
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_model(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]

    print(f"Loading training data from {data_cfg['train_path']}...")
    train_df = pd.read_csv(data_cfg["train_path"])
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    print(f"Training GradientBoostingClassifier (n_estimators={train_cfg['n_estimators']}, max_depth={train_cfg['max_depth']})...")
    model = GradientBoostingClassifier(
        n_estimators=train_cfg["n_estimators"],
        learning_rate=train_cfg["learning_rate"],
        max_depth=train_cfg["max_depth"],
        random_state=train_cfg["random_state"]
    )
    model.fit(X_train, y_train)

    # Оценка на обучающей выборке
    preds = model.predict(X_train)
    probs = model.predict_proba(X_train)[:, 1]

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, preds)),
        "train_f1": float(f1_score(y_train, preds)),
        "train_roc_auc": float(roc_auc_score(y_train, probs))
    }

    joblib.dump(model, train_cfg["model_path"])
    print(f"Model saved to {train_cfg['model_path']}")

    with open(train_cfg["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {train_cfg['metrics_path']}: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    train_model(args.config)
