import argparse
import yaml
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def prepare_data(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["data"]

    print("Generating synthetic customer churn classification data...")
    X, y = make_classification(
        n_samples=cfg["n_samples"],
        n_features=cfg["n_features"],
        n_informative=cfg["n_informative"],
        random_state=cfg["random_state"]
    )
    feature_names = [f"feature_{i}" for i in range(cfg["n_features"])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    df.to_csv(cfg["raw_path"], index=False)
    print(f"Saved raw dataset to {cfg['raw_path']} (shape: {df.shape})")

    train_df, test_df = train_test_split(
        df,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=df["target"]
    )
    train_df.to_csv(cfg["train_path"], index=False)
    test_df.to_csv(cfg["test_path"], index=False)
    print(f"Saved train set ({train_df.shape}) to {cfg['train_path']}")
    print(f"Saved test set ({test_df.shape}) to {cfg['test_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    prepare_data(args.config)
