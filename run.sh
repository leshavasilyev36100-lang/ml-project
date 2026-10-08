#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/config.yaml}"
STAGE="${1:-all}"

# Проверка и активация venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found. Run 'uv venv && uv sync' first."
    exit 1
fi

echo "=========================================="
echo "Starting ML Pipeline"
echo "Stage: ${STAGE}"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

run_data() {
    echo "[1/3] Running Data Preparation..."
    python src/data.py --config "${CONFIG_PATH}"
}

run_train() {
    echo "[2/3] Running Model Training..."
    python src/train.py --config "${CONFIG_PATH}"
}

run_predict() {
    echo "[3/3] Running Inference..."
    python src/predict.py --config "${CONFIG_PATH}"
}

case "${STAGE}" in
    data)
        run_data
        ;;
    train)
        run_train
        ;;
    predict)
        run_predict
        ;;
    all)
        run_data
        run_train
        run_predict
        ;;
    *)
        echo "Unknown stage: ${STAGE}"
        echo "Usage: ./run.sh [data|train|predict|all]"
        exit 1
        ;;
esac

echo "=========================================="
echo "Pipeline finished successfully!"
echo "=========================================="
