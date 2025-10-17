#!/bin/bash
set -e
DATA_PATH=${1:-data/hockey_shots*.csv}
MODELS_DIR=${2:-models/checkpoints}
OUTPUT_DIR=${3:-models/single_test_results}

# Expand glob pattern
FILES=($DATA_PATH)
LATEST_FILE="${FILES[${#FILES[@]}-1]}"

# Test each model individually
for MODEL in tcn xgboost svm random_forest knn; do
    if [[ "$MODEL" == "tcn" ]]; then
        MODEL_FILE="$MODELS_DIR/best_tcn_model.pth"
    else
        MODEL_FILE="$MODELS_DIR/${MODEL}_model.pkl"
    fi
    
    if [ -f "$MODEL_FILE" ]; then
        python3 -m src.classifiers.test_models --mode single --model-path "$MODEL_FILE" --model-type "$MODEL" --data "$LATEST_FILE" --output "$OUTPUT_DIR" --num-samples 1
    fi
done

