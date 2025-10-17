#!/bin/bash
set -e

# Usage: ./control.sh [model_type] [shot_type]
# Defaults: xgboost, drag_flick (working demo scenario)

MODEL_TYPE=${1:-"xgboost"}
SHOT_TYPE=${2:-"drag_flick"}

# Map model type to model path
case "$MODEL_TYPE" in
    "tcn")
        MODEL_PATH="models/checkpoints/best_tcn_model.pth"
        ;;
    "xgboost"|"xgb")
        MODEL_PATH="models/checkpoints/xgboost_model.pkl"
        MODEL_TYPE="xgboost"
        ;;
    "svm")
        MODEL_PATH="models/checkpoints/svm_model.pkl"
        ;;
    "random_forest"|"rf")
        MODEL_PATH="models/checkpoints/random_forest_model.pkl"
        MODEL_TYPE="random_forest"
        ;;
    "knn")
        MODEL_PATH="models/checkpoints/knn_model.pkl"
        ;;
    *)
        echo "❌ Unknown model type: $MODEL_TYPE"
        echo "Available: tcn, xgboost (xgb), svm, random_forest (rf), knn"
        exit 1
        ;;
esac

echo "🎯 Running control system demo"
echo "   Model: $MODEL_TYPE"
echo "   Intended Shot: $SHOT_TYPE"
echo ""

python3 -m src.control_system.run_control "$MODEL_PATH" "$MODEL_TYPE" --shot-type "$SHOT_TYPE" --output models/control_results --max-iterations 15

