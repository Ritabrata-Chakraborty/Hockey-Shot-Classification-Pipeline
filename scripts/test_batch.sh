#!/bin/bash
set -e
DATA_PATH=${1:-"data/hockey_shots*.csv"}
python3 -m src.classifiers.test_models --mode batch --data "$DATA_PATH" --output models/batch_test_results --models-dir models/checkpoints

