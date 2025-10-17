#!/bin/bash
set -e
DATA_PATH=${1:-"data/hockey_shots*.csv"}
EPOCHS=${2:-150}
python3 -m src.classifiers.train_all "$DATA_PATH" models/checkpoints --epochs "$EPOCHS"

