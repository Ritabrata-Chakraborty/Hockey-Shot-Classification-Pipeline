#!/bin/bash
set -e
DATA_PATH=${1:-data/hockey_shots*.csv}
# Expand glob and get latest file
FILES=($DATA_PATH)
LATEST_FILE="${FILES[${#FILES[@]}-1]}"
python3 -m src.physics.statistical_plots "$LATEST_FILE" analysis/statistics
python3 -m src.physics.animation_display "$LATEST_FILE" analysis

