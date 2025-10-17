#!/bin/bash
set -e
SHOTS=${1:-1000}
python3 -m src.physics.procedural_shot_generation --shots "$SHOTS" --output-dir data

