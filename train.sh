#!/usr/bin/env bash

mkdir -p results
full=${1##*/}
game=${full%.*}
python src/main.py --save_weights_path snapshots --csv_file results/$game.csv $*
