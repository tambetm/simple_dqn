#!/usr/bin/env bash

full=${1##*/}
game=${full%.*}
python src/main.py --save_weights_path snapshots --csv_file results/$game.csv $*
