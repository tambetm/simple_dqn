#!/usr/bin/env bash

snapshot=$1
full=${1##*/}
game=${full%%_*}
rom="roms/$game.bin"
shift

python src/main.py --random_steps 0 --train_steps 0 --epochs 1 --load_weights $snapshot $rom $*
