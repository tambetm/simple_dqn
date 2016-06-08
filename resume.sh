#!/usr/bin/env bash

snapshot=$1
full=${1##*/}
noext=${full%.*}
epoch=${noext##*_}
game_label=${noext%_*}
snapshots=snapshots/$game_label
results=results/${game_label}_${epoch}.csv
game=${full%%_*}
rom="roms/$game.bin"
shift

python src/main.py --random_steps 0 --save_weights_prefix $snapshots --csv_file $results --load_weights $snapshot --start_epoch $epoch $rom $*
