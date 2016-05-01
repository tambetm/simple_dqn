#!/usr/bin/env bash

snapshot=$1
full=${1##*/}
game=${full%%_*}
shift

python src/test_gym.py $game results/$game --load_weights $snapshot $*
