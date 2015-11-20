#!/usr/bin/env bash

mkdir -p results
mkdir -p snapshots

file=$1
full=${file##*/}
game=${full%.*}

if [[ -z $2 ]] || [[ $2 == -* ]]; then
  snapshots=snapshots/$game
  results=results/$game.csv
  shift
else
  # additional parameter is experiment label
  label=$2
  snapshots=snapshots/${game}_${label}
  results=results/${game}_${label}.csv
  shift
  shift
fi

python src/main.py --save_weights_prefix $snapshots --csv_file $results $file $*
