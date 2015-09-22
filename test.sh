#!/usr/bin/env bash

python src/main.py --random_steps 1 --replay_size 4 --train_steps 0 --epochs 1 $*
