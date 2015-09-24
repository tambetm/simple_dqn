#!/usr/bin/env bash

python src/main.py --replay_size 32 --random_steps 0 --train_steps 0 --epochs 1 $*
