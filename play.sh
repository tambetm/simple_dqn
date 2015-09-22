#!/usr/bin/env bash

python src/main.py --random_steps 1 --replay_size 4 --play_games 1 --backend cpu --display_screen $*
