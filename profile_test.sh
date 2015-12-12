#!/usr/bin/env bash

# never explore, always predict
python -m cProfile -s cumtime $* src/main.py --exploration_rate_test=0 --random_steps=0 --train_steps=0 --test_steps=10000 --epochs=1 roms/pong.bin
