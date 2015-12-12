#!/usr/bin/env bash

# never explore, always predict
python -m cProfile -s cumtime $* src/main.py --random_steps=10000 --train_steps=0 --test_steps=0 --epochs=1 roms/pong.bin
