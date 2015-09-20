#!/usr/bin/env bash

python -m cProfile -s cumtime $* src/main.py --exploration_steps=1 --random_steps=32 --train_steps=1032 --test_steps=0 --epochs=1 roms/pong.bin
