#!/usr/bin/env bash

python -m cProfile -s cumtime $* src/main.py --exploration_test=0 --random_steps=32 --train_steps=0 --test_steps=1000 --epochs=1 --save_weights_path '' roms/pong.bin
