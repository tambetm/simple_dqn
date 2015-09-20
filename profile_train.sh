#!/usr/bin/env bash

python -m cProfile -s tottime $* src/main.py --exploration_end=1 --random_steps=32 --train_steps=1000 --test_steps=0 --epochs=1 --save_weights_path '' roms/pong.bin
