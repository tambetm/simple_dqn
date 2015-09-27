#!/usr/bin/env bash

# predict all moves by random, to separate prediction and training time
python -m cProfile -s cumtime $* src/main.py --replay_size 100 --exploration_rate_end=1 --random_steps=5 --train_steps=1000 --test_steps=0 --epochs=1 --train_frequency 1 --target_steps 0 roms/pong.bin
