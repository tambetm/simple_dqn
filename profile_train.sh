#!/usr/bin/env bash

python -m cProfile -s time src/agent.py --exploration_steps=1 --learn_start=32 --train_steps=1032 --test_steps=0 --epochs=1 roms/pong.bin $*
