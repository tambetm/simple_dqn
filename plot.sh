#!/usr/bin/env bash

file=${1%.*}
python src/plot.py --png_file $file.png $*
