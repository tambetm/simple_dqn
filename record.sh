#!/usr/bin/env bash

full=${1##*/}
game=${full%.*}
#python src/main.py --random_steps 1 --replay_size 4 --play_games 1 --record_screen_path $path/$game --record_sound_filename $path/$game.wav $*
#avconv -r 60 -i $path/$game/%06d.png -i $path/$game.wav -f mov -c:a mp3 -c:v libx264 $path/$game.mov
python src/main.py --random_steps 1 --replay_size 4 --play_games 1 --record_screen_path images/$game $*
avconv -r 60 -i images/$game/%06d.png -f mov -c:v libx264 videos/$game.mov
