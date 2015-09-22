#!/usr/bin/env bash

full=${1##*/}
game=${full%.*}
#python src/main.py --replay_size 4 --play_games 1 --record_screen_path images/$game --record_sound_filename images/$game.wav $*
#avconv -r 60 -i images/$game/%06d.png -i images/$game.wav -f mov -c:a mp3 -c:v libx264 videos/$game.mov
python src/main.py --replay_size 4 --play_games 1 --record_screen_path images/$game $*
avconv -r 60 -i images/$game/%06d.png -f mov -c:v libx264 videos/$game.mov
