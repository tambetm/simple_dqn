#!/usr/bin/env bash

full=${1##*/}
game=${full%.*}
rm -r videos/$game
#python src/main.py --replay_size 4 --play_games 1 --record_screen_path videos/$game --record_sound_filename videos/$game.wav $*
#avconv -r 60 -i videos/$game/%06d.png -i videos/$game.wav -f mov -c:a mp3 -c:v libx264 -y videos/$game.mov
python src/main.py --replay_size 4 --play_games 1 --record_screen_path videos/$game $*
avconv -r 60 -i videos/$game/%06d.png -f mov -c:v libx264 -y videos/$game.mov
