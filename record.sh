#!/usr/bin/env bash

snapshot=$1
full=${1##*/}
game=${full%%_*}
file=${full%.*}
rom="roms/$game.bin"
shift

rm -r videos/$game
#python src/main.py --play_games 1 --record_screen_path videos/$game --record_sound_filename videos/$game.wav --load_weights $snapshot $rom $*
#avconv -r 60 -i videos/$game/%06d.png -i videos/$game.wav -f mov -c:a mp3 -c:v libx264 -y videos/$file.mov
python src/main.py --play_games 1 --record_screen_path videos/$game --load_weights $snapshot $rom $*
avconv -r 60 -i videos/$game/%06d.png -f mov -c:v libx264 -y videos/$file.mov
