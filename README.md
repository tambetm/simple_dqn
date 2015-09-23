# Simple DQN

Deep Q-learning agent for replicating DeepMind's results in paper ["Human-level control through deep reinforcement learning"](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html).

This code grew out of frustration working with [original DeepMind code](https://github.com/tambetm/DeepMind-Atari-Deep-Q-Learner). It is designed to be simple, fast and easy to extend. In particular:
 * It's Python :).
 * New [ALE Python interface](https://github.com/bbitmaster/ale_python_interface/wiki/Code-Tutorial) is used.
 * [Fastest convolutions](https://github.com/soumith/convnet-benchmarks) from [Neon deep learning library](http://neon.nervanasys.com/docs/latest/index.html).
 * Every screen is kept only once in replay memory, fast minibatch sampling with Numpy array slicing.
 * The number of array and datatype conversions is minimized.

Restriction from Neon is, that it currently works only on Maxwell architecture GPU-s. Hopefully this [will change](https://github.com/NervanaSystems/neon/issues/80). You can still simulate playing from pretrained models using just CPU, see the example below.

## Installation

Currently only instructions for Ubuntu are provided. For OS X refer to [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/doc/manual/manual.pdf) and [Neon](http://neon.nervanasys.com/docs/latest/user_guide.html#installation) documentation.

### Neon

Install prerequisites:
```
sudo apt-get install libhdf5-dev libyaml-dev libopencv-dev pkg-config
sudo apt-get install python python-dev python-pip python-virtualenv
```
Check out and compile the code:
```
git clone https://github.com/NervanaSystems/neon.git
cd neon
make
```
Neon installs itself into virtual environment in `.venv`. You need to activate that to import Neon in Python:
```
source .venv/bin/activate
```

### Arcade Learning Environment

Install prerequisites:
```
sudo apt-get install cmake libsdl1.2-dev
```
Check out and compile the code:
```
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
make -j 4
```
Install Python library (assuming you have activated Neon virtual environment):
```
pip install .
```

### Simple DQN

Prerequisities:
```
pip install numpy argparse logging
```
Neon virtual environment already contains those libraries, but they are listed here, just in case.

Also you need OpenCV, which is pain to install to virtual environment. I ended up with this hack:
```
sudo apt-get install python-opencv
ln -s /usr/lib/python2.7/dist-packages/cv2.so NEON_HOME/.venv/lib/python2.7/site-packages/
```
Then just check out the code:
```
git clone https://github.com/tambetm/simple_dqn.git
cd simple_dqn
```

## Running the code

### Training

To run training for Pong:
```
./train.sh roms/pong.bin
```
There are plethora of options, just run `./train.sh --help` to see them. While training, the network weights are saved to `snapshots` folder after each epoch. Name of the file is `<rom_name>_<epoch_nr>.pkl`. 

### Resuming training

You can resume training by running 
```
./train.sh roms/pong.bin --load_weights snapshots/pong_10.pkl
```
Pay attention, that exploration rate starts from 1 and replay memory is empty. You may want to start with lower exploration rate, e.g. for epoch 10 usual exploration rate would be 1 - (1 - 0.1) * (10 * 50000 / 1000000) = 0.55. Add  `--exploration_rate_start 0.55` to the command line.

### Only testing

To run only testing on pre-trained model:
```
./test.sh roms/pong.bin --load_weights snapshots/pong_61.pkl --minimal_action_set
```
Option `--minimal_action_set` must be added to current Pong model, because it was trained with minimal actions (left, right and nothing).

### Play one game with visualization

To see the game screen while playing run
```
./play.sh roms/pong.bin --load_weights snapshots/pong_61.pkl --minimal_action_set
```
You can do this even without GPU, by adding `--backend cpu` to command line.

### Record game video

To play one game and record video
```
./record.sh roms/pong.bin --load_weights snapshots/pong_61.pkl --minimal_action_set
```
First game frames are extracted to `videos/pong` folder as PNG files. Then `avconv` is used to convert these into video.

### Profiling

There are two additional scripts for profiling:
 * `profile_train.sh` - runs Pong game 1000 steps in training mode. This is for figuring out bottlenecks in minibatch sampling and network training code. Prediction is disabled by setting exploration rate to 1.
 * `profile_test.sh` - runs Pong game 1000 steps in testing mode. This is for figuring out bottlenecks in prediction code. Exploration is disabled by setting exploration rate to 0.

## Credits

This wouldn't have happened without inspiration and preceding work from my fellow PhD students [Kristjan Korjus](https://github.com/kristjankorjus), [Ardi Tampuu](https://github.com/RDTm), [Ilya Kuzovkin](https://github.com/kuz) and Raul Vicente from [Computational Neuroscience lab](http://neuro.cs.ut.ee/) in [University of Tartu](http://www.ut.ee/en), [Estonia](https://e-estonia.com/). Also I would like to thank [Nathan Sprague](https://github.com/spragunr) and other nice folks at [Deep Q-Learning list](https://groups.google.com/forum/#!forum/deep-q-learning).
