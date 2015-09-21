# Simple DQN

Deep Q-learning agent for replicating DeepMind's results in paper ["Human-level control through deep reinforcement learning"](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html).

This code grew out of frustration working with original DeepMind code. It is designed to be simple, fast and easy to extend. In particular:
 * It's Python :).
 * New [ALE Python interface](https://github.com/bbitmaster/ale_python_interface/wiki/Code-Tutorial) is used.
 * The [fastest convolutions](https://github.com/soumith/convnet-benchmarks) come from [Neon deep learning library](http://neon.nervanasys.com/docs/latest/index.html).
 * Every screen is kept only once in replay memory, fast minibatch sampling with Numpy array slicing.
 * The number of array and datatype conversions is minimized.

## Installation

Currently only instructions for Ubuntu are provided. For OS X refer to [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/doc/manual/manual.pdf) and [Neon](http://neon.nervanasys.com/docs/latest/user_guide.html#installation) documentation.

### Neon

Install prerequisites:
```
sudo apt-get install python python-pip python-virtualenv libhdf5-dev libyaml-dev libopencv-dev pkg-config
```
Check out and compile the code:
```
git clone https://github.com/NervanaSystems/neon.git
cd neon
make
```
Neon installs itself into virtual environment in `.venv`. You need to activate that virtual environment to access Neon from Python:
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
I think Neon virtual environment already contains those libraries, but I list them here, just in case.

Also you need OpenCV, which is pain to install to virtual environment. I ended up with this hack, which seems to work well:
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

To run training for Pong:
```
./train.sh roms/pong.bin
```
There are plethora of options, just run `./train.sh` without arguments to see them. While training, the network weights are saved to `snapshots` folder after each epoch (you may need to create that folder). Name of the file is `<rom_name>_<epoch_nr>.pkl`. You can resume training by running 
```
./train.sh roms/pong.bin --load_weights snapshots/pong_10.pkl
```

## Profiling

There are two additional scripts for profiling:
 * `profile_train.sh` - runs Pong game 1000 steps in training mode. This is for figuring out bottlenecks in minibatch sampling and network training code. Prediction is disabled by setting exploration rate to 1.
 * `profile_test.sh` - runs Pong game 1000 steps in testing mode. This is for figuring out bottlenecks in prediction code.
