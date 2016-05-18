# Simple DQN

Deep Q-learning agent for replicating DeepMind's results in paper ["Human-level control through deep reinforcement learning"](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). It is designed to be simple, fast and easy to extend. In particular:
 * It's Python :).
 * Uses ALE [native Python interface](https://github.com/bbitmaster/ale_python_interface/wiki/Code-Tutorial).
 * Updated to support training and testing with [OpenAI Gym](https://gym.openai.com/).
 * [Fastest convolutions](https://github.com/soumith/convnet-benchmarks) from [Neon deep learning library](http://neon.nervanasys.com/docs/latest/index.html).
 * Every screen is kept only once in replay memory, fast minibatch sampling with Numpy array slicing.
 * The number of array and datatype conversions is minimized.

See the example gameplay videos for Breakout, Pong, Seaquest and Space Invaders:

[![Breakout](http://img.youtube.com/vi/KkIf0Ok5GCE/default.jpg)](https://youtu.be/KkIf0Ok5GCE)
[![Pong](http://img.youtube.com/vi/0ZlgrQS3krg/default.jpg)](https://youtu.be/0ZlgrQS3krg)
[![Seaquest](http://img.youtube.com/vi/b6g6A_n8mUk/default.jpg)](https://youtu.be/b6g6A_n8mUk)
[![Space Invaders](http://img.youtube.com/vi/Qvco7ufsX_0/default.jpg)](https://youtu.be/Qvco7ufsX_0)

## Installation

Currently only instructions for Ubuntu are provided. For OS X refer to [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/doc/manual/manual.pdf) and [Neon](http://neon.nervanasys.com/docs/latest/installation.html) documentation.

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

If you want to try out the filter visualization, use latest Neon and run `make -e VIS=true` instead. If you’ve already installed Neon without enabling visualization dependencies you’ll need to `touch vis_requirements.txt` prior to the `make -e VIS=true` call to ensure virtualenv Python dependencies get triggered.

Neon installs itself into virtual environment in `.venv`. You need to activate that to import Neon in Python:
```
source .venv/bin/activate
```

### Arcade Learning Environment

You can skip this, if you only plan to use OpenAI Gym.

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

### OpenAI Gym

You can skip this, if you only plan to use Arcade Learning Environment directly.

To install OpenAI Gym:
```
pip install gym
pip install gym[atari]
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
`NEON_HOME` here means the folder where you installed (cloned) Neon.

Then just check out the code:
```
git clone https://github.com/tambetm/simple_dqn
cd simple_dqn
```

### Optional

For plotting install `matplotlib`:
```
pip install matplotlib
```

For producing game videos install `avconv`:
```
sudo apt-get install libav-tools
```

## Running the code

### Training

To run training for Breakout:
```
./train.sh roms/breakout.bin
```

If using OpenAI Gym:
```
./train.sh Breakout-v0 --environment gym
```

There are plethora of options, just run `./train.sh --help` to see them. While training, the network weights are saved to `snapshots` folder after each epoch. Name of the file is `<game>_<epoch_nr>.pkl`. Training statistics are saved to `results/<game>.csv`, see below how to produce plots from it.

### Resuming training

You can resume training by running 
```
./resume.sh snapshots/breakout_10.pkl
```
Pay attention that the replay memory is empty.

### Only testing

To run only testing on a pre-trained model:
```
./test.sh snapshots/breakout_77.pkl
```

To test using OpenAI Gym:
```
./test_gym.sh snapshots/Breakout-v0_77.pkl
```

This saves testing results in folder `results/Breakout-v0`. Now you can then upload your results to OpenAI Gym:
```
./upload_gym.sh results/Breakout-v0 --api_key <your_key>
```
Note that the OpenAI Gym environment differs from the default environment so testing using OpenAI Gym should use a model trained using OpenAI Gym.

### Play one game with visualization

To play one game and show game screen while playing:
```
./play.sh snapshots/breakout_77.pkl
```
You can do this even without GPU, by adding `--backend cpu` to command line. During gameplay you can use following keys:
* `a` - slow down,
* `s` - speed up,
* `m` - manual control mode,
* `[` - volume down,
* `]` - volume up.

Visualization works even in text terminal!

### Record game video

To play one game and record a video:
```
./record.sh snapshots/breakout_77.pkl
```
First game frames are extracted to `videos/<game>` folder as PNG files. Then `avconv` is used to convert these into video, which is saved to `videos/<game>_<epoch_nr>.mov`.

### Plotting results

To plot results:
```
./plot.sh results/breakout.csv
```
This produces `results/breakout.png`, which includes four main figures: average reward per game, number of games per phase (training, test or random), average Q-value of validation set and average network loss. You can customize the plotting result with `--fields` option - list comma separated CSV field names (the first row). For example default results are achieved with `--fields average_reward,meanq,nr_games,meancost`. Order of figures is left to right, top to bottom.

### Visualizing filters

To produce filter visualizations with guided backpropagation:

```
./nvis.sh snapshots/breakout_77.pkl
```

What the filter visualization does:

1. first it plays one game to produce a set of states (one state is 4 frames), 
2. then it finds the states which activate each filter the most,
3. finally it carries out guided backpropagation to show which parts of the screen affect the "activeness" of each filter the most. 

The result is written to file `results/<game>.html`. By default only 4 filters from each convolutional layer are visualized. To see more filters add `--visualization_filters <nr_filters>` to the command line.

NB! Because it is not very clear how to visualize the state consisting of  4 frames, I made a simplification - I'm using only the last 3 frames and putting them to different color channels. So everything that is gray hasn't changed, blue is the most recent change, then green and then red. It is easier to understand if you look at the trace of a ball - it is marked by red-green-blue.

### Nervana Cloud

To train a model with Nervana Cloud, first install and configure [Nervana Cloud](http://doc.cloud.nervanasys.com/docs/latest/ncloud.html).

Assuming the necessary dependencies are installed, run
```
ncloud train src/main.py --args "roms/breakout.bin --save_weights_prefix snapshopts/breakout --csv_file results/breakout.csv" --custom_code_url https://github.com/NervanaSystems/simple_dqn
```
This will download the repo and run the training script.

To test a model using Nervana Cloud run:
```
ncloud train src/main.py --args "roms/breakout.bin --random_steps 0 --train_steps 0 --epochs 1 --load_weights snapshops/breakout_77.pkl" --custom_code_url https://github.com/NervanaSystems/simple_dqn
```

### Profiling

There are three additional scripts for profiling:
 * `profile_train.sh` - runs Pong game 1000 steps in training mode. This is for figuring out bottlenecks in minibatch sampling and network training code. Prediction is disabled by setting exploration rate to 1.
 * `profile_test.sh` - runs Pong game 1000 steps in testing mode. This is for figuring out bottlenecks in prediction code. Exploration is disabled by setting exploration rate to 0.
 * `profile_random.sh` - runs Pong game 1000 steps with random actions. This is for measuring performance of ALE interface, network is not used at all.

### Known differences

 * Testing experiences are stored in replay memory. I initially tought it was harmless and made code simpler, but on a second thought it may affect the replay memory distribution in unexpected ways and it's just not correct to see test data during training.
 * DeepMind considers loss of life as episode end, but only during training.

These are currently in TODO list.

## Credits

This wouldn't have happened without inspiration and preceding work from my fellow PhD students [Kristjan Korjus](https://github.com/kristjankorjus), [Ardi Tampuu](https://github.com/RDTm), [Ilya Kuzovkin](https://github.com/kuz) and [Taivo Pungas](https://github.com/taivop) from [Computational Neuroscience lab](http://neuro.cs.ut.ee/) run by Raul Vicente in [University of Tartu](http://www.ut.ee/en), [Estonia](https://e-estonia.com/). Also I would like to thank [Nathan Sprague](https://github.com/spragunr) and other nice folks at [Deep Q-Learning list](https://groups.google.com/forum/#!forum/deep-q-learning).
