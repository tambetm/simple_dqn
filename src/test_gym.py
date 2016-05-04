import gym
import random
import argparse
import numpy as np
import cv2
from environment import GymEnvironment
from deepqnetwork import DeepQNetwork
from agent import Agent

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("env_id", help="Which atari game to test such as Breakout-v0")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")
envarg.add_argument("--display", type=str2bool, default=False, help="Display screen during testing.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--target_steps", type=int, default=10000, help="Copy main network to target network after this many steps.")
netarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
netarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")
netarg.add_argument("--batch_norm", type=str2bool, default=False, help="Use batch normalization in all layers.")

neonarg = parser.add_argument_group('Neon')
neonarg.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='backend type')
neonarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
neonarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')
neonarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False, help='use stochastic rounding [will round to BITS number of bits if specified]')

antarg = parser.add_argument_group('Agent')
antarg.add_argument("--exploration_rate_start", type=float, default=1, help="Exploration rate at the beginning of decay.")
antarg.add_argument("--exploration_rate_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
antarg.add_argument("--exploration_decay_steps", type=float, default=1000000, help="How many steps to decay the exploration rate.")
antarg.add_argument("--exploration_rate_test", type=float, default=0.05, help="Exploration rate used during testing.")
antarg.add_argument("--train_frequency", type=int, default=4, help="Perform training after this many game steps.")
antarg.add_argument("--train_repeat", type=int, default=1, help="Number of times to sample minibatch during training.")
antarg.add_argument("--random_starts", type=int, default=30, help="Perform max this number of dummy actions after game restart, to produce more random game dynamics.")

mainarg = parser.add_argument_group('Main loop')
mainarg.add_argument("--load_weights", help="Load network from file.")
mainarg.add_argument("--save_weights_prefix", help="Save network to given file. Epoch and extension will be appended.")

comarg = parser.add_argument_group('Common')
comarg.add_argument("output_folder", help="Where to write results to.")
comarg.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to test.")
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
args = parser.parse_args()

if args.random_seed:
  random.seed(args.random_seed)

env = GymEnvironment(args.env_id, args)
net = DeepQNetwork(env.numActions(), args)
mem = None
agent = Agent(env, mem, net, args)

if args.load_weights:
  print "Loading weights from %s" % args.load_weights
  net.load_weights(args.load_weights)

env.gym.monitor.start(args.output_folder, force=True)
agent.play(args.num_episodes)
env.gym.monitor.close()
