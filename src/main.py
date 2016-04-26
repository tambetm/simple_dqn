import logging
logging.basicConfig(format='%(asctime)s %(message)s')

from environment import Environment, GymEnvironment
from replay_memory import ReplayMemory
from deepqnetwork import DeepQNetwork
from agent import Agent
from statistics import Statistics
import random
import argparse
import sys
import os

parser = argparse.ArgumentParser()

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

envarg = parser.add_argument_group('Environment')
envarg.add_argument("rom_file", help="ROM bin file or env id such as Breakout-v0 if training with Open AI Gym")
envarg.add_argument("--display_screen", type=str2bool, default=False, help="Display game screen during training and testing.")
#envarg.add_argument("--sound", type=str2bool, default=False, help="Play (or record) sound.")
envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
envarg.add_argument("--repeat_action_probability", type=float, default=0, help="Probability, that chosen action will be repeated. Otherwise random action is chosen during repeating.")
envarg.add_argument("--minimal_action_set", dest="minimal_action_set", type=str2bool, default=True, help="Use minimal action set.")
envarg.add_argument("--color_averaging", type=str2bool, default=True, help="Perform color averaging with previous frame.")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")
envarg.add_argument("--record_screen_path", help="Record game screens under this path. Subfolder for each game is created.")
envarg.add_argument("--record_sound_filename", help="Record game sound in this file.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")
memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")
memarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
memarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--target_steps", type=int, default=10000, help="Copy main network to target network after this many steps.")

#netarg.add_argument("--rescale_r", type=str2bool, help="Rescale rewards.")
#missing: bufferSize=512,valid_size=500,min_reward=-1,max_reward=1

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

nvisarg = parser.add_argument_group('Visualization')
nvisarg.add_argument("--visualization_filters", type=int, default=4, help="Number of filters to visualize from each convolutional layer.")
nvisarg.add_argument("--visualization_file", help="Write layer visualization to this file.")

mainarg = parser.add_argument_group('Main loop')
mainarg.add_argument("--random_steps", type=int, default=50000, help="Populate replay memory with random steps before starting learning.")
mainarg.add_argument("--train_steps", type=int, default=250000, help="How many training steps per epoch.")
mainarg.add_argument("--test_steps", type=int, default=125000, help="How many testing steps after each epoch.")
mainarg.add_argument("--epochs", type=int, default=200, help="How many epochs to run.")
mainarg.add_argument("--play_games", type=int, default=0, help="How many games to play, suppresses training and testing.")
mainarg.add_argument("--load_weights", help="Load network from file.")
mainarg.add_argument("--save_weights_prefix", help="Save network to given file. Epoch and extension will be appended.")
mainarg.add_argument("--csv_file", help="Write training progress to this file.")

comarg = parser.add_argument_group('Common')
comarg.add_argument("--train_gym", type=str2bool, default=False, help="Whether to train agent using OpenAI Gym")
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(args.log_level)

if args.random_seed:
  random.seed(args.random_seed)

# instantiate classes
env = GymEnvironment(args.rom_file, args) if args.train_gym else Environment(args.rom_file, args)
mem = ReplayMemory(args.replay_size, args)
net = DeepQNetwork(env.numActions(), args)
agent = Agent(env, mem, net, args)
stats = Statistics(agent, net, mem, env, args)

if args.load_weights:
  logger.info("Loading weights from %s" % args.load_weights)
  net.load_weights(args.load_weights)

if args.play_games:
  logger.info("Playing for %d game(s)" % args.play_games)
  stats.reset()
  agent.play(args.play_games)
  stats.write(0, "play")
  if args.visualization_file:
    from visualization import visualize
    # use states recorded during gameplay. NB! Check buffer size, that it can accomodate one game!
    states = [agent.mem.getState(i) for i in xrange(agent.history_length, agent.mem.current - agent.random_starts)]
    logger.info("Collected %d game states" % len(states))
    import numpy as np
    states = np.array(states)
    states = states / 255.
    visualize(net.model, states, args.visualization_filters, args.visualization_file)
  sys.exit()

if args.random_steps:
  # populate replay memory with random steps
  logger.info("Populating replay memory with %d random moves" % args.random_steps)
  stats.reset()
  agent.play_random(args.random_steps)
  stats.write(0, "random")

# loop over epochs
for epoch in xrange(args.epochs):
  logger.info("Epoch #%d" % (epoch + 1))

  if args.train_steps:
    logger.info(" Training for %d steps" % args.train_steps)
    stats.reset()
    agent.train(args.train_steps, epoch)
    stats.write(epoch + 1, "train")

    if args.save_weights_prefix:
      filename = args.save_weights_prefix + "_%d.pkl" % (epoch + 1)
      logger.info("Saving weights to %s" % filename)
      net.save_weights(filename)

  if args.test_steps:
    logger.info(" Testing for %d steps" % args.test_steps)
    stats.reset()
    agent.test(args.test_steps, epoch)
    stats.write(epoch + 1, "test")

stats.close()
logger.info("All done")
