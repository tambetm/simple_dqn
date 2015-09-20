import logging
logging.basicConfig(format='%(message)s')

from environment import Environment
from replay_memory import ReplayMemory
from deepqnetwork import DeepQNetwork
from agent import Agent
from statistics import Statistics
import random
import argparse
import sys
import os

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("rom_file")
envarg.add_argument("--display_screen", action="store_true", default=False, help="Display game screen during training and testing.")
envarg.add_argument("--minimal_action_set", action="store_true", default=False, help="Use minimal action set instead of full.")
envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
envarg.add_argument("--color_averaging", action="store_true", default=True, help="Perform color averaging with previous frame.")
#envarg.add_argument("--random_starts", type=int, default=30, help="Perform random number of moves in the beginning of game.")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")
memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
#netarg.add_argument("--rmsprop_decay_rate", type=float, default=0.95, help="Decay rate for RMSProp algorithm.")
#netarg.add_argument("--clip_error", action="store_true", default=True, help="Clip error term in update between -1 and 1.")

#netarg.add_argument("--rescale_r", action="store_true", help="Rescale rewards.")
#missing: bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1

neonarg = parser.add_argument_group('Neon')
neonarg.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='backend type')
neonarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
neonarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')
neonarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False, help='use stochastic rounding [will round to BITS number of bits if specified]')

antarg = parser.add_argument_group('Agent')
antarg.add_argument("--exploration_start", type=float, default=1, help="Exploration rate at the beginning of decay.")
antarg.add_argument("--exploration_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
antarg.add_argument("--exploration_steps", type=float, default=1000000, help="How many steps to decay the exploration rate.")
antarg.add_argument("--exploration_test", type=float, default=0.05, help="Exploration rate used during testing.")
#antarg.add_argument("--n_replay", type=int, default=1, help="Number of times to sample minibatch during training.")
#antarg.add_argument("--update_freq", type=int, default=4, help="Perform training after this many game steps.")

antarg.add_argument("--random_steps", type=int, default=50000, help="Populate replay memory with random steps before starting learning.")
antarg.add_argument("--train_steps", type=int, default=50000, help="How many training steps per epoch.")
antarg.add_argument("--test_steps", type=int, default=10000, help="How many testing steps after each epoch.")
antarg.add_argument("--epochs", type=int, default=1000, help="How many epochs to run.")
antarg.add_argument("--load_weights", help="Load network from file.")
antarg.add_argument("--save_weights_path", default="snapshots", help="Save network to path. File name will be rom name + epoch.")
#antarg.add_argument("--prog_freq", type=int, default=10000, help="How often to print out training statistics.")
#antarg.add_argument("--save_freq", type=int, default=250000, help="How often to save snapshot of the network.")

comarg = parser.add_argument_group('Common')
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
args = parser.parse_args()

assert args.random_steps >= args.history_length, "random_steps must be bigger or equal than history_length"

logger = logging.getLogger()
logger.setLevel(args.log_level)

if args.random_seed:
  random.seed(args.random_seed)

# instantiate classes
env = Environment(args)
mem = ReplayMemory(args)
net = DeepQNetwork(env.numActions(), args)
agent = Agent(env, mem, net, args)
stats = Statistics(env, mem, net, agent)

if args.load_weights:
  logger.info("Loading weights from %s" % args.load_weights)
  net.load_weights(args.load_weights)

# populate replay memory with random steps
logger.info("Populating replay memory with %d random moves" % args.random_steps)
stats.reset()
agent.play_random(args.random_steps)
stats.log()

# loop over epochs
for epoch in xrange(args.epochs):
  logger.info("Epoch #%d" % (epoch + 1))
  
  logger.info(" Training for %d steps" % args.train_steps)
  stats.reset()
  agent.train(args.train_steps, epoch)
  stats.log()

  if args.save_weights_path:
    rom_name, ext = os.path.splitext(os.path.basename(args.rom_file))
    filename = os.path.join(args.save_weights_path, "%s_%d.pkl" % (rom_name, epoch + 1))
    logger.info("Saving weights to %s" % filename)
    net.save_weights(filename)

  logger.info(" Testing for %d steps" % args.test_steps)
  stats.reset()
  agent.test(args.test_steps, args.exploration_test, epoch)
  stats.log()
