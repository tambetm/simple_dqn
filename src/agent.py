from environment import Environment
from replay_memory import ReplayMemory
from deepqnetwork import DeepQNetwork
import random
import numpy as np
import argparse
import logging
import time
import sys
logger = logging.getLogger(__name__)

class Agent:
  def __init__(self, environment, replay_memory, deep_q_network, 
      exploration_start = 1, exploration_end = 0.1, exploration_steps = 50000, learn_start_steps = 50000):
    self.env = environment
    self.mem = replay_memory
    self.net = deep_q_network
    self.num_actions = self.env.numActions()

    self.exploration_start = exploration_start
    self.exploration_end = exploration_end
    self.exploration_steps = exploration_steps
    self.learn_start_steps = learn_start_steps
    self.total_train_steps = 0

  def exploration_rate(self):
    # calculate decaying exploration rate
    if self.total_train_steps < self.exploration_steps:
      return self.exploration_start - self.total_train_steps * (self.exploration_start - self.exploration_end) / self.exploration_steps
    else:
      return self.exploration_end

  def step(self, exploration_rate):
    logger.debug("Exploration rate %f" % exploration_rate)
    # exploration rate determines the probability of random moves
    if random.random() < exploration_rate:
      action = random.randint(0, self.num_actions - 1)
      logger.debug("Random action = %d" % action)
    else:
      # otherwise choose action with highest Q-value
      state = self.mem.getCurrentState()
      qvalues = self.net.predict(state)
      action = np.argmax(qvalues)
      logger.debug("Predicted action = %d" % action)
    # perform the action
    reward, screen, terminal = self.env.act(action)
    # print reward
    if reward <> 0:
      logger.debug("Reward: %d" % reward)
      self.game_reward += reward
    # add transition to the memory (otherwise we wouldn't have current state)
    self.mem.add(action, reward, screen, terminal)
    # restart the game if over
    if terminal:
      self.env.restart()
      logger.debug("Game over, restarting")
      # collect statistics
      self.games += 1
      self.rewards.append(self.game_reward)
      self.game_reward = 0

  def train(self, train_steps = 100000, epoch = 0):
    for i in xrange(train_steps):
      # perform game step
      self.step(self.exploration_rate())
      # if memory contains enough transitions
      if self.mem.count > self.learn_start_steps:
        # sample minibatch
        minibatch = self.mem.getMinibatch()
        # train the network
        self.net.train(minibatch, epoch)
        # increase number of training steps
        self.total_train_steps += 1

  def test(self, test_steps = 10000, exploration_rate = 0.05, epoch = 0):
    for i in xrange(test_steps):
      # perform game step
      self.step(exploration_rate)

  def reset_stats(self):
    self.game_reward = 0
    self.games = 0
    self.rewards = []
    self.time = time.time()

  def print_stats(self, phase):
    logger.info("%s - games: %d, average_reward: %f, replay_memory_size: %d, total_train_steps: %d, exploration_rate: %f, time_elapsed: %d" % 
        (phase, self.games, np.mean(self.rewards), self.mem.count, self.total_train_steps, self.exploration_rate(), time.time() - self.time))
    
  def main(self):
    for epoch in xrange(args.epochs):
      logger.info("Epoch %d" % (epoch + 1))
      
      self.reset_stats()
      agent.train(args.train_steps, epoch)
      self.print_stats(" Train")

      self.reset_stats()
      agent.test(args.test_steps, args.exploration_test, epoch)
      self.print_stats("  Test")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  envarg = parser.add_argument_group('Environment')
  envarg.add_argument("rom_file")
  envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
  #envarg.add_argument("--random_starts", type=int, default=30, help="Perform random number of moves in the beginning of game.")
  envarg.add_argument("--resize_width", type=int, default=84, help="Screen width after resize.")
  envarg.add_argument("--resize_height", type=int, default=84, help="Screen height after resize.")

  memarg = parser.add_argument_group('Replay memory')
  memarg.add_argument("--replay_memory", type=int, default=200000, help="Maximum size of replay memory.")
  memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

  netarg = parser.add_argument_group('Q-learning network')
  netarg.add_argument("--lr", type=float, default=0.00025, help="Learning rate.")
  netarg.add_argument("--discount", type=float, default=0.99, help="Discount rate for future rewards.")
  netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
  #netarg.add_argument("--rescale_r", action="store_true", help="Rescale rewards.")
  #missing: bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1
  netarg.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='backend type')
  netarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
  netarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')
  netarg.add_argument('--rounding', const=True, type=int, nargs='?', default=False, help='use stochastic rounding [will round to BITS number of bits if specified]')

  antarg = parser.add_argument_group('Agent')
  #antarg.add_argument("--n_replay", type=int, default=1, help="Number of times to sample minibatch during training.")
  #antarg.add_argument("--update_freq", type=int, default=4, help="Perform training after this many game steps.")
  antarg.add_argument("--learn_start", type=int, default=50000, help="Size of replay memory when learning starts.")
  antarg.add_argument("--exploration_start", type=float, default=1, help="Exploration rate at the beginning of decay.")
  antarg.add_argument("--exploration_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
  antarg.add_argument("--exploration_steps", type=float, default=1000000, help="How many steps to decay the exploration rate.")
  antarg.add_argument("--exploration_test", type=float, default=0.05, help="Exploration rate used during testing.")
  antarg.add_argument("--epochs", type=int, default=100, help="How many epochs to run.")
  antarg.add_argument("--train_steps", type=int, default=50000, help="How many training steps per epoch.")
  antarg.add_argument("--test_steps", type=int, default=10000, help="How many testing steps after each epoch.")
  #antarg.add_argument("--prog_freq", type=int, default=10000, help="How often to print out training statistics.")
  #antarg.add_argument("--save_freq", type=int, default=250000, help="How often to save snapshot of the network.")
  #antarg.add_argument("--network", help="Load network from file.")

  comarg = parser.add_argument_group('Common')
  comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
  comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
  args = parser.parse_args()

  if args.random_seed:
    random.seed(args.random_seed)

  logger.setLevel(args.log_level)

  env = Environment(args.rom_file, 
      dims = (args.resize_height, args.resize_width), 
      frame_skip = args.frame_skip, 
      random_seed = args.random_seed)

  mem = ReplayMemory(args.replay_memory, 
      history_length = args.history_length, 
      screen_dims = (args.resize_height, args.resize_width),
      batch_size = args.batch_size)

  net = DeepQNetwork(env.numActions(), 
      learning_rate = args.lr, 
      discount_rate = args.discount, 
      batch_size = args.batch_size,
      # Neon options
      backend = args.backend, 
      random_seed = args.random_seed, 
      device_id = args.device_id, 
      datatype = args.datatype, 
      stochastic_round = args.rounding)

  agent = Agent(env, mem, net, 
      learn_start_steps = args.learn_start, 
      exploration_start = args.exploration_start, 
      exploration_end = args.exploration_end, 
      exploration_steps = args.exploration_steps)

  agent.main()
