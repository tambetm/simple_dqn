from environment import Environment
from replay_memory import ReplayMemory
from deepqnetwork import DeepQNetwork
import random
import numpy as np
import argparse

class Agent:
  def __init__(environment, replay_memory, deep_q_network, 
      exploration_start = 1, exploration_end = 0.1, exploration_steps = 50000, learn_start_steps = 50000):
    self.env = environment
    self.mem = replay_memory
    self.net = deep_q_network
    self.num_actions = env.numActions()

    self.exploration_start = exploration_start
    self.exploration_end = exploration_end
    self.exploration_steps = exploration_steps
    self.learn_start_steps = learn_start_steps
    self.total_train_steps = 0

  def step(exploration_rate):
    # exploration rate determines the probability of random moves
    if random.random() < exploration_rate:
      action = random.randint(0, num_actions - 1)
    else
      # otherwise choose action with highest Q-value
      state = self.mem.getCurrentState()
      qvalues = self.net.predict(state)
      action = np.argmax(qvalues)
    # perform the action
    reward, screen, terminal = self.env.act(action)
    # add transition to the memory (otherwise we wouldn't have current state)
    self.mem.add(action, reward, screen, terminal)
    # restart the game if over
    if terminal:
      self.env.restart()

  def train(train_steps = 100000):
    for i in xrange(train_steps):
      # calculate decaying exploration rate
      if self.total_train_steps < self.exploration_steps:
        exploration_rate = self.exploration_start - self.total_train_steps * (self.exploration_start - self.exploration_end) / self.exploration_steps
      else:
        exploration_rate = self.exploration_end
      # perform game step
      self.step(exploration_rate)
      # if memory contains enough transitions
      if self.mem.count > self.learn_start_steps:
        # sample minibatch
        minibatch = self.mem.getMinibatch(self.net.batch_size)
        # train the network
        self.net.train(minibatch)
      
      self.total_train_steps += 1

  def test(test_steps = 10000, exploration_rate = 0.05):
    for i in xrange(test_steps):
      # perform game step
      self.step(exploration_rate)


def main():
  parser = argparse.ArgumentParser()

  envarg = parser.add_argument_group('Environment')
  envarg.add_argument("rom_file")
  envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
  #envarg.add_argument("--random_starts", type=int, default=30, help="Perform random number of moves in the beginning of game.")
  envarg.add_argument("--resize_width", type=int, default=84, help="Screen width after resize.")
  envarg.add_argument("--resize_height", type=int, default=84, help="Screen height after resize.")

  memarg = parser.add_argument_group('Replay memory')
  memarg.add_argument("--replay_memory", type=int, default=1000000, help="Maximum size of replay memory.")
  memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

  netarg = parser.add_argument_group('Q-learning network')
  netarg.add_argument("--lr", type=float, default=0.00025, help="Learning rate.")
  netarg.add_argument("--discount", type=float, default=0.99, help="Discount rate for future rewards.")
  netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
  #netarg.add_argument("--rescale_r", action="store_true", help="Rescale rewards.")
  #bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1

  antarg = parser.add_argument_group('Agent')
  #antarg.add_argument("--n_replay", type=int, default=1, help="Number of times to sample minibatch during training.")
  #antarg.add_argument("--update_freq", type=int, default=4, help="Perform training after this many game steps.")
  antarg.add_argument("--learn_start", type=int, default=50000, help="Size of replay memory when learning starts.")
  antarg.add_argument("--exploration_start", type=float, default=1, help="Exploration rate at the beginning of decay.")
  antarg.add_argument("--exploration_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
  antarg.add_argument("--exploration_steps", type=float, default=1000000, help="How many steps to decay the exploration rate.")
  antarg.add_argument("--epochs", type=int, default=100, help="How many epochs to run.")
  antarg.add_argument("--train_steps", type=int, default=50000, help="How many training steps per epoch.")
  antarg.add_argument("--test_steps", type=int, default=10000, help="How many testing steps per epoch.")
  #antarg.add_argument("--prog_freq", type=int, default=10000, help="How often to print out training statistics.")
  #antarg.add_argument("--save_freq", type=int, default=250000, help="How often to save snapshot of the network.")
  #antarg.add_argument("--network", help="Load network from file.")

  comarg = parser.add_argument_group('Common')
  comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")

  args = parser.parse_args()

  if args.random_seed:
    random.seed(args.random_seed)

  env = Environment(args.rom_file, dims = (args.resize_height, args.resize_width), frame_skip = args.frame_skip, random_seed = args.random_seed)
  mem = ReplayMemory(args.replay_memory, history_length = args.history_length, dims=(args.resize_height, args.resize_width))
  net = DeepQNetwork(env.numActions(), learning_rate = args.lr, discount_rate = args.discount, 
      batch_size = args.batch_size)
  agent = Agent(env, mem, net, learn_start_steps = args.learn_start, 
      exploration_end = args.exploration_end, exploration_steps = args.exploration_steps)

  for i in xrange(args.epochs):
    agent.train(args.train_steps)
    agent.test(args.test_steps)

if __name__ == "__main__":
    main()