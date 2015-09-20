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
  def __init__(self, environment, replay_memory, deep_q_network, args):
    self.env = environment
    self.mem = replay_memory
    self.net = deep_q_network
    self.num_actions = self.env.numActions()

    self.exploration_start = args.exploration_start
    self.exploration_end = args.exploration_end
    self.exploration_steps = args.exploration_steps
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

  def play_random(self, random_steps = 50000):
    for i in xrange(random_steps):
      # use exploration rate 1
      self.step(1)

  def train(self, train_steps = 50000, epoch = 0):
    for i in xrange(train_steps):
      # perform game step
      self.step(self.exploration_rate())
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

  def print_stats(self):
    logger.info("games: %d, average_reward: %f, replay_memory_size: %d, total_train_steps: %d, exploration_rate: %f, time_elapsed: %d" % 
        (self.games, np.mean(self.rewards), self.mem.count, self.total_train_steps, self.exploration_rate(), time.time() - self.time))
