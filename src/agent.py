from environment import Environment
from replay_memory import ReplayMemory
from deepqnetwork import DeepQNetwork
import numpy as np
import random
import time
import sys
import logging
logger = logging.getLogger(__name__)

class Agent:
  def __init__(self, environment, replay_memory, deep_q_network, args):
    self.env = environment
    self.mem = replay_memory
    self.net = deep_q_network
    self.num_actions = self.env.numActions()
    self.random_starts = args.random_starts
    self.history_length = args.history_length

    self.exploration_rate_start = args.exploration_rate_start
    self.exploration_rate_end = args.exploration_rate_end
    self.exploration_decay_steps = args.exploration_decay_steps
    self.exploration_rate_test = args.exploration_rate_test
    self.total_train_steps = 0

    self.train_frequency = args.train_frequency
    self.train_repeat = args.train_repeat

    self.callback = None

  def _restartRandom(self):
    self.env.restart()
    # perform random number of dummy actions to produce more random game dynamics
    for i in xrange(random.randint(self.history_length, self.random_starts) + 1):
      reward = self.env.act(0)
      screen = self.env.getScreen()
      terminal = self.env.isTerminal()
      assert not terminal, "terminal state occurred during random initialization"
      # add dummy states to replay memory to guarantee history_length screens
      self.mem.add(0, reward, screen, terminal)

  def _exploration_rate(self):
    # calculate decaying exploration rate
    if self.total_train_steps < self.exploration_decay_steps:
      return self.exploration_rate_start - self.total_train_steps * (self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
    else:
      return self.exploration_rate_end

  def step(self, exploration_rate, training = False):
    # exploration rate determines the probability of random moves
    if random.random() < exploration_rate:
      action = random.randrange(self.num_actions)
      logger.debug("Random action = %d" % action)
    else:
      # otherwise choose action with highest Q-value
      state = self.mem.getCurrentState()
      # for convenience getCurrentState() returns minibatch
      # where first item is the current state
      qvalues = self.net.predict(state)
      assert len(qvalues[0]) == self.num_actions
      # choose highest Q-value of first state
      action = np.argmax(qvalues[0])
      logger.debug("Predicted action = %d" % action)

    # perform the action
    reward = self.env.act(action)
    screen = self.env.getScreen()
    terminal = self.env.isTerminal(training)

    # print reward
    if reward <> 0:
      logger.debug("Reward: %d" % reward)

    # always add transition to the memory (otherwise we wouldn't have current state)
    self.mem.add(action, reward, screen, terminal)

    # restart the game if over
    if terminal:
      logger.debug("Terminal state, restarting")
      self._restartRandom()

    # call callback to record statistics
    if self.callback:
      self.callback.on_step(action, reward, terminal, screen, exploration_rate)

    return terminal

  def play_random(self, random_steps):
    # play given number of steps
    for i in xrange(random_steps):
      # use exploration rate 1 = completely random
      self.step(1)

  def train(self, train_steps, epoch = 0):
    # do not do restart here, continue from testing
    # initially there should be enough random steps to produce current state
    assert self.mem.count >= self.history_length, "Not enough history in replay memory, increase random steps."
    # play given number of steps
    for i in xrange(train_steps):
      # perform game step, with training = True
      self.step(self._exploration_rate(), True)
      # train after every train_frequency steps
      if i % self.train_frequency == 0:
        # train for train_repeat times
        for j in xrange(self.train_repeat):
          # sample minibatch
          minibatch = self.mem.getMinibatch()
          # train the network
          self.net.train(minibatch, epoch)
      # increase number of training steps for epsilon decay
      self.total_train_steps += 1

  def test(self, test_steps, epoch = 0):
    # just make sure there is history_length screens to form a state
    self._restartRandom()
    # play given number of steps
    for i in xrange(test_steps):
      # perform game step
      self.step(self.exploration_rate_test)

  def play(self, num_games):
    # just make sure there is history_length screens to form a state
    self._restartRandom()
    for i in xrange(num_games):
      # play until terminal state
      while not self.step(self.exploration_rate_test):
        pass
