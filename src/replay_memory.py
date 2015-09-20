import numpy as np
import random
import logging
logger = logging.getLogger(__name__)

class ReplayMemory:
  def __init__(self, args):
    self.actions = np.empty(args.replay_size, dtype = np.int8)
    self.rewards = np.empty(args.replay_size, dtype = np.integer)
    self.screens = np.empty((args.replay_size, args.screen_height, args.screen_width), dtype = np.int8)
    self.terminals = np.empty(args.replay_size, dtype = np.bool)
    self.size = args.replay_size
    self.history = args.history_length
    self.dims = (args.screen_height, args.screen_width)
    self.batch_size = args.batch_size
    self.count = 0
    self.current = 0

    self.prestates = np.empty((self.batch_size, self.history) + self.dims)
    self.poststates = np.empty((self.batch_size, self.history) + self.dims)

    logger.info("Replay memory size: %d" % self.size)

  def add(self, action, reward, screen, terminal):
    assert screen.shape == self.dims
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.current = (self.current + 1) % self.size
    self.count = max(self.count, self.current)
    logger.debug("Memory count %d" % self.count)

  
  def getState(self, index):
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history - 1:
      # use faster slicing
      return self.screens[index - (self.history - 1):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history))]
      return self.screens[indexes, ...]

  def getCurrentState(self):
    return self.getState(self.current - 1)

  def getMinibatch(self):
    # sample random indexes
    indexes = random.sample(xrange(self.count), self.batch_size)
    for i,j in enumerate(indexes):
      # NB! having index first is fastest in C-order matrices
      self.prestates[i, ...] = self.getState(j - 1)
      self.poststates[i, ...] = self.getState(j)
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    return self.prestates, actions, rewards, self.poststates, terminals
