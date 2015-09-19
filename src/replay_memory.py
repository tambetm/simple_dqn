import numpy as np
import random
import logging
logger = logging.getLogger(__name__)

class ReplayMemory:
  def __init__(self, size = 1000000, history_length = 4, screen_dims = (84,84), batch_size = 32):
    self.actions = np.empty(size, dtype = np.integer)
    self.rewards = np.empty(size)
    self.screens = np.empty((size,) + screen_dims)
    self.terminals = np.empty(size)
    self.size = size
    self.history = history_length
    self.dims = screen_dims
    self.batch_size = batch_size
    self.count = 0
    self.current = 0

    self.prestates = np.empty((self.batch_size, self.history) + self.dims)
    self.poststates = np.empty((self.batch_size, self.history) + self.dims)

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
      self.prestates[i, ...] = self.getState(j - 1)
      self.poststates[i, ...] = self.getState(j)
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    return self.prestates, actions, rewards, self.poststates, terminals
