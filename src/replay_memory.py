import numpy as np
import random

class ReplayMemory:
  def __init__(size = 1000000, history_length = 4, screen_dims = (84,84)):
    self.actions = np.empty(size)
    self.rewards = np.empty(size)
    self.screens = np.empty((size,) + screen_dims)
    self.terminals = np.empty(size)
    self.size = size
    self.history = history_length
    self.dims = screen_dims
    self.count = 0
    self.current = 0

  def add(self, action, reward, screen, terminal):
    assert screen.shape == self.dims
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.current = (self.current + 1) % self.size
    self.count = max(self.count, self.current)
  
  def getState(self, index):
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if self.index >= history - 1:
      # use faster slicing
      return self.screens[index - (self.history - 1):index, ...]
    else
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(history))]
      return self.screens[indexes, ...]

  def getCurrentState(self):
    return getState(self.current - 1)

  def getMinibatch(self, batch_size):
    # sample random indexes
    indexes = random.sample(xrange(self.count), batch_size)
    prestates = np.ndarray([ self.getState(i - 1) for i in indexes ])
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    poststates = np.ndarray([ self.getState(i) for i in indexes ])
    terminals = self.terminals[indexes]
    return prestates, actions, rewards, poststates, terminals
