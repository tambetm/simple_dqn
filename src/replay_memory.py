import numpy as np
import random
import logging
import gpu_backend
logger = logging.getLogger(__name__)

class ReplayMemory:
  def __init__(self, size, args):
    self.size = size
    self.be = gpu_backend.initialize_backend(args)
    # preallocate memory
    self.actions = np.empty(self.size, dtype = np.uint8)
    self.rewards = np.empty(self.size, dtype = np.integer)
    self.screens = np.empty((self.size, args.screen_height, args.screen_width), dtype=np.uint8)
    self.terminals = np.empty(self.size, dtype = np.bool)
    self.history_length = args.history_length
    self.dims = (args.screen_height, args.screen_width)
    self.batch_size = args.batch_size
    self.min_reward = args.min_reward
    self.max_reward = args.max_reward
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = self.be.empty((self.batch_size, self.history_length, ) + self.dims, dtype = np.uint8)
    self.poststates = self.be.empty((self.batch_size, self.history_length, ) + self.dims, dtype = np.uint8)
    # Save slices into arrays so we don't have to compute them each time
    #self.screens_view = [self.screens[i, ...] for i in xrange(self.size)]
    self.prestates_view = [self.prestates[i, ...] for i in xrange(self.batch_size)]
    self.poststates_view = [self.poststates[i, ...] for i in xrange(self.batch_size)]

    logger.info("Replay memory size: %d" % self.size)

  def add(self, action, reward, screen, terminal):
    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    # clip reward between -1 and 1
    if self.min_reward and reward < self.min_reward:
      #logger.debug("Smaller than min_reward: %d" % reward)
      reward = max(reward, self.min_reward)
      #logger.info("After clipping: %d" % reward)
    if self.max_reward and reward > self.max_reward:
      #logger.debug("Bigger than max_reward: %d" % reward)
      reward = min(reward, self.max_reward)
      #logger.info("After clipping: %d" % reward)
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.size
    #logger.debug("Memory count %d" % self.count)

  
  def getState(self, index):
    assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def getCurrentState(self):
    # reuse first row of prestates in minibatch to minimize memory consumption
    self.prestates_view[0][:] = self.getState(self.current - 1)
    return self.prestates

  def getMinibatch(self):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break

      self.prestates_view[len(indexes)][:] = self.getState(index - 1)
      self.poststates_view[len(indexes)][:] = self.getState(index)
      indexes.append(index)

    # copy actions, rewards and terminals with direct slicing
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    return self.prestates, actions, rewards, self.poststates, terminals
