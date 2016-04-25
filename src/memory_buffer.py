import numpy as np

class MemoryBuffer:
  # For testing with rl-gym api
  def __init__(self, args):
    self.history_length = args.history_length
    self.dims = (args.screen_height, args.screen_width)
    self.batch_size = args.batch_size
    self.mini_batch = np.zeros((self.batch_size, self.history_length) + self.dims, dtype=np.uint8)
    self.screens = [np.zeros((args.screen_height, args.screen_width), dtype=np.uint8) for _ in xrange(self.history_length)]
    self.state_buffer = np.zeros((self.history_length, args.screen_height, args.screen_width), dtype=np.uint8)
    self.current = 0

  def add(self, observation):
    index = self.current % self.history_length
    self.screens[index][:] = observation
    self.current += 1

  def getState(self):
    assert self.current > self.history_length - 1, "replay memory is empty"
    for i in xrange(self.history_length):
      screen_index = (self.current + i) % self.history_length
      self.state_buffer[i, :, :] = self.screens[screen_index]
    return self.state_buffer

  def getMiniBatch(self):
    self.mini_batch[0, :, :, :] = self.getState().copy()
    return self.mini_batch

  def reset(self):
    self.current = 0