import numpy as np

class StateBuffer:
  """
  While ReplayMemory could have been used for fetching the current state,
  this also means that test time states make their way to training process.
  Having separate StateBuffer ensures that test data doesn't leak into training.
  """
  def __init__(self, args):
    self.history_length = args.history_length
    self.dims = (args.screen_height, args.screen_width)
    self.batch_size = args.batch_size
    self.buffer = np.zeros((self.batch_size, self.history_length) + self.dims, dtype=np.uint8)

  def add(self, observation):
    assert observation.shape == self.dims
    self.buffer[0, :-1] = self.buffer[0, 1:]
    self.buffer[0, -1] = observation

  def getState(self):
    return self.buffer[0]

  def getStateMinibatch(self):
    return self.buffer

  def reset(self):
    self.buffer *= 0

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--screen_width", type=int, default=40, help="Screen width after resize.")
  parser.add_argument("--screen_height", type=int, default=52, help="Screen height after resize.")
  parser.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
  parser.add_argument("--loops", type=int, default=1000000, help="Number of loops in testing.")
  args = parser.parse_args()

  import numpy as np
  mem = StateBuffer(args)
  for i in xrange(args.loops):
    mem.add(np.zeros((args.screen_height, args.screen_width)))
    if i >= args.history_length:
      state = mem.getState()
      batch = mem.getStateMinibatch()