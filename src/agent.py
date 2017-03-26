import random
import logging
import numpy as np
logger = logging.getLogger(__name__)
from state_buffer import StateBuffer

class Agent:
  def __init__(self, environment, replay_memory, deep_q_network, args):
    self.env = environment
    self.mem = replay_memory
    self.net = deep_q_network
    self.buf = StateBuffer(args)
    self.num_actions = self.env.numActions()
    self.random_starts = args.random_starts
    self.history_length = args.history_length

    self.exploration_rate_start = args.exploration_rate_start
    self.exploration_rate_end = args.exploration_rate_end
    self.exploration_decay_steps = args.exploration_decay_steps
    self.exploration_rate_test = args.exploration_rate_test
    self.total_train_steps = args.start_epoch * args.train_steps

    self.train_frequency = args.train_frequency
    self.train_repeat = args.train_repeat
    self.target_steps = args.target_steps

    self.callback = None

  def _restartRandom(self):
    self.env.restart()
    # perform random number of dummy actions to produce more stochastic games
    for i in xrange(random.randint(self.history_length, self.random_starts) + 1):
      reward = self.env.act(0)
      terminal = self.env.isTerminal()
      if terminal:
          self.env.restart()
      screen = self.env.getScreen()
      # add dummy states to buffer
      self.buf.add(screen)

  def _explorationRate(self):
    # calculate decaying exploration rate
    if self.total_train_steps < self.exploration_decay_steps:
      return self.exploration_rate_start - self.total_train_steps * (self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
    else:
      return self.exploration_rate_end

  def step(self, exploration_rate):
    # exploration rate determines the probability of random moves
    if random.random() < exploration_rate:
      action = random.randrange(self.num_actions)
      logger.debug("Random action = %d" % action)
    else:
      # otherwise choose action with highest Q-value
      state = self.buf.getStateMinibatch()
      # for convenience getStateMinibatch() returns minibatch
      # where first item is the current state
      qvalues = self.net.predict(state)
      assert len(qvalues[0]) == self.num_actions
      # choose highest Q-value of first state
      action = np.argmax(qvalues[0])
      logger.debug("Predicted action = %d" % action)

    # perform the action
    reward = self.env.act(action)
    screen = self.env.getScreen()
    terminal = self.env.isTerminal()

    # print reward
    if reward <> 0:
      logger.debug("Reward: %d" % reward)

    # add screen to buffer
    self.buf.add(screen)

    # restart the game if over
    if terminal:
      logger.debug("Terminal state, restarting")
      self._restartRandom()

    # call callback to record statistics
    if self.callback:
      self.callback.on_step(action, reward, terminal, screen, exploration_rate)

    return action, reward, screen, terminal

  def play_random(self, random_steps):
    # play given number of steps
    for i in xrange(random_steps):
      # use exploration rate 1 = completely random
      action, reward, screen, terminal = self.step(1)
      self.mem.add(action, reward, screen, terminal)

  def train(self, train_steps, epoch = 0):
    # do not do restart here, continue from testing
    #self._restartRandom()
    # play given number of steps
    for i in xrange(train_steps):
      # perform game step
      action, reward, screen, terminal = self.step(self._explorationRate())
      self.mem.add(action, reward, screen, terminal)
      # Update target network every target_steps steps
      if self.target_steps and i % self.target_steps == 0:
        self.net.update_target_network()
      # train after every train_frequency steps
      if self.mem.count > self.mem.batch_size and i % self.train_frequency == 0:
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
      terminal = False
      while not terminal:
        action, reward, screen, terminal = self.step(self.exploration_rate_test)
        # add experiences to replay memory for visualization
        self.mem.add(action, reward, screen, terminal)
