import numpy as np
import time
import logging
logger = logging.getLogger(__name__)

class Statistics:
  def __init__(self, environment, replay_memory, deep_q_network, agent):
    self.env = environment
    self.mem = replay_memory
    self.net = deep_q_network
    self.agent = agent
    self.agent.callback = self
    self.reset()

  def reset(self):
    self.games = 0
    self.game_rewards = 0
    self.rewards = []
    self.time = time.time()
    self.num_steps = 0

  def on_step(self, action, reward, screen, terminal):
    self.game_rewards += reward
    self.num_steps += 1

    if terminal:
      self.games += 1
      self.rewards.append(self.game_rewards)
      self.game_rewards = 0

  def log(self):
    elapsed_time = time.time() - self.time
    logger.info("  games: %d, average_reward: %f, replay_memory_count: %d, total_train_steps: %d, exploration_rate: %f, time_elapsed: %d, steps_per_second: %d" % 
        (self.games, np.mean(self.rewards), self.mem.count, self.agent.total_train_steps, self.agent.exploration_rate(), elapsed_time, self.num_steps / elapsed_time))
