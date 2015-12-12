import sys
import csv
import time
import logging
import numpy as np
logger = logging.getLogger(__name__)

class Statistics:
  def __init__(self, agent, net, mem, env, args):
    self.agent = agent
    self.net = net
    self.mem = mem
    self.env = env

    self.agent.callback = self
    self.net.callback = self

    self.csv_name = args.csv_file
    if self.csv_name:
      logger.info("Results are written to %s" % args.csv_file)
      self.csv_file = open(self.csv_name, "wb")
      self.csv_writer = csv.writer(self.csv_file)
      self.csv_writer.writerow((
          "epoch",
          "phase",
          "steps",
          "nr_games",
          "average_reward",
          "min_game_reward",
          "max_game_reward", 
          "last_exploration_rate",
          "total_train_steps",
          "replay_memory_count",
          "meanq",
          "meancost",
          "weight_updates",
          "total_time",
          "epoch_time",
          "steps_per_second"
        ))
      self.csv_file.flush()

    self.start_time = time.clock()
    self.validation_states = None

  def reset(self):
    self.epoch_start_time = time.clock()
    self.num_steps = 0
    self.num_games = 0
    self.game_rewards = 0
    self.average_reward = 0
    self.min_game_reward = sys.maxint
    self.max_game_reward = -sys.maxint - 1
    self.last_exploration_rate = 1
    self.average_cost = 0

  # callback for agent
  def on_step(self, action, reward, terminal, screen, exploration_rate):
    self.game_rewards += reward
    self.num_steps += 1
    self.last_exploration_rate = exploration_rate

    if terminal:
      self.num_games += 1
      self.average_reward += float(self.game_rewards - self.average_reward) / self.num_games
      self.min_game_reward = min(self.min_game_reward, self.game_rewards)
      self.max_game_reward = max(self.max_game_reward, self.game_rewards)
      self.game_rewards = 0

  def on_train(self, cost):
    self.average_cost += (cost - self.average_cost) / self.net.train_iterations

  def write(self, epoch, phase):
    current_time = time.clock()
    total_time = current_time - self.start_time
    epoch_time = current_time - self.epoch_start_time
    steps_per_second = self.num_steps / epoch_time

    if self.num_games == 0:
      self.num_games = 1
      self.average_reward = self.game_rewards

    if self.validation_states is None and self.mem.count > self.mem.batch_size:
      # sample states for measuring Q-value dynamics
      prestates, actions, rewards, poststates, terminals = self.mem.getMinibatch()
      self.validation_states = prestates

    if self.csv_name:
      if self.validation_states is not None:
        qvalues = self.net.predict(self.validation_states)
        maxqs = np.max(qvalues, axis=1)
        assert maxqs.shape[0] == qvalues.shape[0]
        meanq = np.mean(maxqs)
      else:
        meanq = 0

      self.csv_writer.writerow((
          epoch,
          phase,
          self.num_steps,
          self.num_games,
          self.average_reward,
          self.min_game_reward,
          self.max_game_reward,
          self.last_exploration_rate,
          self.agent.total_train_steps,
          self.mem.count,
          meanq,
          self.average_cost,
          self.net.train_iterations,
          total_time,
          epoch_time,
          steps_per_second
        ))
      self.csv_file.flush()
    
    logger.info("  num_games: %d, average_reward: %f, min_game_reward: %d, max_game_reward: %d" % 
        (self.num_games, self.average_reward, self.min_game_reward, self.max_game_reward))
    logger.info("  last_exploration_rate: %f, epoch_time: %ds, steps_per_second: %d" %
        (self.last_exploration_rate, epoch_time, steps_per_second))

  def close(self):
    if self.csv_name:
      self.csv_file.close()
