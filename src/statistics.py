import csv
import time
import logging
logger = logging.getLogger(__name__)

class Statistics:
  def __init__(self, agent, net, mem, env, args):
    self.agent = agent
    self.net = net
    self.mem = mem
    self.env = env

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
    self.agent.resetStats()

  def write(self, epoch, phase):
    current_time = time.clock()
    total_time = current_time - self.start_time
    epoch_time = current_time - self.epoch_start_time
    steps_per_second = self.agent.num_steps / epoch_time

    if self.agent.num_games == 0:
      num_games = 1
      average_reward = self.agent.game_rewards
    else:
      num_games = self.agent.num_games
      average_reward = self.agent.average_reward

    if self.validation_states is None:
      # sample states for measuring Q-value dynamics
      prestates, actions, rewards, poststates, terminals = self.mem.getMinibatch()
      self.validation_states = prestates

    if self.csv_name:
      meanq = self.net.getMeanQ(self.validation_states)
      meancost = self.net.average_cost

      self.csv_writer.writerow((
          epoch,
          phase,
          self.agent.num_steps,
          num_games,
          average_reward,
          self.agent.min_game_reward,
          self.agent.max_game_reward,
          self.agent.last_exploration_rate,
          self.agent.total_train_steps,
          self.mem.count,
          meanq,
          meancost,
          self.net.train_iterations,
          total_time,
          epoch_time,
          steps_per_second
        ))
      self.csv_file.flush()
    
    logger.info("  num_games: %d, average_reward: %f, min_game_reward: %d, max_game_reward: %d" % 
        (num_games, average_reward, self.agent.min_game_reward, self.agent.max_game_reward))
    logger.info("  last_exploration_rate: %f, epoch_time: %ds, steps_per_second: %d" %
        (self.agent.last_exploration_rate, epoch_time, steps_per_second))

  def close(self):
    if self.csv_name:
      self.csv_file.close()
