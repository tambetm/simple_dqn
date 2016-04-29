import sys
import os
import logging
import cv2
logger = logging.getLogger(__name__)

class Environment:
  def __init__(self):
    pass

  def numActions(self):
    # Returns number of actions
    raise NotImplementedError

  def restart(self):
    # Restarts environment
    raise NotImplementedError

  def act(self, action):
    # Performs action and returns reward
    raise NotImplementedError

  def getScreen(self):
    # Gets current game screen
    raise NotImplementedError

  def isTerminal(self):
    # Returns if game is done
    raise NotImplementedError

class ALEEnvironment(Environment):
  def __init__(self, rom_file, args):
    from ale_python_interface import ALEInterface
    self.ale = ALEInterface()
    if args.display_screen:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        self.ale.setBool('sound', False) # Sound doesn't work on OSX
      elif sys.platform.startswith('linux'):
        self.ale.setBool('sound', True)
      self.ale.setBool('display_screen', True)

    self.ale.setInt('frame_skip', args.frame_skip)
    self.ale.setFloat('repeat_action_probability', args.repeat_action_probability)
    self.ale.setBool('color_averaging', args.color_averaging)

    if args.random_seed:
      self.ale.setInt('random_seed', args.random_seed)

    if args.record_screen_path:
      if not os.path.exists(args.record_screen_path):
        logger.info("Creating folder %s" % args.record_screen_path)
        os.makedirs(args.record_screen_path)
      logger.info("Recording screens to %s", args.record_screen_path)
      self.ale.setString('record_screen_dir', args.record_screen_path)

    if args.record_sound_filename:
      logger.info("Recording sound to %s", args.record_sound_filename)
      self.ale.setBool('sound', True)
      self.ale.setString('record_sound_filename', args.record_sound_filename)

    self.ale.loadROM(rom_file)

    if args.minimal_action_set:
      self.actions = self.ale.getMinimalActionSet()
      logger.info("Using minimal action set with size %d" % len(self.actions))
    else:
      self.actions = self.ale.getLegalActionSet()
      logger.info("Using full action set with size %d" % len(self.actions))
    logger.debug("Actions: " + str(self.actions))

    # OpenCV expects width as first and height as second
    self.dims = (args.screen_width, args.screen_height)

  def numActions(self):
    return len(self.actions)

  def restart(self):
    self.ale.reset_game()

  def act(self, action):
    return self.ale.act(self.actions[action])

  def getScreen(self):
    return cv2.resize(self.ale.getScreenGrayscale(), self.dims)

  def isTerminal(self):
    return self.ale.game_over()

class GymEnvironment(Environment):
  # For use with Open AI Gym Environment
  def __init__(self, env_id, args):
    import gym
    self.gym = gym.make(env_id)
    self.obs = None
    self.terminal = None
    # OpenCV expects width as first and height as second s
    self.dims = (args.screen_width, args.screen_height)

  def numActions(self):
    import gym
    assert isinstance(self.gym.action_space, gym.spaces.Discrete)
    return self.gym.action_space.n

  def restart(self):
    self.obs = self.gym.reset()
    self.terminal = False

  def act(self, action):
    self.obs, reward, self.terminal, _ = self.gym.step(action)
    return reward

  def getScreen(self):
    assert self.obs is not None
    return cv2.resize(cv2.cvtColor(self.obs, cv2.COLOR_RGB2GRAY), self.dims)

  def isTerminal(self):
    assert self.terminal is not None
    return self.terminal