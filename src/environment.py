import sys
import os
import logging
logger = logging.getLogger(__name__)

class Environment:
  def __init__(self, env_name, args): raise NotImplementedError
  def numActions(self): raise NotImplementedError
  def restart(self): raise NotImplementedError
  def act(self, action): raise NotImplementedError
  def getScreen(self): raise NotImplementedError
  def isTerminal(self): raise NotImplementedError

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
    reward = self.ale.act(self.actions[action])
    return reward

  def getScreen(self):
    screen = self.ale.getScreenGrayscale()
    import cv2
    resized = cv2.resize(screen, self.dims)
    return resized

  def isTerminal(self):
    return self.ale.game_over()

class GymEnvironment(Environment):
  # For training with Open AI Gym Environment
  def __init__(self, env_id, args):
    import gym
    self.gym = gym.make(env_id)
    self.obs = None
    self.terminal = None

  def numActions(self):
    import gym
    assert isinstance(self.gym.action_space, gym.spaces.Discrete)
    return self.gym.action_space.n

  def restart(self):
    self.gym.reset()
    self.obs = None
    self.terminal = None

  def act(self, action):
    self.obs, reward, self.terminal, _ = self.gym.step(action)
    return reward

  def getScreen(self):
    assert self.obs is not None
    return self.obs

  def isTerminal(self):
    assert self.terminal is not None
    return self.terminal