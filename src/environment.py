import sys
import os
from ale_python_interface import ALEInterface
import cv2
import logging
logger = logging.getLogger(__name__)

class Environment:
  def __init__(self, args):
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

    self.ale.loadROM(args.rom_file)

    if args.minimal_action_set:
      self.actions = self.ale.getMinimalActionSet()
      logger.info("Using minimal action set with size %d" % len(self.actions))
    else:
      self.actions = self.ale.getLegalActionSet()
      logger.info("Using full action set with size %d" % len(self.actions))
    logger.debug("Actions: " + str(self.actions))

    self.dims = (args.screen_height, args.screen_width)

  def numActions(self):
    return len(self.actions)

  def restart(self):
    self.ale.reset_game()
    self.lives = self.ale.lives()

  def act(self, action):
    reward = self.ale.act(self.actions[action])
    return reward

  def getScreen(self):
    screen = self.ale.getScreenGrayscale()
    resized = cv2.resize(screen, self.dims)
    return resized

  def isTerminal(self, training = False):
    # during training loss of life is considered terminal state
    lives = self.ale.lives()
    if training and lives < self.lives:
      terminal = True
    else:
      terminal = self.ale.game_over()
    self.lives = lives

    return terminal
