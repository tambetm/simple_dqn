import sys
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
    self.ale.setBool('color_averaging', args.color_averaging)

    if args.random_seed:
      self.ale.setInt('random_seed', args.random_seed)

    self.ale.loadROM(args.rom_file)
    if args.minimal_action_set:
      self.actions = self.ale.getMinimalActionSet()
      logger.info("Using minimal action set with size %d" % len(self.actions))
    else:
      self.actions = self.ale.getLegalActionSet()
      logger.info("Using full action set with size %d" % len(self.actions))

    self.dims = (args.screen_height, args.screen_width)

  def numActions(self):
    return len(self.actions)

  def restart(self):
    self.ale.reset_game()

  def getScreen(self):
    screen = self.ale.getScreenGrayscale()
    resized = cv2.resize(screen, self.dims)
    return resized

  def act(self, action):
    reward = self.ale.act(action)
    return reward, self.getScreen(), self.ale.game_over()
