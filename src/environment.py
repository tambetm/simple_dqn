import sys
from ale_python_interface import ALEInterface
import cv2

class Environment:
  def __init__(self, rom_file, dims=(84,84), minimal_action_set=False, display_screen=False, frame_skip=4, random_seed=None):
    self.ale = ALEInterface()
    if display_screen:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        self.ale.setBool('sound', False) # Sound doesn't work on OSX
      elif sys.platform.startswith('linux'):
        self.ale.setBool('sound', True)
      self.ale.setBool('display_screen', True)

    self.ale.setInt('frame_skip', frame_skip)

    if random_seed:
      self.ale.setInt('random_seed', random_seed)

    self.ale.loadROM(rom_file)
    if minimal_action_set:
      self.actions = ale.getMinimalActionSet()
    else:
      self.actions = ale.getLegalActionSet()

    self.dims = dims

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
