from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Gaussian
from neon.optimizers import RMSProp
from neon.layers import Affine, Conv, Merge, GeneralizedCost
from neon.transforms import Rectlin
from neon.models import Model
from neon.transforms import SumSquared
import numpy as np

class DeepQNetwork:
  def __init__(self, num_actions, learning_rate = 0.002, discount_rate = 0.9, history_length = 4, screen_dim = (84,84), batch_size = 32,
      backend = 'gpu', random_seed = None, device_id = 0, datatype = 'float32', stochastic_round = False):
    # create Neon backend
    self.be = gen_backend(backend = backend,
                 batch_size = batch_size,
                 rng_seed = random_seed,
                 device_id = device_id,
                 default_dtype = np.dtype(datatype).type,
                 stochastic_round = stochastic_round)

    # create network
    init_norm = Gaussian(loc=0.0, scale=0.01)
    self.layers = []
    self.layers.append(Conv((8, 8, 32), strides=4, init=init_norm, activation=Rectlin()))
    self.layers.append(Conv((4, 4, 64), strides=2, init=init_norm, activation=Rectlin()))
    self.layers.append(Conv((3, 3, 64), strides=1, init=init_norm, activation=Rectlin()))
    self.layers.append(Affine(nout=512, init=init_norm, activation=Rectlin()))
    self.layers.append(Affine(nout = num_actions, init = init_norm))
    self.cost = GeneralizedCost(costfunc = SumSquared())
    self.model = Model(layers = self.layers)
    self.optimizer = RMSProp(learning_rate = learning_rate, stochastic_round = stochastic_round)
    self.prepare_layers(self.layers)

    # remember parameters
    self.num_actions = num_actions
    self.batch_size = batch_size
    self.discount_rate = discount_rate

    # prepare tensors once and reuse them
    self.input_shape = (history_length,) + screen_dim + (batch_size,)
    self.tensor = self.be.empty(self.input_shape)
    self.tensor.lshape = self.input_shape
    self.targets = self.be.empty((num_actions, batch_size))

  def train(self, minibatch, epoch):
    # expand components of minibatch
    prestates, actions, rewards, poststates, terminals = minibatch
    assert len(prestates.shape) == 4
    assert len(poststates.shape) == 4
    assert len(actions.shape) == 1
    assert len(rewards.shape) == 1
    assert len(terminals.shape) == 1
    assert prestates.shape == poststates.shape
    assert prestates.shape[3] == actions.shape[0] == rewards.shape[0] == poststates.shape[3] == terminals.shape[0]
    # feed-forward pass for poststates to get Q-values
    self.tensor.set(poststates)
    postq = self.model.fprop(self.tensor, inference = True)
    assert postq.shape == (self.num_actions, self.batch_size)
    # calculate max Q-value for each poststate
    maxpostq = self.be.max(postq, axis=0).asnumpyarray()
    assert maxpostq.shape == (1, self.batch_size)
    # feed-forward pass for prestates
    self.tensor.set(prestates)
    preq = self.model.fprop(self.tensor, inference = False)
    assert preq.shape == (self.num_actions, self.batch_size)
    # make copy of Q-values as targets
    self.targets.copy(preq)
    # update Q-value targets for actions taken
    for i, action in enumerate(actions):
      if terminals[i]:
        self.targets[action, i] = rewards[i]
      else:
        self.targets[action, i] = rewards[i] + self.discount_rate * maxpostq[0,i]
    # calculate errors
    deltas = self.cost.get_errors(preq, self.targets)
    # perform back-propagation of gradients
    self.model.bprop(deltas)
    # perform optimization
    self.optimizer.optimize(self.layers_to_optimize, epoch)

  def predict(self, state):
    self.tensor.set(np.resize(state, self.input_shape))
    qvalues = self.model.fprop(self.tensor, inference = True)
    return qvalues.asnumpyarray()[:,0]

  def prepare_layers(self, layers):
    self.layers = []
    self.layers_to_optimize = []

    for layer in layers:
        if isinstance(layer, list):
            self.layers.extend(layer)
        else:
            self.layers.append(layer)

    for layer in self.layers:
        if layer.has_params:
            self.layers_to_optimize.append(layer)
        elif isinstance(layer, Merge):
            self.layers_to_optimize += layer.layers_to_optimize
