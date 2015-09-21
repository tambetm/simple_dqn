from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Gaussian
from neon.optimizers import RMSProp
from neon.layers import Affine, Conv, Merge, GeneralizedCost
from neon.transforms import Rectlin
from neon.models import Model
from neon.transforms import SumSquared
from neon.util.persist import save_obj
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

class DeepQNetwork:
  def __init__(self, num_actions, args):
    # create Neon backend
    self.be = gen_backend(backend = args.backend,
                 batch_size = args.batch_size,
                 rng_seed = args.random_seed,
                 device_id = args.device_id,
                 default_dtype = np.dtype(args.datatype).type,
                 stochastic_round = args.stochastic_round)

    layers = self.createLayers(num_actions)
    self.cost = GeneralizedCost(costfunc = SumSquared())
    self.model = Model(layers = layers)
    self.optimizer = RMSProp(learning_rate = args.learning_rate, 
        decay_rate = args.rmsprop_decay_rate, 
        stochastic_round = args.stochastic_round)
    self.prepare_layers(layers)

    # create target model
    self.target_steps = args.target_steps
    if self.target_steps:
      self.target_model = Model(layers = self.createLayers(num_actions))
      self.save_weights_path = args.save_weights_path
    else:
      self.target_model = self.model
    self.train_iterations = 0

    # remember parameters
    self.num_actions = num_actions
    self.batch_size = args.batch_size
    self.discount_rate = args.discount_rate
    self.history_length = args.history_length
    self.screen_dim = (args.screen_height, args.screen_width)
    self.clip_error = args.clip_error

    # prepare tensors once and reuse them
    self.input_shape = (self.history_length,) + self.screen_dim + (self.batch_size,)
    self.tensor = self.be.empty(self.input_shape)
    self.tensor.lshape = self.input_shape # needed for convolutional networks
    self.targets = self.be.empty((self.num_actions, self.batch_size))

  def createLayers(self, num_actions):
    # create network
    init_norm = Gaussian(loc=0.0, scale=0.01)
    layers = []
    # The first hidden layer convolves 32 filters of 8x8 with stride 4 with the input image and applies a rectifier nonlinearity.
    layers.append(Conv((8, 8, 32), strides=4, init=init_norm, activation=Rectlin()))
    # The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
    layers.append(Conv((4, 4, 64), strides=2, init=init_norm, activation=Rectlin()))
    # This is followed by a third convolutional layer that convolves 64 filters of 3x3 with stride 1 followed by a rectifier.
    layers.append(Conv((3, 3, 64), strides=1, init=init_norm, activation=Rectlin()))
    # The final hidden layer is fully-connected and consists of 512 rectifier units.
    layers.append(Affine(nout=512, init=init_norm, activation=Rectlin()))
    # The output layer is a fully-connected linear layer with a single output for each valid action.
    layers.append(Affine(nout = num_actions, init = init_norm))
    return layers

  def train(self, minibatch, epoch):
    # expand components of minibatch
    prestates, actions, rewards, poststates, terminals = minibatch
    assert len(prestates.shape) == 4
    assert len(poststates.shape) == 4
    assert len(actions.shape) == 1
    assert len(rewards.shape) == 1
    assert len(terminals.shape) == 1
    assert prestates.shape == poststates.shape
    assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

    if self.target_steps and self.train_iterations % self.target_steps == 0:
      # push something through network, so that weights exist
      self.model.fprop(self.tensor)
      filename = os.path.join(self.save_weights_path, "target_network.pkl")
      save_obj(self.model.serialize(keep_states = False), filename)
      self.target_model.load_weights(filename)

    # feed-forward pass for poststates to get Q-values
    # change order of axes to match what Neon expects
    # copy() shouldn't be necessary here, but Neon doesn't work on views
    self.tensor.set(np.transpose(poststates, axes = (1, 2, 3, 0)).copy())
    self.be.divide(self.tensor, 255, self.tensor)
    postq = self.target_model.fprop(self.tensor, inference = True)
    assert postq.shape == (self.num_actions, self.batch_size)

    # calculate max Q-value for each poststate
    maxpostq = self.be.max(postq, axis=0).asnumpyarray()
    assert maxpostq.shape == (1, self.batch_size)

    # feed-forward pass for prestates
    # change order of axes to match what Neon expects
    # copy() shouldn't be necessary here, but Neon doesn't work on views
    self.tensor.set(np.transpose(prestates, axes = (1, 2, 3, 0)).copy())
    self.be.divide(self.tensor, 255, self.tensor)
    preq = self.model.fprop(self.tensor, inference = False)
    assert preq.shape == (self.num_actions, self.batch_size)

    # make copy of Q-values as targets
    self.targets.copy(preq)

    # update Q-value targets for actions taken
    for i, action in enumerate(actions):
      if terminals[i]:
        self.targets[action, i] = float(rewards[i])
      else:
        self.targets[action, i] = float(rewards[i]) + self.discount_rate * maxpostq[0,i]

    # calculate errors
    deltas = self.cost.get_errors(preq, self.targets)
    assert deltas.shape == (self.num_actions, self.batch_size)
    #assert np.count_nonzero(deltas.asnumpyarray()) == 32

    #cost_before = self.cost.get_cost(preq, self.targets).asnumpyarray()[0,0]
    #qvalues_before = preq.asnumpyarray()[:,0]
    #targets = self.targets.asnumpyarray()[:,0]

    # clip errors
    if self.clip_error:
      self.be.minimum(deltas, self.clip_error, out = deltas)
      self.be.maximum(deltas, -self.clip_error, out = deltas)

    # perform back-propagation of gradients
    self.model.bprop(deltas)

    # perform optimization
    self.optimizer.optimize(self.layers_to_optimize, epoch)

    #preq = self.model.fprop(self.tensor, inference = False)
    #cost_after = self.cost.get_cost(preq, self.targets).asnumpyarray()[0,0]
    #qvalues_after = preq.asnumpyarray()[:,0]
    #if rewards[0] > 0:
    #  print "cost_before: %g, cost_after: %g" % (cost_before, cost_after) 
    #  print "qvalues_before: ", qvalues_before, ", action: ", actions[0], ", reward: ", rewards[0]
    #  print "targets: ", targets
    #  print "qvalues_after: ", qvalues_after
    #  raw_input("Press ENTER")

    self.train_iterations += 1

  def predict(self, state):
    assert state.shape == ((self.batch_size, self.history_length,) + self.screen_dim)
    # assign to tensor
    self.tensor.set(np.transpose(state, axes = (1, 2, 3, 0)).copy())
    self.be.divide(self.tensor, 255, self.tensor)
    # calculate Q-values for the state
    qvalues = self.model.fprop(self.tensor, inference = True)
    assert qvalues.shape == (self.num_actions, self.batch_size)
    logger.debug("Q-values: " + str(qvalues.asnumpyarray()[:,0]))
    # find the action with highest q-value
    actions = self.be.argmax(qvalues, axis = 0)
    # take only first result
    return actions.asnumpyarray()[0,0]

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

  def load_weights(self, load_path):
    self.model.load_weights(load_path)

  def save_weights(self, save_path):
    save_obj(self.model.serialize(keep_states = True), save_path)
