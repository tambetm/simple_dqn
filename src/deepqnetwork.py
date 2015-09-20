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

class DeepQNetwork:
  def __init__(self, num_actions, args):
    # create Neon backend
    self.be = gen_backend(backend = args.backend,
                 batch_size = args.batch_size,
                 rng_seed = args.random_seed,
                 device_id = args.device_id,
                 default_dtype = np.dtype(args.datatype).type,
                 stochastic_round = args.stochastic_round)

    # create network
    init_norm = Gaussian(loc=0.0, scale=0.01)
    self.layers = []
    # The first hidden layer convolves 32 filters of 8x8 with stride 4 with the input image and applies a rectifier nonlinearity.
    self.layers.append(Conv((8, 8, 32), strides=4, init=init_norm, activation=Rectlin()))
    # The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
    self.layers.append(Conv((4, 4, 64), strides=2, init=init_norm, activation=Rectlin()))
    # This is followed by a third convolutional layer that convolves 64 filters of 3x3 with stride 1 followed by a rectifier.
    self.layers.append(Conv((3, 3, 64), strides=1, init=init_norm, activation=Rectlin()))
    # The final hidden layer is fully-connected and consists of 512 rectifier units.
    self.layers.append(Affine(nout=512, init=init_norm, activation=Rectlin()))
    # The output layer is a fully-connected linear layer with a single output for each valid action.
    self.layers.append(Affine(nout = num_actions, init = init_norm))
    self.cost = GeneralizedCost(costfunc = SumSquared())
    self.model = Model(layers = self.layers)
    self.optimizer = RMSProp(learning_rate = args.learning_rate, stochastic_round = args.stochastic_round)
    self.prepare_layers(self.layers)

    # remember parameters
    self.num_actions = num_actions
    self.batch_size = args.batch_size
    self.discount_rate = args.discount_rate
    self.history_length = args.history_length
    self.screen_dim = (args.screen_height, args.screen_width)

    # prepare tensors once and reuse them
    self.input_shape = (self.history_length,) + self.screen_dim + (self.batch_size,)
    self.tensor = self.be.empty(self.input_shape)
    self.tensor.lshape = self.input_shape # needed for convolutional networks
    self.targets = self.be.empty((self.num_actions, self.batch_size))

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

    # feed-forward pass for poststates to get Q-values
    # change order of axes to match what Neon expects
    # copy() shouldn't be necessary here, but Neon doesn't work on views
    self.tensor.set(np.transpose(poststates, axes = (1, 2, 3, 0)).copy())
    postq = self.model.fprop(self.tensor, inference = True)
    assert postq.shape == (self.num_actions, self.batch_size)

    # calculate max Q-value for each poststate
    maxpostq = self.be.max(postq, axis=0).asnumpyarray()
    assert maxpostq.shape == (1, self.batch_size)

    # feed-forward pass for prestates
    # change order of axes to match what Neon expects
    # copy() shouldn't be necessary here, but Neon doesn't work on views
    self.tensor.set(np.transpose(prestates, axes = (1, 2, 3, 0)).copy())
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

    # perform back-propagation of gradients
    self.model.bprop(deltas)

    # perform optimization
    self.optimizer.optimize(self.layers_to_optimize, epoch)

  def predict(self, state):
    assert state.shape == ((self.history_length,) + self.screen_dim)
    # resize (duplicate) the input to match batch size
    self.tensor.set(np.resize(state, self.input_shape))
    # calculate Q-values for the state
    qvalues = self.model.fprop(self.tensor, inference = True)
    assert qvalues.shape == (self.num_actions, self.batch_size)
    # take only first result
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

  def load_weights(self, load_path):
    self.model.load_weights(load_path)

  def save_weights(self, save_path):
    save_obj(self.model.serialize(keep_states=True), save_path)
