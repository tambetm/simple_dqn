from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Gaussian
from neon.optimizers import RMSProp
from neon.layers import Affine, Conv, GeneralizedCost
from neon.models import Model
from neon.transforms import SumSquared

class DeepQNetwork:
  def __init__(num_actions, learning_rate = 0.002, discount_rate = 0.9, batch_size = 32):
    # parse Neon command line arguments
    parser = NeonArgparser()
    args = parser.parse_args()
    # create Neon backend
    self.be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype,
                 stochastic_round=False)

    # create network
    init_norm = Gaussian(loc=0.0, scale=0.01)
    self.layers = []
    self.layers.append(Conv((8, 8, 32), strides=4, init=init_norm, activation=Rectlin()))
    self.layers.append(Conv((4, 4, 64), strides=2, init=init_norm, activation=Rectlin()))
    self.layers.append(Conv((3, 3, 64), strides=1, init=init_norm, activation=Rectlin()))
    self.layers.append(Affine(nout=512, init=init_norm, activation=Rectlin()))
    self.layers.append(Affine(nout=num_actions, init=init_norm))
    self.cost = GeneralizedCost(costfunc=SumSquared())
    self.model = Model(layers=layers)
    self.optimizer = RMSProp(learning_rate=learning_rate, stochastic_round=args.rounding)

    self.batch_size = batch_size
    self.discount_rate = discount_rate

  def train(self, minibatch):
    # expand components of minibatch
    prestates, actions, rewards, poststates, terminals = minibatch
    assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]
    assert prestates.shape == poststates.shape
    assert len(prestates.shape) == 4
    assert len(poststates.shape) == 4
    assert len(actions.shape) == 1
    assert len(rewards.shape) == 1
    assert len(terminals.shape) == 1
    # convert prestates and poststates to tensors
    prestates = self.be.array(prestates)
    poststates = self.be.array(poststates)
    # feed-forward pass for poststates to get Q-values
    postq = self.model.fprop(poststates, inference = True)
    # calculate max Q-value for each poststate
    maxpostq = self.be.max(postq, axis=1)
    # feed-forward pass for prestates
    preq = self.model.fprop(prestates, inference = False)
    # make copy of Q-values as targets
    targets = preq.copy()
    # update Q-value targets for actions taken
    for i, action in enumerate(actions):
      if terminals[i]:
        targets[i, action] = reward[i]
      else:
        targets[i, action] = reward[i] + self.discount_rate * maxpostq[i])
    # calculate errors
    deltas = self.cost.get_errors(preq, targets)
    # perform back-propagation of gradients
    self.model.bprop(deltas)
    # perform optimization
    self.optimizer.optimize(self.layers, 0)

  def predict(self, state):
    shape = state.shape
    state = self.be.array(state)
    state.reshape((1,) + shape)
    qvalues = self.model.fprop(state, inference = True)
    return qvalues.asnumpyarray()[0]
