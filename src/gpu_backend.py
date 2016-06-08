from neon.backends import gen_backend
import numpy as np

global backend
backend = None

def initialize_backend(args):
    global backend
    if backend is None:
        backend = gen_backend(backend=args.backend,
                              batch_size=args.batch_size,
                              rng_seed=args.random_seed,
                              device_id=args.device_id,
                              datatype=np.dtype(args.datatype).type,
                              stochastic_round=args.stochastic_round)
    return backend