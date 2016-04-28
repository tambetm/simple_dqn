import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

with open(args.file, 'r') as fid:
    model_param = pickle.load(fid)
    
for lay_param in model_param['layer_params_states']:
    w = lay_param['params']
    lay_param['params'] = {'W': w}
            
with open(args.file, 'w') as fid:
    pickle.dump(model_param, fid)
