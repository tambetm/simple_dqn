import argparse
import gym

parser = argparse.ArgumentParser()
parser.add_argument("results_folder")
parser.add_argument("--api_key")
args = parser.parse_args()

gym.upload(args.results_folder, api_key=args.api_key)
