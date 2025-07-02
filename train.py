from env.env import Env
from plots.plot import Plot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dots",type=int,default=4,help="no. of dots in game")
parser.add_argument("--agent", type=str, required=True, help="alphazero,mcts,dqn,qt")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")

args = parser.parse_args()
if args.agent1 == 'alphazero':
  from agents.alphazero import alphazero
  trainer = alphazero()
if args.agent1 == 'mcts':
  from agents.mcts import mcts
  trainer = mcts()
if args.agent1 == 'dqn' :
  from agents.dqn import dqn
  trainer = dqn()
if args.agent1 == 'qt':
  from agents.tabular_q import qt
  trainer = qt()

trainer.train(args.dots,args.epochs)