from env.env import Env
from plots.plot import Plot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dots",type=int,default=4,help="no. of dots in game")
parser.add_argument("--agent1", type=str, required=True, help="alphazero,minmax,mcts,dqn,qt,self")
parser.add_argument("--agent2", type=str, required=True, help="alphazero,minmax,mcts,dqn,qt,self")
parser.add_argument("--games", type=int, default=1, help="Number of games to play")
parser.add_argument("--secs",type=int,default=1,help="How many seconds agent should think before move(for mcts and alphazero)")

args = parser.parse_args()
if args.agent1 == 'alphazero':
  from agents.alphazero import Play as play1
if args.agent1 == 'mcts':
  from agents.mcts import Play as play1
if args.agent1 == 'dqn':
  from agents.dqn import Play as play1
if args.agent1 == 'qt':
  from agents.tabular_q import Play as play1
if args.agent1 == 'minmax':
   from agents.minmax import Play as play1
if not args.agent1=="self":
  player1 = play1()

if args.agent2 == 'alphazero':
  from agents.alphazero import Play
if args.agent2 == 'mcts':
  from agents.mcts import Play
if args.agent2 == 'dqn':
  from agents.dqn import Play
if args.agent2 == 'qt':
  from agents.tabular_q import Play
if args.agent2 == 'minmax':
   from agents.minmax import Play
if not args.agent2=="self":
  player2 = Play()
plotter = Plot()
# You now have:
# args.agent1  → e.g., "alphazero", "minmax", or path to a saved model
# args.agent2  → same as above
# args.games   → integer
device = 'cpu'
env = Env(args.dots)
wins = 0
draws = 0
loss = 0
#player 1 first
for game in range(args.games):
    env.reset()
    turn = 1
    value = 0
    while(not env.gameover()):
        if turn == 1:
            if not args.agent1 == "self":
                action = player1.play(env,turn,args.secs)
            else : 
                while(action not in env.action_space()):
                  action = int(input("enter the number where you want to put the line"))
        else:
            if not args.agent2 == "self":
                action = player2.play(env,turn,args.secs)
            else : 
                while(action not in env.action_space()):
                  action = int(input("enter the number where you want to put the line"))

        reward = float(env.step(action,turn))
        if args.games < 3 or args.agent1 == "self" or args.agent2 == "self":
            env.render()
        if(reward == 0):
            turn = -turn
        action = float('inf')
    value = sum(env.boxes)
    if value > 0:
       wins +=1
    elif value < 0:
       loss +=1
    else:
       draws += 1

print(f"agent1 wins {wins} agent2 wins {loss} draws {draws}")
if not (args.agent1 == "self" or args.agent2 == "self"):
  plotter.plot(args.agent1,args.agent2,wins,draws,loss,args.dots)
  print('plot saved in DotGame/plots/')