import env.gameenv as gameenv
import numpy as np
import elo
from plots.plot import Plot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--h", type=int, default=4, help="no. of horizontal dots in game")
parser.add_argument("--v", type=int, default=4, help="no. of vertical dots in game")
parser.add_argument("--p1", type=str, required=True, help="alphazero,minmax,mcts,dqn,qt,self")
parser.add_argument("--p2", type=str, required=True, help="alphazero,minmax,mcts,dqn,qt,self")
parser.add_argument("--g", type=int, default=1, help="Number of games to play")
parser.add_argument("--t", type=float, default=1.0, help="thinking time limit in seconds")

args = parser.parse_args()
if args.p1 == 'alphazero':
    from agents.alphazero import Play as play1
if args.p1 == 'mcts':
    from agents.mcts import Play as play1
if args.p1 == 'dqn':
    from agents.dqn import Play as play1
if args.p1 == 'qt':
    from agents.tabular_q import Play as play1
if args.p1 == 'minmax':
    from agents.minmax import Play as play1
if not args.p1 == "self":
    player1 = play1(args.h, args.v)

if args.p2 == 'alphazero':
    from agents.alphazero import Play
if args.p2 == 'mcts':
    from agents.mcts import Play
if args.p2 == 'dqn':
    from agents.dqn import Play
if args.p2 == 'qt':
    from agents.tabular_q import Play
if args.p2 == 'minmax':
    from agents.minmax import Play
if not args.p2 == "self":
    player2 = Play(args.h, args.v)

plotter = Plot()
wins = 0
draws = 0
loss = 0

for game in range(args.g):
    state = gameenv.get_initial_state((args.h, args.v))
    turn = 1
    action = None
    
    while not gameenv.is_gameover(state):
        if args.g < 3 or args.p1 == "self" or args.p2 == "self":
            gameenv.render(state)
            
        action = None
        if turn == 1:
            if not args.p1 == "self":
                action = player1.play(state, turn, args.t)
            else:
                actions = gameenv.get_action_space(state)
                while action not in actions:
                    try:
                        action_str = input("Player 1 move (row col) > ")
                        r, c = map(int, action_str.replace(',', ' ').split())
                        action = (r, c)
                        if action not in actions:
                            print("Invalid move! Coordinates must represent an empty line.")
                    except ValueError:
                        print("Invalid format. Enter two numbers separated by space (e.g. 0 1)")
        else:
            if not args.p2 == "self":
                action = player2.play(state, turn, args.t)
            else:
                actions = gameenv.get_action_space(state)
                while action not in actions:
                    try:
                        action_str = input("Player 2 move (row col) > ")
                        r, c = map(int, action_str.replace(',', ' ').split())
                        action = (r, c)
                        if action not in actions:
                            print("Invalid move! Coordinates must represent an empty line.")
                    except ValueError:
                        print("Invalid format. Enter two numbers separated by space (e.g. 0 1)")

        state, reward = gameenv.step(state, action)
        
        # Extract turn for the next step from state Channel 0 of the latest frame
        turn = 1 if state[-1][0][0][0] > 0 else -1

    if args.g < 3 or args.p1 == "self" or args.p2 == "self":
        gameenv.render(state)

    # Calculate final box difference from final state
    p1_boxes = np.sum(state[-1][3])
    p2_boxes = np.sum(state[-1][4])
    value = p1_boxes - p2_boxes
    
    if value > 0:
        wins += 1
        outcome = 1.0
    elif value < 0:
        loss += 1
        outcome = 0.0
    else:
        draws += 1
        outcome = 0.5

    elo.log_match(args.p1, args.p2, outcome, args.h, args.v)
    ratings = elo._load_ratings(args.h, args.v)
    p1_elo = ratings.get(args.p1, {}).get("rating", 100.0)
    p2_elo = ratings.get(args.p2, {}).get("rating", 100.0)
    print(f"Updated Ratings - {args.p1}: {p1_elo:.2f}, {args.p2}: {p2_elo:.2f}\n")

print(f"agent1 wins {wins} agent2 wins {loss} draws {draws}")
if not (args.p1 == "self" or args.p2 == "self"):
    plotter.plot(args.p1, args.p2, wins, draws, loss, (args.h, args.v))
    print('plot saved in DotGame/plots/')