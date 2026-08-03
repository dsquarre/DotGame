import os
import argparse
import random
import json
import numpy as np
import elo
import env.gameenv as gameenv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--h", type=int, default=3, help="no. of horizontal boxes in game")
parser.add_argument("--v", type=int, default=3, help="no. of vertical boxes in game")
parser.add_argument("--a", type=str, required=True, choices=["alphazero", "mcts", "dqn", "qt"], help="alphazero,mcts,dqn,qt")
parser.add_argument("--e", type=int, default=100, help="Number of epochs for training")
parser.add_argument("--eval_games", type=int, default=100, help="Number of games to play during each evaluation interval")
parser.add_argument("--eval_interval", type=int, default=50, help="Number of training epochs between evaluations")

args = parser.parse_args()
h_dots = args.h + 1
v_dots = args.v + 1

if args.a == 'alphazero':
    from agents.alphazero import az
    trainer = az()
if args.a == 'mcts':
    from agents.mcts import mcts
    trainer = mcts()
if args.a == 'dqn':
    from agents.dqn import dqn
    trainer = dqn()
if args.a == 'qt':
    from agents.tabular_q import qt
    trainer = qt()

def run_eval_games(agent_player, agent_name, h, v, count=100, agent=None):
    """Runs evaluation games between the target agent and Minmax."""
    from agents.minmax import Play as Play_Minmax
    minmax = Play_Minmax(h, v)
    agent_as_p1_wins = 0
    agent_as_p1_losses = 0
    agent_as_p1_draws = 0
    
    agent_as_p2_wins = 0
    agent_as_p2_losses = 0
    agent_as_p2_draws = 0
    
    for i in range(count):
        if i % 2 == 0:
            p1, p2 = agent_player, minmax
            p1_name, p2_name = agent_name, "minmax"
            p1_is_agent = True
        else:
            p1, p2 = minmax, agent_player
            p1_name, p2_name = "minmax", agent_name
            p1_is_agent = False
            
        state = gameenv.get_initial_state((h, v))
        turn = 1
        while not gameenv.is_gameover(state):
            if turn == 1:
                action = p1.play(state, turn, 0.03)
            else:
                action = p2.play(state, turn, 0.03)
            state, reward = gameenv.step(state, action)
            turn = 1 if state[0][0][0] > 0 else -1
            
        p1_boxes = np.sum(state[3])
        p2_boxes = np.sum(state[4])
        if p1_boxes > p2_boxes:
            outcome = 1.0
            if p1_is_agent:
                agent_as_p1_wins += 1
            else:
                agent_as_p2_losses += 1
        elif p2_boxes > p1_boxes:
            outcome = 0.0
            if p1_is_agent:
                agent_as_p1_losses += 1
            else:
                agent_as_p2_wins += 1
        else:
            outcome = 0.5
            if p1_is_agent:
                agent_as_p1_draws += 1
            else:
                agent_as_p2_draws += 1
        elo.log_match(p1_name, p2_name, outcome, h, v, agent=agent)
        
    agent_wins = agent_as_p1_wins + agent_as_p2_wins
    agent_losses = agent_as_p1_losses + agent_as_p2_losses
    draws = agent_as_p1_draws + agent_as_p2_draws
    win_rate = (agent_wins * 1.0 + draws * 0.5) / count
    print(f"Evaluation against Minmax ({count} games):")
    print(f"  Agent as P1: {agent_as_p1_wins} wins, {agent_as_p1_losses} losses, {agent_as_p1_draws} draws")
    print(f"  Agent as P2: {agent_as_p2_wins} wins, {agent_as_p2_losses} losses, {agent_as_p2_draws} draws")
    print(f"  Overall: Agent wins {agent_wins}, Losses {agent_losses}, Draws {draws} (Win Rate: {win_rate:.2%})")

if args.a in ['qt', 'dqn', 'mcts', 'alphazero']:
    ratings = elo._load_ratings(h_dots, v_dots, agent=args.a)
    
    # Initialize epoch 0 ratings history
    if args.a not in ratings:
        ratings[args.a] = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "rating": 0.0,
            "rating_history": [0.0],
            "epoch_history": {"0": 0.0}
        }
    if "minmax" not in ratings:
        ratings["minmax"] = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "rating": 1000.0,
            "rating_history": [1000.0],
            "epoch_history": {"0": 1000.0}
        }
    elo._save_ratings(ratings, h_dots, v_dots, agent=args.a)
        
    total_epochs = args.e
    interval = args.eval_interval
    
    agent_ratings_history = ratings.get(args.a, {})
    epoch_history = agent_ratings_history.get("epoch_history", {})
    
    start_epoch = interval
    if epoch_history:
        epoch_keys = [int(k) for k in epoch_history.keys() if int(k) > 0]
        if epoch_keys:
            start_epoch = max(epoch_keys) + interval
        
    for epochs_so_far in range(start_epoch, start_epoch + total_epochs, interval):
        # Ensure we always resume from the current champion checkpoint to prevent value drift
        import shutil
        grid_str = f"{args.h}x{args.v}"
        if args.a == 'dqn':
            src = f"trained_models/dqn{grid_str}.pth"
            dst = f"trained_models/dqn{grid_str}_best.pth"
        elif args.a == 'alphazero':
            src = f"trained_models/alphazero{grid_str}.pt"
            dst = f"trained_models/alphazero{grid_str}_best.pt"
        else:
            src, dst = None, None
            
        if src and dst and os.path.exists(dst):
            print(f"Resuming training from current champion checkpoint...")
            shutil.copy(dst, src)

        print(f"\n--- Training {args.a} Epochs {epochs_so_far - interval + 1} to {epochs_so_far} ---")
        trainer.train(h_dots, v_dots, interval, start_epoch=epochs_so_far - interval + 1)
        
        if args.a == 'qt':
            from agents.tabular_q import Play as AgentPlayer
        elif args.a == 'dqn':
            from agents.dqn import Play as AgentPlayer
        elif args.a == 'mcts':
            from agents.mcts import Play as AgentPlayer
        elif args.a == 'alphazero':
            from agents.alphazero import Play as AgentPlayer
            
        # 1. Champion selection head-to-head match (played first)
        import shutil
        os.makedirs("trained_models", exist_ok=True)
        
        if args.a == 'dqn':
            src = f"trained_models/dqn{args.h}x{args.v}.pth"
            dst = f"trained_models/dqn{args.h}x{args.v}_best.pth"
        elif args.a == 'alphazero':
            src = f"trained_models/alphazero{args.h}x{args.v}.pt"
            dst = f"trained_models/alphazero{args.h}x{args.v}_best.pt"
        else:
            src, dst = None, None
            
        if src and os.path.exists(src):
            if not os.path.exists(dst):
                # No champion exists yet! Save this one as the first champion.
                print(f"No champion model found. Saving current model as first champion...")
                shutil.copy(src, dst)
            else:
                # Champion exists. Play 100 games head-to-head!
                print(f"Evaluating new model against current champion...")
                # Instantiate challenger (loading src) and champion (loading dst)
                challenger = AgentPlayer(h_dots, v_dots, model_path=src)
                champion = AgentPlayer(h_dots, v_dots, model_path=dst)
                
                challenger_as_p1_wins = 0
                challenger_as_p1_losses = 0
                challenger_as_p1_draws = 0
                
                challenger_as_p2_wins = 0
                challenger_as_p2_losses = 0
                challenger_as_p2_draws = 0
                
                champion_as_p1_wins = 0
                champion_as_p1_losses = 0
                champion_as_p1_draws = 0
                
                champion_as_p2_wins = 0
                champion_as_p2_losses = 0
                champion_as_p2_draws = 0
                
                # Play 100 games (challenger vs champion)
                for g_idx in range(100):
                    # Alternate starting player
                    if g_idx % 2 == 0:
                        p1, p2 = challenger, champion
                        p1_is_challenger = True
                    else:
                        p1, p2 = champion, challenger
                        p1_is_challenger = False
                        
                    state = gameenv.get_initial_state((h_dots, v_dots))
                    turn = 1
                    while not gameenv.is_gameover(state):
                        action = p1.play(state, turn) if turn == 1 else p2.play(state, turn)
                        state, reward = gameenv.step(state, action)
                        turn = 1 if state[0][0][0] > 0 else -1
                        
                    p1_boxes = np.sum(state[3])
                    p2_boxes = np.sum(state[4])
                    if p1_boxes > p2_boxes:
                        if p1_is_challenger:
                            challenger_as_p1_wins += 1
                            champion_as_p2_losses += 1
                        else:
                            champion_as_p1_wins += 1
                            challenger_as_p2_losses += 1
                    elif p2_boxes > p1_boxes:
                        if p1_is_challenger:
                            champion_as_p2_wins += 1
                            challenger_as_p1_losses += 1
                        else:
                            challenger_as_p2_wins += 1
                            champion_as_p1_losses += 1
                    else:
                        if p1_is_challenger:
                            challenger_as_p1_draws += 1
                            champion_as_p2_draws += 1
                        else:
                            champion_as_p1_draws += 1
                            challenger_as_p2_draws += 1
                        
                challenger_wins = challenger_as_p1_wins + challenger_as_p2_wins
                champion_wins = champion_as_p1_wins + champion_as_p2_wins
                draws = challenger_as_p1_draws + challenger_as_p2_draws
                score = challenger_wins * 1.0 + draws * 0.5
                win_rate = score / 100.0
                
                print(f"Head-to-head (Challenger vs Champion over 100 games):")
                print(f"  Challenger as P1: {challenger_as_p1_wins} wins, {challenger_as_p1_losses} losses, {challenger_as_p1_draws} draws")
                print(f"  Challenger as P2: {challenger_as_p2_wins} wins, {challenger_as_p2_losses} losses, {challenger_as_p2_draws} draws")
                print(f"  Champion as P1:   {champion_as_p1_wins} wins, {champion_as_p1_losses} losses, {champion_as_p1_draws} draws")
                print(f"  Champion as P2:   {champion_as_p2_wins} wins, {champion_as_p2_losses} losses, {champion_as_p2_draws} draws")
                print(f"  Overall: Challenger wins {challenger_wins}, Champion wins {champion_wins}, Draws {draws} (Challenger Win Rate: {win_rate:.2%})")
                
                if win_rate >= 0.50:
                    print(f"Challenger defeated Champion with {win_rate:.2%} win rate! Overwriting champion checkpoint...")
                    shutil.copy(src, dst)
                else:
                    print(f"Challenger failed to defeat Champion ({win_rate:.2%} < 50.00%). Champion checkpoint preserved.")

        # 2. Evaluate the champion (stored in dst or the default agent_player) against Minmax
        eval_path = dst if (dst and os.path.exists(dst)) else None
        agent_player = AgentPlayer(h_dots, v_dots, model_path=eval_path)
        
        print(f"Playing {args.eval_games} evaluation games (Champion vs Minmax)...")
        run_eval_games(agent_player, args.a, h_dots, v_dots, count=args.eval_games, agent=args.a)
        
        ratings = elo._load_ratings(h_dots, v_dots, agent=args.a)
        if args.a not in ratings:
            ratings[args.a] = {"rating": 0.0, "epoch_history": {}}
        if "epoch_history" not in ratings[args.a]:
            ratings[args.a]["epoch_history"] = {}
            
        current_elo = ratings.get(args.a, {}).get("rating", 100.0)
        
        # Log evaluation results for this epoch
        ratings[args.a]["epoch_history"][str(epochs_so_far)] = current_elo
        if "minmax" in ratings:
            if "epoch_history" not in ratings["minmax"]:
                ratings["minmax"]["epoch_history"] = {}
            ratings["minmax"]["epoch_history"][str(epochs_so_far)] = ratings["minmax"]["rating"]
            
        elo._save_ratings(ratings, h_dots, v_dots, agent=args.a)
        print(f"Epoch {epochs_so_far} - ELO rating of {args.a} (Champion): {current_elo:.2f}")
        
        plt.figure(figsize=(10, 6))
        
        # Load and plot histories for all learning agents
        for agent_name in ['dqn', 'qt', 'mcts', 'alphazero']:
            try:
                agent_ratings = elo._load_ratings(h_dots, v_dots, agent=agent_name)
                if agent_name in agent_ratings and "epoch_history" in agent_ratings[agent_name]:
                    hist = agent_ratings[agent_name]["epoch_history"]
                    if not hist:
                        continue
                    sorted_keys = sorted([int(k) for k in hist.keys()])
                    sorted_vals = [hist[str(k)] for k in sorted_keys]
                    
                    if agent_name == 'dqn':
                        label, color = 'DQN (2D CNN)', 'blue'
                    elif agent_name == 'qt':
                        label, color = 'Tabular Q', 'green'
                    elif agent_name == 'mcts':
                        label, color = 'MCTS', 'purple'
                    elif agent_name == 'alphazero':
                        label, color = 'AlphaZero', 'orange'
                        
                    plt.plot(sorted_keys, sorted_vals, marker='o', label=label, color=color)
            except Exception:
                pass

        # Plot fixed benchmark lines
        plt.axhline(y=1000, color='red', linestyle='--', linewidth=2, label='Minmax (0.02s) (Fixed 1000)')
        if args.h == 2 and args.v == 2:
            plt.axhline(y=1250, color='black', linestyle=':', linewidth=2, label='Perfect Solver (Fixed 1250)')
        
        plt.xlabel('Training Epochs')
        plt.ylabel('ELO Rating')
        plt.title(f'Agent ELO Progression ({args.h}x{args.v} Boxes)')
        plt.grid(True)
        plt.legend()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/elo_learning_curve_{args.h}x{args.v}.png")
        plt.close()
        
    print(f"\nTraining and ELO evaluation loop complete!")
    print(f"Learning curve saved to plots/elo_learning_curve_{args.h}x{args.v}.png")
else:
    trainer.train(h_dots, v_dots, args.e)