import os
import sys
import pickle
import numpy as np

# Add workspace to path
sys.path.append(os.getcwd())
import env.gameenv as gameenv
from agents.solver import compress_state_4x4

# Load the solved compact database
db_path = "trained_models/solved4x4_compact.pkl"
if not os.path.exists(db_path):
    print(f"Error: {db_path} not found! Please run the solver first.")
    sys.exit(1)

with open(db_path, "rb") as f:
    best_actions = pickle.load(f)

def decode_action(action_idx):
    return (action_idx // 7, action_idx % 7)

def get_outcome_score(state):
    """Calculates box difference (P1 - P2) at terminal states."""
    p1_boxes = np.sum(state[3])
    p2_boxes = np.sum(state[4])
    return int(p1_boxes - p2_boxes)

def evaluate_state_value(state, alpha=-100, beta=100):
    """
    Computes the exact minimax value of any state.
    Uses the solved database to fast-forward evaluations along optimal paths.
    Runs a highly optimized alpha-beta search with window constraints.
    """
    if gameenv.is_gameover(state):
        return get_outcome_score(state)
        
    c_state = compress_state_4x4(state)
    if c_state in best_actions:
        # If the state is in the database, follow the best actions to the end
        curr = state
        while not gameenv.is_gameover(curr):
            c_curr = compress_state_4x4(curr)
            if c_curr in best_actions:
                act = decode_action(best_actions[c_curr])
                curr, _ = gameenv.step(curr, act)
            else:
                return evaluate_state_value(curr, alpha, beta)
        return get_outcome_score(curr)
        
    # If not in database, run a quick recursive minimax search with Alpha-Beta Pruning
    turn = 1 if state[0][0][0] > 0 else -1
    actions = gameenv.get_action_space(state)
    
    if turn == 1:
        best_val = -100
        for action in actions:
            next_state, _ = gameenv.step(state, action)
            val = evaluate_state_value(next_state, alpha, beta)
            if val > best_val:
                best_val = val
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break # Beta cut-off
        return best_val
    else:
        best_val = 100
        for action in actions:
            next_state, _ = gameenv.step(state, action)
            val = evaluate_state_value(next_state, alpha, beta)
            if val < best_val:
                best_val = val
            beta = min(beta, best_val)
            if beta <= alpha:
                break # Alpha cut-off
        return best_val

def measure_play():
    state = gameenv.get_initial_state((4, 4))
    move_count = 1
    
    print("=" * 60)
    print("EMPERICAL MARGIN OF ERROR ANALYSIS (4x4 Dots / 3x3 Boxes)")
    print("=" * 60)
    
    while not gameenv.is_gameover(state):
        turn = 1 if state[0][0][0] > 0 else -1
        player_name = "Player 1" if turn == 1 else "Player 2"
        
        # 1. Get the current optimal outcome value (instant database lookup)
        curr_optimal_val = evaluate_state_value(state)
        
        # 2. Evaluate all legal actions
        actions = gameenv.get_action_space(state)
        optimal_actions = []
        blunder_actions = []
        
        for action in actions:
            next_state, _ = gameenv.step(state, action)
            
            # Use tight zero-window bounds depending on whose turn it is
            if turn == 1:
                # P1 wants to maximize. We check if next_state value is >= curr_optimal_val
                val = evaluate_state_value(next_state, alpha=curr_optimal_val - 1, beta=curr_optimal_val)
                if val >= curr_optimal_val:
                    optimal_actions.append(action)
                else:
                    blunder_actions.append(action)
            else:
                # P2 wants to minimize. We check if next_state value is <= curr_optimal_val
                val = evaluate_state_value(next_state, alpha=curr_optimal_val, beta=curr_optimal_val + 1)
                if val <= curr_optimal_val:
                    optimal_actions.append(action)
                else:
                    blunder_actions.append(action)
                
        # 3. Print analysis for this move
        total_legal = len(actions)
        num_optimal = len(optimal_actions)
        pct_optimal = (num_optimal / total_legal) * 100
        
        print(f"\nMove {move_count}: {player_name}'s Turn")
        print(f"  Current Board Value (P1 - P2): {curr_optimal_val}")
        print(f"  Legal Moves: {total_legal} | Optimal: {num_optimal} ({pct_optimal:.1f}%) | Blunders: {total_legal - num_optimal}")
        if blunder_actions:
            print(f"  Sample Blunders: {', '.join(str(a) for a in blunder_actions[:5])}")
            
        # 4. Take the perfect move from database to advance the path
        c_state = compress_state_4x4(state)
        best_act_idx = best_actions[c_state]
        best_act = decode_action(best_act_idx)
        
        state, _ = gameenv.step(state, best_act)
        move_count += 1
        
    print("\n" + "=" * 60)
    print(f"Game finished! Final score: {get_outcome_score(state)}")
    print("=" * 60)

if __name__ == "__main__":
    measure_play()
