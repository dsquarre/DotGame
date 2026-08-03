import os
import random
import pickle
from collections import defaultdict
import numpy as np
import env.gameenv as gameenv

class qt:
    """Tabular Q-learning trainer for the Dots and Boxes game."""
    def train(self, h, v, epochs, start_epoch=0):
        discount_factor = 0.7
        learning_rate = 0.1
        Q = defaultdict(lambda: defaultdict(float))
        model_path = f"trained_models/Q{h}x{v}.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    loaded = pickle.load(f)
                    for s, actions in loaded.items():
                        for a, val in actions.items():
                            Q[s][a] = val
                print(f"Loaded existing Q-table from {model_path} to continue training.")
            except Exception:
                pass
        
        for episode in range(epochs + 1):
            state = gameenv.get_initial_state((h, v))
            turn = 1 if state[0][0][0] > 0 else -1
            
            while not gameenv.is_gameover(state):
                valid_actions = gameenv.get_action_space(state)
                global_episode = start_epoch + episode
                epsilon = max(0.01, 0.995**global_episode)
                
                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    valid_q_values = [Q[state][a] for a in valid_actions]
                    if turn == 1:
                        action = valid_actions[np.argmax(valid_q_values)]
                    else:
                        action = valid_actions[np.argmin(valid_q_values)]
                        
                q_state_action = Q[state][action]
                
                next_state, reward = gameenv.step(state, action)
                
                if gameenv.is_gameover(next_state):
                    best_q_next = reward
                    next_turn = turn
                else:
                    valid_actions_next = gameenv.get_action_space(next_state)
                    valid_q_values_next = [Q[next_state][a] for a in valid_actions_next]
                    
                    next_turn = 1 if next_state[0][0][0] > 0 else -1
                    
                    if next_turn == 1:
                        best_q_next = np.max(valid_q_values_next)
                    else:
                        best_q_next = np.min(valid_q_values_next)
                        
                Q[state][action] += learning_rate * (reward + discount_factor * best_q_next - q_state_action)
                
                state = next_state
                turn = next_turn
                
            print_interval = max(1, epochs // 10)
            if (episode + 1) % print_interval == 0:
                print(f"no of q values stored: {len(Q)} ,episodes : {episode + 1}/{epochs} (epsilon: {epsilon:.4f})")
                
        os.makedirs("trained_models", exist_ok=True)
        serializable_Q = {s: dict(actions) for s, actions in Q.items()}
        with open(f"trained_models/Q{h}x{v}.pkl", "wb") as f:
            pickle.dump(serializable_Q, f)
        print(f"training finished, model stored in trained_models/Q{h}x{v}.pkl")

class Play:
    """Tabular Q-learning player interface for Dots and Boxes."""
    def __init__(self, h, v):
        self.h = h
        self.v = v
        self.Q = defaultdict(lambda: defaultdict(float))
        model_path = f"trained_models/Q{h}x{v}.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    loaded = pickle.load(f)
                    for s, actions in loaded.items():
                        for a, val in actions.items():
                            self.Q[s][a] = val
            except Exception:
                pass

    def play(self, state, turn, secs=0):
        actions = gameenv.get_action_space(state)
        if not actions:
            return None
        valid_q_values = np.array([self.Q[state][a] for a in actions], dtype=np.float32)
        
        import random
        cdef float best_q = np.max(valid_q_values) if turn == 1 else np.min(valid_q_values)
        
        # Select all actions within a safe 0.1 threshold of the best action
        if turn == 1:
            safe_indices = [i for i, q in enumerate(valid_q_values) if q >= best_q - 0.1]
        else:
            safe_indices = [i for i, q in enumerate(valid_q_values) if q <= best_q + 0.1]
            
        best_idx = random.choice(safe_indices)
        return actions[best_idx]
