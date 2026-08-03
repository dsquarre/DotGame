import os
import pickle
import random
import env.gameenv as gameenv

def compress_state(state):
    # state shape (6, H, W) where H=5, W=5
    H = len(state[0])
    W = len(state[0][0])
    
    # 1. Line bitmask (12 bits)
    bitmask = 0
    bit_idx = 0
    for r in range(H):
        for c in range(W):
            if (r + c) % 2 == 1:
                if state[1][r][c] > 0.0 or state[2][r][c] > 0.0:
                    bitmask |= (1 << bit_idx)
                bit_idx += 1
                
    # 2. Box ownership (4 boxes)
    boxes = []
    for br in [1, 3]:
        for bc in [1, 3]:
            if state[3][br][bc] > 0.0:
                boxes.append(1)
            elif state[4][br][bc] > 0.0:
                boxes.append(2)
            else:
                boxes.append(0)
    box_tuple = tuple(boxes)
    
    # 3. Turn
    turn = 1 if state[0][0][0] > 0 else -1
    
    # 4. Consecutive bonus
    bonus = 1 if state[5][0][0] > 0 else 0
    
    return (bitmask, box_tuple, turn, bonus)

class Play:
    """Perfect solver player interface for 3x3 Dots and Boxes using compact database."""
    def __init__(self, h, v, model_path=None):
        self.h = h
        self.v = v
        self.best_actions = {}
        
        if h == 3 and v == 3:
            db_path = "trained_models/solved3x3_compact.pkl"
            if os.path.exists(db_path):
                try:
                    with open(db_path, "rb") as f:
                        self.best_actions = pickle.load(f)
                    print(f"Loaded perfect compact 3x3 Dots & Boxes database containing {len(self.best_actions)} states.")
                except Exception as e:
                    print(f"Error loading solved compact database: {e}")
            else:
                print("Warning: solved3x3_compact.pkl not found! Falling back to Minmax search.")
                from agents.minmax import Play as MinmaxPlay
                self.fallback_player = MinmaxPlay(h, v)
        else:
            print(f"Exhaustive solver is only pre-computed for 3x3. Falling back to Minmax search for {h}x{v}.")
            from agents.minmax import Play as MinmaxPlay
            self.fallback_player = MinmaxPlay(h, v)

    def play(self, state, turn, time_limit=1.0):
        c_state = compress_state(state)
        if self.best_actions and c_state in self.best_actions:
            return self.best_actions[c_state]
            
        if hasattr(self, 'fallback_player'):
            return self.fallback_player.play(state, turn, time_limit)
            
        actions = gameenv.get_action_space(state)
        return random.choice(actions) if actions else None
