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

def compress_state_4x4(state):
    # 1. Line bitmask (24 lines on a 7x7 grid)
    bitmask = 0
    bit_idx = 0
    for r in range(7):
        for c in range(7):
            if (r + c) % 2 == 1:
                if state[1][r][c] > 0.0 or state[2][r][c] > 0.0:
                    bitmask |= (1 << bit_idx)
                bit_idx += 1
                
    # 2. Box ownership (9 boxes)
    box_code = 0
    box_idx = 0
    for br in [1, 3, 5]:
        for bc in [1, 3, 5]:
            if state[3][br][bc] > 0.0:
                code = 1
            elif state[4][br][bc] > 0.0:
                code = 2
            else:
                code = 0
            box_code |= (code << (2 * box_idx))
            box_idx += 1
            
    # 3. Turn
    turn_bit = 1 if state[0][0][0] > 0.0 else 0
    
    # 4. Consecutive bonus
    bonus_bit = 1 if state[5][0][0] > 0.0 else 0
    
    return (bitmask << 20) | (box_code << 2) | (turn_bit << 1) | bonus_bit

class Play:
    """Perfect solver player interface for Dots and Boxes using compact databases."""
    def __init__(self, h, v, model_path=None):
        self.h = h
        self.v = v
        self.best_actions = {}
        
        if h == 3 and v == 3:
            db_path = "trained_models/solved2x2_compact.pkl"
            if os.path.exists(db_path):
                try:
                    with open(db_path, "rb") as f:
                        self.best_actions = pickle.load(f)
                    print(f"Loaded perfect compact 2x2 Dots & Boxes database containing {len(self.best_actions)} states.")
                except Exception as e:
                    print(f"Error loading solved compact database: {e}")
            else:
                print("Warning: solved2x2_compact.pkl not found! Falling back to Minmax search.")
                from agents.minmax import Play as MinmaxPlay
                self.fallback_player = MinmaxPlay(h, v)
        elif h == 4 and v == 4:
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
            print(f"Exhaustive solver is only pre-computed for 2x2 and 3x3. Falling back to Minmax search for {h-1}x{v-1}.")
            from agents.minmax import Play as MinmaxPlay
            self.fallback_player = MinmaxPlay(h, v)

    def play(self, state, turn, time_limit=1.0):
        if self.h == 4 and self.v == 4:
            c_state = compress_state_4x4(state)
            if self.best_actions and c_state in self.best_actions:
                action_idx = self.best_actions[c_state]
                return (action_idx // 7, action_idx % 7)
        else:
            c_state = compress_state(state)
            if self.best_actions and c_state in self.best_actions:
                return self.best_actions[c_state]
            
        if hasattr(self, 'fallback_player'):
            return self.fallback_player.play(state, turn, time_limit)
            
        actions = gameenv.get_action_space(state)
        return random.choice(actions) if actions else None
