import os
import math
import time
import pickle
import random
import numpy as np
import env.gameenv as gameenv
import agents.symmetry as symmetry

def get_greedy_action(state, turn):
    """Selects a box-completing action if one exists, otherwise selects randomly."""
    actions = gameenv.get_action_space(state)
    if not actions:
        return None
    for action in actions:
        next_state, reward = gameenv.step(state, action)
        next_turn = 1 if next_state[0][0][0] > 0 else -1
        if next_turn == turn:
            return action
    return random.choice(actions)

def get_canonical_state(state, h, v):
    """Returns the lexicographically smallest nested tuple symmetry and its sym_idx."""
    flat_state = np.array(state, dtype=np.float32)
    num_syms = symmetry.get_symmetry_count(h, v)
    best_state = None
    best_sym = 0
    for sym_idx in range(num_syms):
        sym_s, _ = symmetry.transform_state_and_policy(flat_state, None, sym_idx)
        sym_tuple = gameenv.array_to_nested_tuple(sym_s)
        if best_state is None or sym_tuple < best_state:
            best_state = sym_tuple
            best_sym = sym_idx
    return best_state, best_sym

class Play:
    """Monte Carlo Tree Search player interface for Dots and Boxes."""
    def __init__(self, h, v, model_path=None):
        self.h = h
        self.v = v
        self.positions = {}
        if model_path is None:
            model_path = f"trained_models/mcts{self.h}x{self.v}.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.positions = pickle.load(f)
                print("model loaded")
            except Exception:
                pass

    def play(self, state, turn, time_limit=1.0):
        canonical_state, sym_idx = get_canonical_state(state, self.h, self.v)
        root = self.positions.get(canonical_state)
        if root is None:
            root = Node((self.h, self.v), canonical_state, turn)
            self.positions[canonical_state] = root

        thinker = mcts()
        canonical_action = thinker.think(
            self.h,
            self.v,
            root=root,
            positions=self.positions,
            time_limit=time_limit
        )
        
        # Map canonical action back to the physical coordinate system
        physical_actions = gameenv.get_action_space(state)
        for act in physical_actions:
            sym_act = symmetry.transform_action(act, sym_idx, 2 * self.h - 1, 2 * self.v - 1)
            if sym_act == canonical_action:
                return act
        return random.choice(physical_actions) if physical_actions else canonical_action

class Node:
    """Monte Carlo Tree Search tree node."""
    def __init__(self, dots, state, turn):
        self.state = state
        self.dots = dots
        self.turn = turn
        self.children = {}
        self.total_reward = 0
        self.visits = 0

    def is_expanded(self):
        return len(gameenv.get_action_space(self.state)) == len(self.children)

    def expand(self, positions):
        actions = gameenv.get_action_space(self.state)
        h, v = self.dots
        for action in actions:
            new_state, reward = gameenv.step(self.state, action)
            canonical_state, sym_idx = get_canonical_state(new_state, h, v)
            
            new_node = positions.get(canonical_state)
            if new_node is None:
                new_turn = 1 if canonical_state[0][0][0] > 0 else -1
                new_node = Node(self.dots, canonical_state, new_turn)
                positions[canonical_state] = new_node
                
            self.children[action] = new_node

    def best_child(self, c_param=1.4, thinking=True):
        expected_reward = {}
        actions = gameenv.get_action_space(self.state)
        for action in actions:
            child = self.children.get(action)
            if child is None:
                continue
            q = (child.total_reward / (child.visits + 1e-8)) * self.turn
            if thinking:
                expected_reward[action] = q + c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-8))
            else:
                expected_reward[action] = q

        if not expected_reward:
            return random.choice(actions) if actions else None

        best_reward = max(expected_reward.values())
        best_actions = [a for a, r in expected_reward.items() if r == best_reward]
        return random.choice(best_actions)

class mcts:
    """Monte Carlo Tree Search execution manager."""
    def simulate(self, node):
        state = node.state
        turn = 1 if state[0][0][0] > 0 else -1
        while not gameenv.is_gameover(state):
            action = get_greedy_action(state, turn)
            state, reward = gameenv.step(state, action)
            turn = 1 if state[0][0][0] > 0 else -1

        p1_boxes = np.sum(state[3])
        p2_boxes = np.sum(state[4])
        if p1_boxes > p2_boxes:
            return 1
        elif p2_boxes > p1_boxes:
            return -1
        return 0

    def think(self, h, v, root, positions, time_limit=1.0):
        start_time = time.time()
        simulations = 0
        
        while time.time() - start_time < time_limit:
            path = []
            node = root
            path.append(node)
            action = None
            
            # 1. Selection
            while node.is_expanded() and not gameenv.is_gameover(node.state):
                action = node.best_child(c_param=1.4)
                if action is None:
                    break
                node = node.children[action]
                path.append(node)
                
            # 2. Expansion
            if not gameenv.is_gameover(node.state) and (action is not None or not node.children):
                node.expand(positions)
                action = node.best_child(c_param=1.4)
                if action is not None:
                    node = node.children[action]
                    path.append(node)
                    
            # 3. Simulation
            reward = self.simulate(node)
            
            # 4. Backpropagation (Path stack traversal, preventing OOM reference-cycle leaks)
            for p_node in path:
                p_node.visits += 1
                p_node.total_reward += reward
                
            simulations += 1
            
        best_action = root.best_child(c_param=0, thinking=False)
        return best_action

    def train(self, h, v, epochs, start_epoch=0):
        positions = {}
        model_path = f"trained_models/mcts{h}x{v}.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    positions = pickle.load(f)
                print(f"Loaded existing MCTS model from {model_path} to continue training.")
            except Exception:
                pass

        for episode in range(epochs):
            state = gameenv.get_initial_state((h, v))
            turn = 1
            
            canonical_state, sym_idx = get_canonical_state(state, h, v)
            root = positions.get(canonical_state)
            if root is None:
                root = Node((h, v), canonical_state, turn)
                positions[canonical_state] = root

            while not gameenv.is_gameover(state):
                # Search using the canonical root
                canonical_action = self.think(h, v, root, positions, time_limit=0.01)
                
                # Transform canonical action back to physical action space
                physical_actions = gameenv.get_action_space(state)
                chosen_physical_action = None
                for act in physical_actions:
                    sym_act = symmetry.transform_action(act, sym_idx, 2 * h - 1, 2 * v - 1)
                    if sym_act == canonical_action:
                        chosen_physical_action = act
                        break
                
                if chosen_physical_action is None:
                    chosen_physical_action = random.choice(physical_actions) if physical_actions else canonical_action
                
                # Step the actual game forward
                state, reward = gameenv.step(state, chosen_physical_action)
                turn = 1 if state[0][0][0] > 0 else -1
                
                # Get next canonical state and retrieve/create the next search tree root
                canonical_state, sym_idx = get_canonical_state(state, h, v)
                root = positions.get(canonical_state)
                if root is None:
                    root = Node((h, v), canonical_state, turn)
                    positions[canonical_state] = root

            print_interval = max(1, epochs // 10)
            if (episode + 1) % print_interval == 0:
                print(f"MCTS self-play game {episode} completed. Tree size: {len(positions)}")

        os.makedirs("trained_models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(positions, f)
        print(f"MCTS training complete. Model saved to {model_path}")
