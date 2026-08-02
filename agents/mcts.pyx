import os
import math
import time
import pickle
import random
import numpy as np
import env.gameenv as gameenv

def get_greedy_action(state, turn):
    """Selects a box-completing action if one exists, otherwise selects randomly."""
    actions = gameenv.get_action_space(state)
    if not actions:
        return None
    for action in actions:
        next_state, reward = gameenv.step(state, action)
        next_turn = 1 if next_state[-1][0][0][0] > 0 else -1
        if next_turn == turn:
            return action
    return random.choice(actions)

class Play:
    """Monte Carlo Tree Search player interface for Dots and Boxes."""
    def __init__(self, h, v):
        self.h = h
        self.v = v
        self.positions = {}
        try:
            with open(f"trained_models/mcts{self.h}x{self.v}.pkl", "rb") as f:
                self.positions = pickle.load(f)
                print("model loaded")
        except Exception:
            pass

    def play(self, state, turn, time_limit=1.0):
        root = self.positions.get(state)
        if root is None:
            root = Node((self.h, self.v), state, turn)
            self.positions[state] = root

        thinker = mcts()
        action = thinker.think(
            self.h,
            self.v,
            root=root,
            positions=self.positions,
            time_limit=time_limit
        )
        return action

class Node:
    """Monte Carlo Tree Search tree node."""
    def __init__(self, dots, state, turn, parent=None):
        self.state = state
        self.dots = dots
        self.turn = turn
        self.parent = parent
        self.children = {}
        self.total_reward = 0
        self.visits = 0

    def is_expanded(self):
        return len(gameenv.get_action_space(self.state)) == len(self.children)

    def expand(self, positions):
        actions = gameenv.get_action_space(self.state)
        for action in actions:
            new_state, reward = gameenv.step(self.state, action)
            new_turn = 1 if new_state[-1][0][0][0] > 0 else -1
            new_node = Node(self.dots, new_state, new_turn, self)
            self.children[action] = new_node
            positions[new_state] = new_node

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
        turn = 1 if state[-1][0][0][0] > 0 else -1
        while not gameenv.is_gameover(state):
            action = get_greedy_action(state, turn)
            state, reward = gameenv.step(state, action)
            turn = 1 if state[-1][0][0][0] > 0 else -1

        p1_boxes = np.sum(state[-1][3])
        p2_boxes = np.sum(state[-1][4])
        if p1_boxes > p2_boxes:
            return 1
        elif p2_boxes > p1_boxes:
            return -1
        return 0

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += reward
            if not node.parent:
                break
            node = node.parent
        return node

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
            root = positions.get(state)
            if root is None:
                root = Node((h, v), state, turn)
                positions[state] = root

            while not gameenv.is_gameover(state):
                action = self.think(h, v, root, positions, time_limit=0.01)
                state, reward = gameenv.step(state, action)
                turn = 1 if state[-1][0][0][0] > 0 else -1
                root = positions.get(state)
                if root is None:
                    root = Node((h, v), state, turn)
                    positions[state] = root

            if episode % 10 == 0:
                print(f"MCTS self-play game {episode} completed. Tree size: {len(positions)}")

        os.makedirs("trained_models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(positions, f)
        print(f"MCTS training complete. Model saved to {model_path}")

    def think(self, h, v, root, positions, time_limit):
        start_time = time.time()
        node = root
        node.parent = None
        simulations_run = 0

        while time.time() - start_time < time_limit:
            node = root
            gameover = gameenv.is_gameover(node.state)

            while node.children and not gameover:
                action = node.best_child()
                if action is None:
                    break
                node = node.children[action]
                gameover = gameenv.is_gameover(node.state)

            reward = 0
            if not gameover:
                node.expand(positions)
                if node.children:
                    child = random.choice(list(node.children.values()))
                    reward = self.simulate(child)
                    node = child
            else:
                p1_boxes = np.sum(node.state[-1][3])
                p2_boxes = np.sum(node.state[-1][4])
                if p1_boxes > p2_boxes:
                    reward = 1
                elif p2_boxes > p1_boxes:
                    reward = -1
                else:
                    reward = 0

            root = self.backpropagate(node, reward)
            simulations_run += 1

        action = root.best_child(thinking=False)
        return action
