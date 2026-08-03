import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import env.gameenv as gameenv
import agents.symmetry as symmetry

class NN(nn.Module):
    """AlphaZero dual-headed Policy/Value network."""
    def __init__(self, in_channels, output_dim, H, W):
        super(NN, self).__init__()
        self.H = H
        self.W = W
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc_policy = nn.Sequential(
            nn.Linear(128 * H * W, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.fc_value = nn.Sequential(
            nn.Linear(128 * H * W, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        policy_logits = self.fc_policy(features)
        value = self.fc_value(features)
        return policy_logits, value

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=0) + 1e-8)

class Node:
    """AlphaZero MCTS Node representation."""
    def __init__(self, dots, state, prior=0.0):
        self.dots = dots
        self.state = state
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def best_child_node_selection(self, c_param=1.4):
        """Standard UCT action selection based on node child statistics (no neural network call)."""
        best_score = -float('inf')
        best_action = None
        turn = 1 if self.state[0][0][0] > 0 else -1
        
        for action, child in self.children.items():
            u = c_param * child.prior * math.sqrt(self.visit_count + 1) / (child.visit_count + 1)
            q = child.value * turn
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action

    def expand_node_children(self, probs, c_param=1.4, training=True):
        """Expands the current node and sets child priors from the given probability vector."""
        H = 2 * self.dots[0] - 1
        W = 2 * self.dots[1] - 1
        actions = gameenv.get_action_space(self.state)
        policy = np.zeros(H * W)
        valid_indices = [r * W + c for r, c in actions]
        sum_prob = sum(probs[idx] for idx in valid_indices) + 1e-8
        for idx in valid_indices:
            policy[idx] = probs[idx] / sum_prob
            
        if training:
            alpha = 0.6
            epsilon = 0.3
            noise = np.random.dirichlet([alpha] * (H * W))
            for idx in valid_indices:
                policy[idx] = (1 - epsilon) * policy[idx] + epsilon * noise[idx]
                
        for action in actions:
            idx = action[0] * W + action[1]
            self.children[action] = Node(self.dots, None, policy[idx])

    def best_child(self, model, c_param=1.4, training=True):
        """Single-state expansion and selection for sequential evaluations."""
        H = 2 * self.dots[0] - 1
        W = 2 * self.dots[1] - 1
        in_channels = 6
        
        device = next(model.parameters()).device
        state_np = np.array(self.state, dtype=np.float32).reshape(in_channels, H, W)
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        if not self.children:
            actions = gameenv.get_action_space(self.state)
            with torch.no_grad():
                logits, _ = model(state_tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            self.expand_node_children(probs, c_param, training)
            
        return self.best_child_node_selection(c_param)

class alphazero:
    """AlphaZero self-play MCTS execution manager."""
    def mcts(self, dots, root, simulations, model, training=True):
        """Sequential single-state MCTS for evaluations."""
        device = next(model.parameters()).device
        H = 2 * dots[0] - 1
        W = 2 * dots[1] - 1
        in_channels = 6
        
        for _ in range(simulations):
            node = root
            search_path = [node]
            
            while node.state is not None and node.children:
                action = node.best_child_node_selection(c_param=1.4)
                child = node.children[action]
                if child.state is None:
                    child.state, _ = gameenv.step(node.state, action)
                node = child
                search_path.append(node)
                
            if not gameenv.is_gameover(node.state):
                if not node.children:
                    state_np = np.array(node.state, dtype=np.float32).reshape(in_channels, H, W)
                    state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        logits, val = model(state_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                    node.expand_node_children(probs, c_param=1.4, training=training)
                    value = val.item()
                else:
                    value = node.value
            else:
                p1_boxes = np.sum(node.state[3])
                p2_boxes = np.sum(node.state[4])
                value = 1.0 if p1_boxes > p2_boxes else (-1.0 if p2_boxes > p1_boxes else 0.0)
                
            for p_node in search_path:
                p_node.visit_count += 1
                p_node.value_sum += value
                
        return root

    def batched_mcts(self, dots, roots, simulations, model, training=True):
        """Parallel batched MCTS running single-batch GPU evaluations across multiple active games."""
        device = next(model.parameters()).device
        H = 2 * dots[0] - 1
        W = 2 * dots[1] - 1
        in_channels = 6
        num_games = len(roots)
        
        for _ in range(simulations):
            leaf_nodes = []
            search_paths = []
            
            # 1. Selection (CPU side path traversal)
            for g in range(num_games):
                node = roots[g]
                path = [node]
                while node.state is not None and node.children:
                    action = node.best_child_node_selection(c_param=1.4)
                    child = node.children[action]
                    if child.state is None:
                        child.state, _ = gameenv.step(node.state, action)
                    node = child
                    path.append(node)
                leaf_nodes.append(node)
                search_paths.append(path)
                
            # 2. Batch expansion & evaluation on GPU
            non_terminal_indices = []
            non_terminal_states = []
            for g in range(num_games):
                node = leaf_nodes[g]
                if not gameenv.is_gameover(node.state):
                    if not node.children:
                        non_terminal_indices.append(g)
                        state_np = np.array(node.state, dtype=np.float32).reshape(in_channels, H, W)
                        non_terminal_states.append(state_np)
                        
            if non_terminal_states:
                states_tensor = torch.tensor(np.array(non_terminal_states), dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits, vals = model(states_tensor)
                logits = logits.cpu().numpy()
                vals = vals.cpu().numpy()
                
                for idx, g in enumerate(non_terminal_indices):
                    node = leaf_nodes[g]
                    probs = softmax(logits[idx])
                    val = vals[idx][0]
                    node.expand_node_children(probs, c_param=1.4, training=training)
                    
                    # Backpropagate non-terminal leaf evaluation
                    for p_node in search_paths[g]:
                        p_node.visit_count += 1
                        p_node.value_sum += val
                        
            # Handle terminal leaf nodes
            for g in range(num_games):
                node = leaf_nodes[g]
                if gameenv.is_gameover(node.state):
                    p1_boxes = np.sum(node.state[3])
                    p2_boxes = np.sum(node.state[4])
                    val = 1.0 if p1_boxes > p2_boxes else (-1.0 if p2_boxes > p1_boxes else 0.0)
                    
                    # Backpropagate terminal outcome
                    for p_node in search_paths[g]:
                        p_node.visit_count += 1
                        p_node.value_sum += val

class RecentBuffer:
    """FIFO replay buffer for recent experiences."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class HistoricalBuffer:
    """Diverse long-term replay buffer with random dropout replacement when full."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = transition

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DualReplayMemory:
    """Manages sampling and updating across recent and historical buffers."""
    def __init__(self, recent_capacity=10000, historical_capacity=20000, push_history_prob=0.10):
        self.recent = RecentBuffer(recent_capacity)
        self.historical = HistoricalBuffer(historical_capacity)
        self.push_history_prob = push_history_prob

    def push(self, state, policy, value):
        transition = (state, policy, value)
        self.recent.push(transition)
        if random.random() < self.push_history_prob:
            self.historical.push(transition)

    def sample(self, batch_size):
        recent_size = batch_size // 2
        hist_size = batch_size - recent_size
        
        if len(self.historical) < hist_size:
            return self.recent.sample(batch_size)
            
        recent_sample = self.recent.sample(recent_size)
        hist_sample = self.historical.sample(hist_size)
        return recent_sample + hist_sample

    def __len__(self):
        return len(self.recent)

class az:
    """AlphaZero training manager."""
    def __init__(self):
        self.replay_buffer = DualReplayMemory(recent_capacity=10000, historical_capacity=20000)

    def train(self, h, v, epochs, start_epoch=0):
        H = 2 * h - 1
        W = 2 * v - 1
        in_channels = 6
        output_dim = H * W
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NN(in_channels, output_dim, H, W).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0004)
        
        model_path = f"trained_models/alphazero{h-1}x{v-1}.pt"
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded existing AlphaZero model from {model_path} to continue training.")
            except Exception:
                pass

        bot = alphazero()

        for episode in range(epochs):
            global_episode = start_epoch + episode
            H = 2 * h - 1
            W = 2 * v - 1
            in_channels = 6
            L = 2 * h * v - h - v
            sims_train = min(100, 10 * L)
            min_transitions = min(5000, 100 * L)
            
            # Generate 10 self-play games in parallel using batched MCTS
            num_games = 10
            states = [gameenv.get_initial_state((h, v)) for _ in range(num_games)]
            roots = [Node((h, v), s) for s in states]
            
            game_states = [[] for _ in range(num_games)]
            game_policies = [[] for _ in range(num_games)]
            game_finished = [False] * num_games
            
            while not all(game_finished):
                active_indices = [g for g in range(num_games) if not game_finished[g]]
                active_roots = [roots[g] for g in active_indices]
                
                # Run batched MCTS on all active games in parallel (single GPU pass per simulation step)
                bot.batched_mcts((h, v), active_roots, simulations=sims_train, model=model, training=True)
                
                for g in active_indices:
                    root = roots[g]
                    state = states[g]
                    
                    policy = np.zeros(H * W)
                    for action, child in root.children.items():
                        idx = action[0] * W + action[1]
                        policy[idx] = child.visit_count
                    policy = policy / (np.sum(policy) + 1e-8)
                    
                    game_states[g].append(state)
                    game_policies[g].append(policy)
                    
                    # Choose best action based on visit counts
                    action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
                    child = root.children[action]
                    if child.state is None:
                        child.state, _ = gameenv.step(root.state, action)
                        
                    states[g] = child.state
                    roots[g] = child
                    
                    if gameenv.is_gameover(states[g]):
                        game_finished[g] = True
                        
            # Push transitions to replay buffer once all games finish
            for g in range(num_games):
                final_state = states[g]
                p1_boxes = np.sum(final_state[3])
                p2_boxes = np.sum(final_state[4])
                final_val = 1.0 if p1_boxes > p2_boxes else (-1.0 if p2_boxes > p1_boxes else 0.0)
                
                for s, p in zip(game_states[g], game_policies[g]):
                    self.replay_buffer.push(s, p, final_val)

            required_transitions = max(256, min_transitions)
            if len(self.replay_buffer) < required_transitions:
                print(f"Buffering self-play games... ({len(self.replay_buffer)}/{required_transitions})")
                continue

            # Run 2 mini-batch gradient updates per self-play block to speed up learning without overfitting
            num_syms = symmetry.get_symmetry_count(h, v)
            num_updates = 2
            epoch_loss = 0.0
            for _ in range(num_updates):
                batch = self.replay_buffer.sample(256)
                states_batch = np.zeros((256, in_channels, H, W), dtype=np.float32)
                policies_batch = np.zeros((256, H * W), dtype=np.float32)
                for idx, (s, p, _) in enumerate(batch):
                    s_np = np.array(s, dtype=np.float32).reshape(in_channels, H, W)
                    p_np = np.array(p, dtype=np.float32).reshape(H, W)
                    
                    sym_idx = random.randint(0, num_syms - 1)
                    s_np, p_np = symmetry.transform_state_and_policy(s_np, p_np, sym_idx)
                    
                    states_batch[idx] = s_np
                    policies_batch[idx] = p_np.flatten()
                    
                X = torch.tensor(states_batch, dtype=torch.float32, device=device)
                Y_policy = torch.tensor(policies_batch, dtype=torch.float32, device=device)
                Y_value = torch.tensor([v for (_, _, v) in batch], dtype=torch.float32, device=device)
                
                model.train()
                pred_policies, pred_values = model(X)
                log_policies = torch.log_softmax(pred_policies, dim=1)
                policy_loss = nn.KLDivLoss(reduction="batchmean")(log_policies, Y_policy)
                value_loss = nn.MSELoss()(pred_values.squeeze(-1), Y_value)
                total_loss = policy_loss + value_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                
            epoch_loss = epoch_loss / num_updates
            
            print_interval = max(1, epochs // 10)
            if (episode + 1) % print_interval == 0:
                print(f"Episode {global_episode + 1}: Loss = {epoch_loss:.4f}")
                
        os.makedirs("trained_models", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
        }, model_path)
        print(f"AlphaZero model saved to {model_path}")

class Play:
    """AlphaZero player interface for Dots and Boxes."""
    def __init__(self, h, v, model_path=None):
        self.h = h
        self.v = v
        H = 2 * h - 1
        W = 2 * v - 1
        in_channels = 6
        output_dim = H * W
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NN(in_channels, output_dim, H, W).to(self.device)
        if model_path is None:
            model_path = f"trained_models/alphazero{h-1}x{v-1}_best.pt"
            if not os.path.exists(model_path):
                model_path = f"trained_models/alphazero{h-1}x{v-1}.pt"
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except Exception:
                pass

    def play(self, state, turn, time_limit=1.0):
        root = Node((self.h, self.v), state)
        bot = alphazero()
        
        # 1. Expand the root first to initialize children and their prior probabilities
        root.best_child(self.model, c_param=1.4, training=False)
        
        # 2. Add Dirichlet noise ONLY to the root's children to diversify search
        H = 2 * self.h - 1
        W = 2 * self.v - 1
        alpha = 0.6
        epsilon = 0.25  # 25% exploration noise
        noise = np.random.dirichlet([alpha] * (H * W))
        for action, child in root.children.items():
            idx = action[0] * W + action[1]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[idx]
            
        # 3. Run MCTS simulations from the perturbed root
        L = 2 * self.h * self.v - self.h - self.v
        sims_play = min(150, 10 * L)
        root = bot.mcts((self.h, self.v), root, simulations=sims_play, model=self.model, training=False)
        
        # 4. Deterministic selection of the best move
        action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return action
