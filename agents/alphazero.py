import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import env.gameenv as gameenv

class NN(nn.Module):
    """2D Convolutional Neural Network for AlphaZero."""
    def __init__(self, in_channels, output_dim, H, W):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.policy_conv = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(2 * H * W, output_dim)
        
        self.value_conv = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(H * W, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        shared = self.trunk(x)
        
        p_feat = self.policy_conv(shared)
        p_feat = p_feat.view(p_feat.size(0), -1)
        policy = self.policy_fc(p_feat)
        
        v_feat = self.value_conv(shared)
        v_feat = v_feat.view(v_feat.size(0), -1)
        value = self.value_fc(v_feat)
        
        return policy, value

class Node:
    """MCTS Node for AlphaZero."""
    def __init__(self, dots, state, parent=None, action=None):
        self.dots = dots
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = {}
        self.total_visits = 0
        self.W = {}

    def is_expanded(self):
        return len(self.children) == len(gameenv.get_action_space(self.state))

    def expand(self):
        actions = gameenv.get_action_space(self.state)
        for action in actions:
            if action not in self.children:
                new_state, reward = gameenv.step(self.state, action)
                self.W[action] = 0.0
                self.N[action] = 0.0
                child = Node(self.dots, new_state, parent=self, action=action)
                self.children[action] = child

    def best_child(self, model, c_param=1.2, training=True, show=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h, v = self.dots
        H = 2 * h - 1
        W = 2 * v - 1
        in_channels = 8 * 5
        state_np = np.array(self.state, dtype=np.float32).reshape(in_channels, H, W)
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            pred_policy, _ = model(state_tensor)
        policy = torch.softmax(pred_policy, dim=1).squeeze(0).cpu().tolist()
        
        output_dim = H * W
        if training:
            alpha = 0.3
            epsilon = 0.25
            noise = np.random.dirichlet([alpha] * output_dim)
            for i in range(output_dim):
                policy[i] = (1 - epsilon) * policy[i] + epsilon * noise[i]
                
        valid_actions = gameenv.get_action_space(self.state)
        utility = {}
        for action in valid_actions:
            action_idx = action[0] * W + action[1]
            if action not in self.N:
                self.N[action] = 0.0
                self.W[action] = 0.0
            
            Q_s_a = self.W[action] / (self.N[action] + 1e-8)
            U_s_a = c_param * policy[action_idx] * math.sqrt(self.total_visits + 1) / (1 + self.N[action])
            utility[action] = Q_s_a + U_s_a
            
        if show:
            print(f"U(s,a) : {utility}")
            
        turn = 1 if self.state[-1][0][0][0] > 0 else -1
        if turn == 1:
            best_utility = max(utility.values())
        else:
            best_utility = min(utility.values())
            
        best_actions = [a for a, u in utility.items() if u == best_utility]
        return random.choice(best_actions)

class alphazero:
    """AlphaZero training manager."""
    def __init__(self):
        self.replay_buffer = deque(maxlen=50000)
        self.completed_games = 0

    def backpropagate(self, node, reward, t=0.5, discount=1.0):
        while node:
            if node.parent:
                node.parent.W[node.action] += reward
                node.parent.N[node.action] += 1.0
            
            h, v = node.dots
            H = 2 * h - 1
            W = 2 * v - 1
            output_dim = H * W
            
            policy = [0.0] * output_dim
            node.total_visits += 1
            reward *= discount
            
            valid_actions = gameenv.get_action_space(node.state)
            numerator = []
            for action in valid_actions:
                action_idx = action[0] * W + action[1]
                if action not in node.N:
                    node.N[action] = 0.0
                    node.W[action] = 0.0
                
                if node.N[action] == 0.0:
                    numerator.append((action_idx, 1.0))
                else:
                    numerator.append((action_idx, node.N[action] ** (1.0 / t)))
                    
            denominator = sum(val for _, val in numerator)
            if denominator > 0:
                for action_idx, val in numerator:
                    policy[action_idx] = val / denominator
                    
            self.replay_buffer.append((node.state, policy, reward))
            if not node.parent:
                break
            node = node.parent
        return node

    def mcts(self, dots, root, simulations, model, training=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        node = root
        threshold = 1000 if training else 200
        movecount = 0
        i = 0
        while i < simulations:
            t = 1.0 if movecount < 4 else 0.5
            while node.children and node.is_expanded():
                min_visits = 1000.0
                min_action = None
                for action in list(node.children.keys()):
                    child_visits = node.children[action].total_visits
                    if child_visits < min_visits:
                        min_visits = child_visits
                        min_action = action
                        
                if min_visits < threshold and min_action is not None:
                    action = min_action
                else:
                    action = node.best_child(model, training=training)
                    
                node = node.children[action]
                movecount += 1
                if gameenv.is_gameover(node.state):
                    movecount = 0
                    break
                    
            if gameenv.is_gameover(node.state):
                p1_boxes = np.sum(node.state[-1][3])
                p2_boxes = np.sum(node.state[-1][4])
                value = 1.0 if p1_boxes > p2_boxes else -1.0 if p2_boxes > p1_boxes else 0.0
                node = self.backpropagate(node, value, t)
                movecount = 0
                i += 1
                self.completed_games += 1
                continue
                
            node.expand()
            
            h, v = node.dots
            H = 2 * h - 1
            W = 2 * v - 1
            in_channels = 8 * 5
            state_np = np.array(node.state, dtype=np.float32).reshape(in_channels, H, W)
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                _, predicted_value = model(state_tensor)
            value = predicted_value.item()
            node = self.backpropagate(node, value, t)
            i += 1
        return node

    def train(self, h, v, epochs, start_epoch=0):
        dots = (h, v)
        state = gameenv.get_initial_state(dots)
        H = 2 * h - 1
        W = 2 * v - 1
        in_channels = 8 * 5
        output_dim = H * W
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root = Node(dots, state)
        model = NN(in_channels, output_dim, H, W).to(device)
        
        model_path = f"trained_models/alphazero{h}x{v}.pt"
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded existing AlphaZero model from {model_path} to continue training.")
            except Exception:
                pass
                
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for episode in range(epochs):
            global_episode = start_epoch + episode
            if global_episode < 100:
                lr = 0.02
            elif global_episode < 200:
                lr = 0.002
            else:
                lr = 0.0002
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            if len(self.replay_buffer) < 5000:
                while len(self.replay_buffer) < 5000:
                    root = self.mcts(dots=dots, root=root, simulations=800, model=model)
            else:
                self.completed_games = 0
                while self.completed_games < 5:
                    root = self.mcts(dots=dots, root=root, simulations=800, model=model)
                
            batch = random.sample(self.replay_buffer, k=512)
            states_batch = np.zeros((512, in_channels, H, W), dtype=np.float32)
            for idx, (s, _, _) in enumerate(batch):
                states_batch[idx] = np.array(s, dtype=np.float32).reshape(in_channels, H, W)
                
            X = torch.tensor(states_batch, dtype=torch.float32, device=device)
            Y_policy = torch.tensor([p for (_, p, _) in batch], dtype=torch.float32, device=device)
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
            
            print_interval = max(1, epochs // 10)
            if (episode + 1) % print_interval == 0:
                print(f"Episode {global_episode + 1}: Loss = {total_loss.item():.4f}")
                
        os.makedirs("trained_models", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
        }, model_path)
        print(f"AlphaZero model saved to {model_path}")

class Play:
    """AlphaZero player interface for Dots and Boxes."""
    def __init__(self, h, v):
        self.h = h
        self.v = v
        H = 2 * h - 1
        W = 2 * v - 1
        in_channels = 8 * 5
        output_dim = H * W
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NN(in_channels, output_dim, H, W).to(self.device)
        model_path = f"trained_models/alphazero{h}x{v}.pt"
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except Exception:
                pass
        self.model.eval()

    def play(self, state, turn, time_limit=1.0):
        root = Node((self.h, self.v), state)
        bot = alphazero()
        # Run MCTS simulations guided by the neural network
        root = bot.mcts((self.h, self.v), root, simulations=800, model=self.model, training=False)
        # Select best move greedily based on Q-values
        action = root.best_child(self.model, c_param=False, training=False)
        return action
