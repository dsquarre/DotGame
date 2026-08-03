import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import env.gameenv as gameenv
import agents.symmetry as symmetry

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
    def __init__(self, recent_capacity=5000, historical_capacity=15000, push_history_prob=0.10):
        self.recent = RecentBuffer(recent_capacity)
        self.historical = HistoricalBuffer(historical_capacity)
        self.push_history_prob = push_history_prob

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
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

class NN(nn.Module):
    """Deep Q-Network model for Dots and Boxes."""
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
        self.fc = nn.Sequential(
            nn.Linear(128 * H * W, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

class dqn:
    """Deep Q-Network training manager."""
    def train(self, h, v, epochs, start_epoch=0):
        H = 2 * h - 1
        W = 2 * v - 1
        in_channels = 6
        output_dim = H * W
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NN(in_channels, output_dim, H, W).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        
        model_path = f"trained_models/dqn{h-1}x{v-1}.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                print(f"Loaded existing DQN model from {model_path} to continue training.")
            except Exception:
                pass

        memory = DualReplayMemory(recent_capacity=5000, historical_capacity=15000)
        discount_factor = 0.99

        for episode in range(epochs):
            global_episode = start_epoch + episode
            epsilon = max(0.01, 0.995**global_episode)

            state = gameenv.get_initial_state((h, v))
            turn = 1

            while not gameenv.is_gameover(state):
                state_np = np.array(state, dtype=np.float32).reshape(in_channels, H, W)
                state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    q_state = model(state_tensor).squeeze()
                valid_actions = gameenv.get_action_space(state)

                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    valid_indices = [r * W + c for r, c in valid_actions]
                    valid_q_values = torch.tensor([q_state[idx] for idx in valid_indices], device=device)
                    if turn == 1:
                        best_idx = torch.argmax(valid_q_values).item()
                    else:
                        best_idx = torch.argmin(valid_q_values).item()
                    action = valid_actions[best_idx]

                next_state, reward = gameenv.step(state, action)
                done = gameenv.is_gameover(next_state)

                memory.push(state, action, reward, next_state, done)

                if len(memory) >= 64:
                    batch = memory.sample(64)
                    states_batch = np.zeros((64, in_channels, H, W), dtype=np.float32)
                    next_states_batch = np.zeros((64, in_channels, H, W), dtype=np.float32)

                    num_syms = symmetry.get_symmetry_count(h, v)
                    batch_sym_indices = []
                    for idx, (s, a, r, ns, d) in enumerate(batch):
                        s_np = np.array(s, dtype=np.float32).reshape(in_channels, H, W)
                        ns_np = np.array(ns, dtype=np.float32).reshape(in_channels, H, W)
                        
                        sym_idx = random.randint(0, num_syms - 1)
                        batch_sym_indices.append(sym_idx)
                        
                        s_np, _ = symmetry.transform_state_and_policy(s_np, None, sym_idx)
                        ns_np, _ = symmetry.transform_state_and_policy(ns_np, None, sym_idx)
                        
                        states_batch[idx] = s_np
                        next_states_batch[idx] = ns_np

                    s_tensor = torch.tensor(states_batch, dtype=torch.float32, device=device)
                    ns_tensor = torch.tensor(next_states_batch, dtype=torch.float32, device=device)

                    q_states = model(s_tensor)
                    with torch.no_grad():
                        q_next_states = model(ns_tensor)

                    loss = 0.0
                    for idx, (s, a, r, ns, d) in enumerate(batch):
                        sym_idx = batch_sym_indices[idx]
                        aug_a = symmetry.transform_action(a, sym_idx, H, W)
                        act_idx = aug_a[0] * W + aug_a[1]
                        q_val = q_states[idx, act_idx]

                        if d:
                            target = r
                        else:
                            valid_actions_next = gameenv.get_action_space(ns)
                            if not valid_actions_next:
                                target = r
                            else:
                                valid_indices_next = [symmetry.transform_action(nr_nc, sym_idx, H, W) for nr_nc in valid_actions_next]
                                valid_indices_flat = [nr * W + nc for nr, nc in valid_indices_next]
                                q_next_vals = q_next_states[idx, valid_indices_flat]

                                next_turn = 1 if ns[0][0][0] > 0 else -1
                                if next_turn == 1:
                                    best_q_next = torch.max(q_next_vals)
                                else:
                                    best_q_next = torch.min(q_next_vals)

                                target = r + discount_factor * best_q_next

                        loss += (target - q_val) ** 2

                    loss = loss / 64
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                state = next_state
                turn = 1 if state[0][0][0] > 0 else -1

            print_interval = max(1, epochs // 10)
            if (episode + 1) % print_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"{episode + 1}/{epochs} epochs done (epsilon: {epsilon:.4f}, lr: {current_lr:.6f})")

        print("training done")
        os.makedirs("trained_models", exist_ok=True)
        torch.save(model.state_dict(), f"trained_models/dqn{h-1}x{v-1}.pth")
        print(f"model saved at trained_models/dqn{h-1}x{v-1}.pth")

class Play:
    """Deep Q-Network player interface for Dots and Boxes."""
    def __init__(self, h, v, model_path=None):
        self.h = h
        self.v = v
        H = 2 * self.h - 1
        W = 2 * self.v - 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = 6
        output_dim = H * W
        self.model = NN(in_channels, output_dim, H, W).to(self.device)
        try:
            if model_path is None:
                model_path = f"trained_models/dqn{self.h-1}x{self.v-1}_best.pth"
                if not os.path.exists(model_path):
                    model_path = f"trained_models/dqn{self.h-1}x{self.v-1}.pth"
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except Exception:
            pass

    def play(self, state, turn, secs=0):
        H = len(state[0])
        W = len(state[0][0])
        in_channels = 6
        actions = gameenv.get_action_space(state)

        state_np = np.array(state, dtype=np.float32).reshape(in_channels, H, W)
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()
            valid_indices = [r * W + c for r, c in actions]
            valid_q_values = torch.tensor([q_values[idx] for idx in valid_indices], device=self.device)

            import random
            best_q = torch.max(valid_q_values).item() if turn == 1 else torch.min(valid_q_values).item()
            
            # Select all actions within a safe 0.1 threshold of the best action
            valid_q_list = valid_q_values.cpu().numpy()
            if turn == 1:
                safe_indices = [i for i, q in enumerate(valid_q_list) if q >= best_q - 0.1]
            else:
                safe_indices = [i for i, q in enumerate(valid_q_list) if q <= best_q + 0.1]
                
            best_idx = random.choice(safe_indices)
            return actions[best_idx]
