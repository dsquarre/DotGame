import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import env.gameenv as gameenv

class NN(nn.Module):
    """2D Convolutional Neural Network with MaxPool layer for spatial feature extraction."""
    def __init__(self, in_channels, output_dim, H, W):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            dummy_out = self.conv(dummy)
            self.flat_dim = dummy_out.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

class ReplayBuffer:
    """Experience Replay Buffer for storing transitions."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class dqn:
    """Deep Q-Network reinforcement learning agent training wrapper."""
    def train(self, h, v, epochs, start_epoch=0):
        discount_factor = 0.65
        H = 2 * h - 1
        W = 2 * v - 1
        in_channels = 8 * 5
        output_dim = H * W
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NN(in_channels, output_dim, H, W).to(device)
        model_path = f"trained_models/dqn{h}x{v}.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                print(f"Loaded existing DQN model from {model_path} to continue training.")
            except Exception:
                pass
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        memory = ReplayBuffer(capacity=10000)

        for episode in range(epochs):
            global_episode = start_epoch + episode
            epsilon = max(0.01, 0.995**global_episode)
            lr = max(0.0001, 0.001 * (0.995**global_episode))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

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

                    for idx, (s, a, r, ns, d) in enumerate(batch):
                        states_batch[idx] = np.array(s, dtype=np.float32).reshape(in_channels, H, W)
                        next_states_batch[idx] = np.array(ns, dtype=np.float32).reshape(in_channels, H, W)

                    s_tensor = torch.tensor(states_batch, dtype=torch.float32, device=device)
                    ns_tensor = torch.tensor(next_states_batch, dtype=torch.float32, device=device)

                    q_states = model(s_tensor)
                    with torch.no_grad():
                        q_next_states = model(ns_tensor)

                    loss = 0.0
                    for idx, (s, a, r, ns, d) in enumerate(batch):
                        act_idx = a[0] * W + a[1]
                        q_val = q_states[idx, act_idx]

                        if d:
                            target = r
                        else:
                            valid_actions_next = gameenv.get_action_space(ns)
                            if not valid_actions_next:
                                target = r
                            else:
                                valid_indices_next = [nr * W + nc for nr, nc in valid_actions_next]
                                q_next_vals = q_next_states[idx, valid_indices_next]

                                next_turn = 1 if ns[-1][0][0][0] > 0 else -1
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
                turn = 1 if state[-1][0][0][0] > 0 else -1

            print_interval = max(1, epochs // 10)
            if (episode + 1) % print_interval == 0:
                print(f"{episode + 1}/{epochs} epochs done (epsilon: {epsilon:.4f}, lr: {lr:.6f})")

        print("training done")
        os.makedirs("trained_models", exist_ok=True)
        torch.save(model.state_dict(), f"trained_models/dqn{h}x{v}.pth")
        print(f"model saved at trained_models/dqn{h}x{v}.pth")

class Play:
    """Deep Q-Network player interface for Dots and Boxes."""
    def __init__(self, h, v):
        self.h = h
        self.v = v
        H = 2 * self.h - 1
        W = 2 * self.v - 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = 8 * 5
        output_dim = H * W
        self.model = NN(in_channels, output_dim, H, W).to(self.device)
        try:
            self.model.load_state_dict(torch.load(f"trained_models/dqn{self.h}x{self.v}.pth", map_location=self.device, weights_only=True))
        except Exception:
            pass

    def play(self, state, turn, secs=0):
        H = len(state[0][0])
        W = len(state[0][0][0])
        in_channels = 8 * 5
        actions = gameenv.get_action_space(state)

        state_np = np.array(state, dtype=np.float32).reshape(in_channels, H, W)
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()
            valid_indices = [r * W + c for r, c in actions]
            valid_q_values = torch.tensor([q_values[idx] for idx in valid_indices], device=self.device)

            if turn == 1:
                best_idx = torch.argmax(valid_q_values).item()
            else:
                best_idx = torch.argmin(valid_q_values).item()

            return actions[best_idx]
