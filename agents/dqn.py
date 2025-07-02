##initializing neural network

import random
import torch
import torch.nn as nn
from ..env.env import Env
class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.model(x)
##training deepQ with self play

class dqn:
  def train(self,dots,epochs):
    discount_factor = 0.65
    env = Env(dots)
    state = tuple(env.grid + env.boxes+[1])
    input_dim = len(state)
    output_dim = len(env.grid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for episode in range(epochs):
      turn = 1
      reward = 0
      best_q_next = 0
      q_state_action = 0
      while(not env.gameover()):
        state = tuple(env.grid + env.boxes+[turn])
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_state = model.forward(x=state_tensor).squeeze()
        valid_actions = env.action_space()
        epsilon = max(0.01,0.1*(0.995**(episode)))  # Exploration rate
        q_state_action = 0
        if random.random() < epsilon:
            action = random.choice(valid_actions) # Explore
            q_state_action = q_state[action]
        else: #Exploit
          valid_q_values = torch.tensor([q_state[a] for a in valid_actions],device=device)
          if turn == 1:
            action = valid_actions[torch.argmax(valid_q_values)] # Exploit
            q_state_action = q_state[action]
          else:
            action = valid_actions[torch.argmin(valid_q_values)] # Exploit
            q_state_action = q_state[action]

        reward= float(env.step(action,turn))/len(env.boxes)
        if(reward == 0):
          turn = -turn
        if(env.gameover()):
           reward = sum(env.boxes)
           best_q_next = reward
        else :   
          next_state = tuple(env.grid + env.boxes+[turn])
          next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
          q_next_state = model.forward(x=next_state_tensor).squeeze()
          valid_actions_next = env.action_space()
          best_q_next = 0
          if valid_actions_next:
            valid_q_values = torch.tensor([q_next_state[a] for a in valid_actions_next],device=device)
            if turn == 1:
              best_q_next = torch.max(valid_q_values)
            else:
              best_q_next = torch.min(valid_q_values)
        loss = (reward + discount_factor * best_q_next - q_state_action)**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
      env.reset()
      if(episode%10000 == 0):
        print("{episode} epochs done")
    print("training done")
    torch.save(model.state_dict(), f"dqn{dots}.pth")
    print(f"model saved at trained_models/dqn{dots}.pth")
    

class Play:
   def play(self,env,turn,secs=0):
      dots = env.dots
      state = tuple(env.grid + env.boxes + [turn])
      input_dim = len(state)
      output_dim = len(env.grid)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model = NN(input_dim, output_dim).to(torch.device(device))
      model.load_state_dict(torch.load(f"trained_models/dqn{dots}.pth", map_location=torch.device(device)))
      state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
      with torch.no_grad():
        q_values = model(state_tensor).squeeze()
        valid_actions = env.action_space()
        valid_q_values = torch.tensor([q_values[a] for a in valid_actions],device=device)
        if turn == 1:
          action = valid_actions[torch.argmax(valid_q_values).item()]
        else:
          action = valid_actions[torch.argmin(valid_q_values).item()]

      return action
