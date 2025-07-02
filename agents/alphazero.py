import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import math
import random
import time
from collections import deque

from env.env import Env
class NN(nn.Module):  # Inherit properly
  def __init__(self, input_dim,output_dim):
        super(NN, self).__init__()  # Correct super call
        self.model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.ReLU()
        )
        self.policy_head = nn.Linear(128, output_dim)
        self.value_head = nn.Linear(128, 1)

  def predict(self,input):
    network = self.model(input)
    return self.policy_head(network),self.value_head(network)

class Node:
  def __init__(self,dots,state,parent=None,action=None):
    self.dots = dots
    self.state = state
    self.parent = parent
    self.action = action
    self.children = {}
    self.visits = {}
    self.total_visits = 0
    self.value = {}
    

  def is_expanded(self):
    return (len(self.children) == len(Env.from_state(self.dots,self.state).action_space()))

  def expand(self):
    child_turn = self.state[-1]
    for action in Env.from_state(self.dots,self.state).action_space():
      self.visits[action] = 0
      new_env = Env.from_state(self.dots,self.state)
      reward = new_env.step(action,child_turn)
      new_turn = 0
      if reward == 0:
        new_turn = -child_turn
      else :
        new_turn = child_turn
      self.value[action] = reward
      state = tuple(new_env.grid + new_env.boxes +[new_turn])
      child = Node(self.dots,state,parent=self,action=action)
      self.children[action]=child

  def best_child(self,model,c_param=1.4,training=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy,_ = model.predict(torch.tensor(self.state, dtype=torch.float32, device=device).unsqueeze(0))
    policy = policy.squeeze(0)
    policy = policy.tolist()
    #print(policy)
    #policy = policy.detach()
    grid = 2 *self.dots*(self.dots-1)
    if training == True:
      alpha = 0.3
      epsilon = 0.25
      
      noise = np.random.dirichlet([alpha] * grid)
      for i in range(grid):
        policy[i] = (1 - epsilon) * policy[i] + epsilon * noise[i]

    utility = [0]*(grid)
    for action in range(grid):
      if action not in self.visits.keys():
        utility[action] = 0
      else:
        Q_s_a = self.value[action]
        utility[action] = Q_s_a + c_param*policy[action]*math.sqrt(self.total_visits)/(1+self.visits[action])
    valid_actions = Env.from_state(self.dots,self.state).action_space()
    best = random.choice(valid_actions)
    #print(utility)
    turn = self.state[-1]
    if turn == 1:
      max = utility[best]
      for action in valid_actions:
        if utility[action] > max:
          max = utility[action]
          best = action
    elif turn == -1:
      min = utility[best]
      for action in valid_actions:
        if utility[action] < min:
          min = utility[action]
          best = action
    self.visits[best] += 1
    #print(len(self.children))
    #print(valid_actions)
    #print(best)
    return self.children[best]



class alphazero:
  def __init__(self):
    self.replay_buffer = deque(maxlen=100000)  # stores up to 100k positions
    

  def backpropagate(self,node,reward,t=0.5,discount=0.995):
    
    while node:
      policy = [0]*2*self.dots*(self.dots-1)
      value = reward

      node.total_visits += 1
      if node:
        node.value[node.action] += reward
        reward=discount
        valid_actions = Env.from_state(self.dots,node.state).action_space()
        for action in valid_actions:
          z = sum((node.visits).values())
          if z == 0 :
            policy[action] = 0
          else:
            policy[action] = node.visits[action]**(1/t) / (z**(1/t))
        self.replay_buffer.append((node.state, policy, value))
        node = node.parent

  def mcts(self,dots,simulations,model,training=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env(dots)
    turn = 1
    root = Node(dots,tuple(env.grid + env.boxes +[turn]),turn) 
    for _ in range(simulations):  
      node = root
      # Selection
      while node.children and node.is_expanded():
          node = node.best_child(model,training)
          if Env.Gameover(self.dots,node.state):
              break

      # Terminal node
      if Env.Gameover(self.dots,node.state):
          reward = sum(Env.from_state(self.dots,node.state).boxes)
          value = 1 if reward > 0 else -1 if reward < 0 else 0
          self.backpropagate(node, value)
          continue

      # Expansion
      node.expand()

      # Evaluation
      state_tensor = torch.tensor(node.state, dtype=torch.float32, device=device).unsqueeze(0)
      with torch.no_grad():
          _, predicted_value = model.predict(state_tensor)
      value = predicted_value.item()
      self.backpropagate(node, value)


  def train(self,dots,epochs=1000,n=500):
    env = Env(dots)
    state = tuple(env.grid + env.boxes+[1])
    input_dim = len(state)
    output_dim = len(env.grid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NN(input_dim,output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint = torch.load(f"trained_models/alphazero{dots}.pt", map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for episode in range(epochs):
      #print(episode)
      if episode % 500 == 0:
        print("{episode} episodes done, saving checkpoint in trained_models/")
        torch.save({
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
          }, f"trained_models/alphazero{dots}_checkpoint_epoch_{episode}.pt")
  
      #play n games to collect training data for nn
      self.mcts(dots=dots,simulations=n)
      #take random sample games for training
      batch = random.sample(self.replay_buffer, k=500)  
      X = torch.tensor([s for (s, _, _) in batch], ...)
      Y_policy = torch.tensor([p for (_, p, _) in batch], ...)
      Y_value = torch.tensor([v for (_, _, v) in batch], ...)

      pred_policies,pred_values = model.predict(X)
      log_policies = torch.log_softmax(pred_policies, dim=1)   # [B, 24]
      policy_loss = nn.KLDivLoss(reduction="batchmean")(log_policies, Y_policy)
      value_loss = nn.MSELoss()(pred_values.squeeze(), Y_value)

      total_loss = policy_loss + value_loss
      if (episode %100 == 0):
        policy_probs_batch = torch.softmax(pred_policies, dim=1)
        print(f"Episode {episode}: Loss = {total_loss.item():.4f}")
        entropy = - (policy_probs_batch * policy_probs_batch.log()).sum(dim=1).mean().item()
        print(f"Policy entropy: {entropy:.4f}")
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
    print("training done")
    torch.save({
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
          }, f"trained_models/alphazero{dots}.pt")
    #saving optimizer for continuous training
    print(f"Model saved at trained_models/alphazero{dots}.pt")


  def think(self,secs,root,model):
    start_time = time.time()
    end_time = time.time()
    node = root
    while(end_time - start_time < secs):
      node = root
      #reward = node.total_reward
      gameover = Env.Gameover(self.dots,node.state)
      while not gameover:
        while node.children:
          #print(len(node.children))
          child = node.best_child(model,training=False)
          node = child
          gameover = Env.Gameover(self.dots,node.state)

        if gameover:
          break

        if not node.children:
          node.expand()

      total_reward = sum(Env.from_state(self.dots,node.state).boxes)
      reward = 0
      if total_reward > 0:
        reward = 1
        self.win += 1
      elif total_reward < 0:
        reward = -1
        self.loss += 1
      else :
        self.draw += 1
        reward = 0
      self.backpropagate(node, reward)
      end_time = time.time()


class Play:
  def play(self,env,turn,secs=1):
    dots = env.dots
    state = tuple(env.grid + env.boxes+[1])
    input_dim = len(state)
    output_dim = len(env.grid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(input_dim,output_dim).to(device)   
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    try:
      checkpoint = torch.load(f"trained_models/alphazero{dots}.pt", map_location=torch.device(device))
      model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
      print("Please train the model first, playing randomnly now")
    
    state = tuple(env.grid + env.boxes + [turn],turn)
    livenode = Node(dots,state,turn)
    thinker = alphazero()
    thinker.think(secs,livenode,model)
    return livenode.best_child().action