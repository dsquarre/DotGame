import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import math
import random
import time
from collections import deque

#from env.env import Env
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

  def forward(self,x):
    network = self.model(x)
    return self.policy_head(network),self.value_head(network)

class Node:
  def __init__(self,dots,state,parent=None,action=None):
    self.dots = dots
    self.state = state
    self.parent = parent
    self.action = action
    self.children = {}
    self.N = {}
    self.total_visits = 0
    self.W = {}


  def is_expanded(self):
    return (len(self.children) == len(Env.from_state(self.dots,self.state).action_space()))

  def expand(self):
    turn = self.state[-1]
    env = Env.from_state(self.dots,self.state)
    for action in env.action_space():
      new_env = env.clone()
      reward = new_env.step(action,turn)
      new_turn = 0
      if reward == 0:
        new_turn = -turn
      else :
        new_turn = turn
      self.W[action] = 0
      self.N[action] = 0
      state = tuple(new_env.grid + new_env.boxes +[new_turn])
      child = Node(self.dots,state,parent=self,action=action)
      self.children[action]=child

  def best_child(self,model,c_param=1.2,training=True,show=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_policy,_ = model(torch.tensor(self.state, dtype=torch.float32, device=device).unsqueeze(0))
    policy = torch.softmax(pred_policy,dim=1).squeeze(0)
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
      if action not in self.N.keys():
        utility[action] = c_param*policy[action]
      else:
        Q_s_a = self.W[action]/(self.N[action]+1e-8)
        U_s_a = c_param*policy[action]*math.sqrt(self.total_visits)/(1+self.N[action])
        utility[action] = Q_s_a + U_s_a
    valid_actions = Env.from_state(self.dots,self.state).action_space()
    best = random.choice(valid_actions)
    if show: print(f"U(s,a) : {utility}")
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
    #self.N[best] += 1
    #print(len(self.children))
    #print(valid_actions)
    #print(best)
    return best



class alphazero:
  def __init__(self):
    self.replay_buffer = deque(maxlen=1000000)


  def backpropagate(self,node,reward,t=0.5,discount=1):
    while node:
      if node.parent:
        #print(f"current state {node.state} parent action : {node.action} parent rewards : {node.parent.W} parent visits : {node.parent.W}")
        node.parent.W[node.action] += reward
        node.parent.N[node.action] += 1
      policy = [0]*2*node.dots*(node.dots-1)
      node.total_visits += 1
      reward *=discount
      valid_actions = Env.from_state(node.dots,node.state).action_space()
      numerator = []
      for i in range(2 * node.dots * (node.dots - 1)):
          if i in valid_actions:
              if node.N[i] == 0:
                  numerator.append(1.0)  # Encourage unexplored actions
              else:
                  numerator.append(node.N[i] ** (1 / t))
          else:
              numerator.append(0.0)

      denominator = sum(numerator)
      policy = [x / denominator if i in valid_actions else 0.0
                for i, x in enumerate(numerator)]

      self.replay_buffer.append((node.state, policy, reward))
      if not node.parent:
        break

      node = node.parent
    return node

  def show(self, node, depth=0):
      indent = "  " * depth
      print(f"{indent}visits: {node.N}, total reward: {node.W}")

      for action, child in node.children.items():
          #expected_reward = child.total_reward / (child.visits + 1e-8)
          print(f"{indent}├── Action {action} → visits: {node.N}, total reward: {node.W}")
          self.show(child, depth + 1)

      if not node.children:
          print(f"{indent}└── [Leaf node]")

  def mcts(self,dots,root,simulations,model,training=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node = root
    #print(f"recieved this {node.state}")
    #self.show(node)
    threshold = 1000 if training else 200
    movecount = 0
    i=0
    while(i < simulations):
      #print(f"simulation {_}")
      #print(f"current tree ")
      #self.show(node)
      # Selection
      t = 1 if movecount < 4 else 0.5
      while node.children and node.is_expanded():
          min_visits = 1000
          min_action = -9
          for action in list(node.children.keys()):
            if node.children[action].total_visits < min_visits:
              min_visits = node.children[action].total_visits
              min_action = action
          if min_visits < threshold:
            action = min_action
          else:
            action = node.best_child(model,training)
          #print(f"choosing best child {action}")
          node = node.children[action]
          movecount += 1
          if Env.Gameover(dots,node.state):
              movecount = 0
              break

      # Terminal node
      if Env.Gameover(dots,node.state):
          reward = sum(Env.from_state(dots,node.state).boxes)
          value = 1 if reward > 0 else -1 if reward < 0 else 0
          #print(f"backpropagating this {node.state} {value}")
          node = self.backpropagate(node, value,t)
          movecount = 0
          i+=1
          #print(f"got this {node.state}")
          continue

      # Expansion
      node.expand()
      #print("expanded node")
      #print(f"node visits {node.N} node w {node.W}")
      # Evaluation
      state_tensor = torch.tensor(node.state, dtype=torch.float32, device=device).unsqueeze(0)
      _, predicted_value = model(state_tensor)
      value = predicted_value.item()
      #print(f"backpropagating this {node.state} {value}")
      node = self.backpropagate(node, value,t)
      i+=1
    #print(i)
    
      #print(f"got this {node.state}")
    #print(f"returning this {node.state}")
    #self.show(node)
    return node

  def train(self,dots,epochs,n):
    env = Env(dots)
    state = tuple(env.grid + env.boxes+[1])
    input_dim = len(state)
    output_dim = len(env.grid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Node(dots,state)
    model = NN(input_dim,output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for episode in range(epochs):
      #print(episode)
      if episode < 100:
        lr = 0.1
      elif episode < 200:
        lr = 0.02
      elif episode < 300:
        lr = 0.002
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
      if episode % 100 == 0 and not episode == 0:
        print(f"{episode} episodes done, saving checkpoint in trained_models/")
        torch.save({
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
          }, f"alphazero{dots}.pt")
        from google.colab import files
        files.download(f'alphazero{dots}.pt')
        greedy = Minmax()
        #bot.train(dots=3,epochs=500,n=10000)
        bot = Play()
        env = Env(dots)
        turn = 1
        win = 0
        loss = 0
        draw = 0
        for _ in range(10):
          while not env.gameover():
            if turn == 1:
              action = bot.play(env,turn)
            else:
              action = greedy.play(env,turn)
            v = env.step(action,turn)
            #env.render()
            if v == 0:
              turn = -turn
          if sum(env.boxes) > 0: win+=1
          elif sum(env.boxes) <0: loss+=1
          else: draw += 1
          env.reset()
          turn = 1
        plotter = Plot()
        plotter.plot("alphazero","minmax",win,draw,loss,3,title=f"epoch {episode}")
        

      #play n games to collect training data for nn
      
      root = self.mcts(dots=dots,root=root,simulations=n,model=model)
      while(len(self.replay_buffer) < 50000):
          root = self.mcts(dots=dots,root=root,simulations=n,model=model)
        #print("simulations done")
      #take random sample games for training
      batch = random.sample(self.replay_buffer, k=50000)
      X = torch.tensor([s for (s, _, _) in batch], dtype=torch.float32, device=device)
      Y_policy = torch.tensor([p for (_, p, _) in batch], dtype=torch.float32, device=device)
      Y_value = torch.tensor([v for (_, _, v) in batch], dtype=torch.float32, device=device)
      model.train()
      pred_policies, pred_values = model(X)
      log_policies = torch.log_softmax(pred_policies, dim=1)   # [B, 24]
      policy_loss = nn.KLDivLoss(reduction="batchmean")(log_policies, Y_policy)
      value_loss = nn.MSELoss()(pred_values.squeeze(), Y_value)
      total_loss = policy_loss + value_loss
      #print(f"policy requires grad: {pred_policies.requires_grad}")
      #print(f"value requires grad: {pred_values.requires_grad}")
      for name, param in model.named_parameters():
        if param.grad is not None and param.grad.isnan().any():
          print(f"NaN in gradients: {name}")
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
      if (episode %10 == 0):
        policy_probs_batch = torch.softmax(pred_policies, dim=1)
        print(f"Episode {episode}: Loss = {total_loss.item():.4f}")
        entropy = - (policy_probs_batch * (policy_probs_batch+1e-8).log()).sum(dim=1).mean().item()
        print(f"Policy entropy: {entropy:.4f}")
    print("training done")
    torch.save({
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
          }, f"alphazero{dots}.pt")
    #saving optimizer for continuous training
    print("DONE")
    print(f"Model saved at alphazero{dots}.pt")


class Play:
  def play(self,env,turn,sims=10000):
    dots = env.dots
    state = tuple(env.grid + env.boxes+[1])
    input_dim = len(state)
    output_dim = len(env.grid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(input_dim,output_dim).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    try:
      checkpoint = torch.load(f"alphazero{dots}.pt")
      model.load_state_dict(checkpoint['model_state_dict'])
      print("model loaded")
    except FileNotFoundError:
      print("Please train the model first, playing randomnly now")

    state = tuple(env.grid + env.boxes + [turn])
    livenode = Node(dots,state)
    bot = alphazero()
    #livenode = bot.mcts(dots,livenode,sims,model,False)

    logits,value= model(torch.tensor(livenode.state, dtype=torch.float32, device=device).unsqueeze(0))
    policy = torch.softmax(logits,dim=1)
    #print(f"model predicited policy = {policy} \nvalue = {value}")
    action = livenode.best_child(model,False,show=False)

    #print(f"choosing action {action}")
    #bot.show(livenode)
    return action

