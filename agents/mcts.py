#now monte carlo tree search
import math
import time
import pickle
import psutil, os
from ..env.env import Env
class Play:
  def forward(self,root, livenode):
    while root:
      if root.state == livenode.state or not root.children:
        return root
      for child in root.children:
        if child.state == livenode.state:
          root = child
          break

  def play(self,env,turn,secs=1):
    dots = env.dots
    with open(f"trained_models/mcts{dots}.pkl", "rb") as f:
      root = pickle.load(f)
    state = tuple(env.grid + env.boxes + [turn],turn)
    livenode = Node(state,turn)
    node = self.forward(root,livenode)
    thinker = mcts()
    thinker.think(secs,node)
    return node.best_child().action
  
class Node:
  def __init__(self,state,turn,parent=None,action=None):
    self.state = state
    self.turn = turn
    self.parent = parent
    self.action = action
    self.children = []

    self.total_reward = 0
    self.visits = 0
    
  
  def is_expanded(self):
    env = Env.from_state(self.state)
    return (len(env.action_space()) == len(self.children))
  
  def expand(self):
    env = Env.from_state(self.state)
    for action in env.action_space():
      new_env = env.clone()
      reward = new_env.step(action=action,turn=self.turn)
      if reward == 0:
        new_turn = -self.turn
      else:
        new_turn = self.turn
      new_state = tuple(new_env.grid + new_env.boxes + [new_turn])
      new_node = Node(new_state,new_turn,self,action)
      self.children.append(new_node)

  def best_child(self,c_param=1.4):
    values = [
            (child.total_reward / (child.visits + 1e-8)) + 
            c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-8))
            for child in self.children
        ]
    
    if self.turn == 1:
      return self.children[values.index(max(values))]
    elif self.turn == -1:
      return self.children[values.index(min(values))]
    return None

class mcts:
  def __init__(self):
    self.root = []

  def train(self,dots,epochs):
    env = Env(dots)
    turn = 1
    self.root = Node(tuple(env.grid + env.boxes +[turn]),turn)
    for episode in range(epochs):  
      node = self.root
      #reward = node.total_reward
      gameover = Env.Gameover(node.state)
      while not gameover:
        while node.children:
          child = node.best_child()
          node = child
          gameover = Env.Gameover(node.state)

        if gameover:
          break

        if not node.children:
          node.expand()

      total_reward = sum(Env.from_state(node.state).boxes)
      reward = 0
      if total_reward > 0:
        reward = 1 
      elif total_reward < 0:
        reward = -1 
      else :
        reward = 0
      self.backpropagate(node, reward)
     
      if episode %5000 == 0:
        print(f"{episode} epochs done")

      if episode % 5000 == 0:
        print(f"Memory used: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")
    
    print("training done")
    with open(f"trained_models/mcts{dots}.pkl", "wb") as f:
      pickle.dump(self.root,f)
    print(f"model saved at trained_models/mcts{dots}.pkl")
    


  def backpropagate(self,node,reward):
    while(node):
      node.visits += 1
      node.total_reward += reward
      node = node.parent
      
  def think(self,secs,root):
    start_time = time.time()
    end_time = time.time()
    node = root
    while(end_time - start_time < secs):
      node = root
      #reward = node.total_reward
      gameover = Env.Gameover(node.state)
      while not gameover:
        while node.children:
          #print(len(node.children))
          child = node.best_child()
          node = child
          gameover = Env.Gameover(node.state)
        
        if gameover:
          break

        if not node.children:
          node.expand()

      total_reward = sum(Env.from_state(node.state).boxes)
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

  


