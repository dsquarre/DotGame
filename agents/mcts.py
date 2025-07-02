#now monte carlo tree search
import math
import time
import pickle
import psutil, os
from env.env import Env
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
    #print('recieved this')
    state = tuple(env.grid + env.boxes + [turn])
    #print(env.action_space())
    livenode = Node(env.dots,state,turn)
    '''try:
      with open(f"trained_models/mcts{dots}.pkl","r") as f:
        livenode = pickle.load(f)
        livenode
    except FileNotFoundError:
        print("Please train the model first, playing randomnly now")'''
    thinker = mcts()
    livenode.expand()
    #print(len(livenode.children))
    livenode = thinker.think(env.dots,secs,livenode)
    #print(len(livenode.children))
    return livenode.best_child().action
  
class Node:
  def __init__(self,dots,state,turn,parent=None,action=None):
    self.state = state
    self.dots = dots
    self.turn = turn
    self.parent = parent
    self.action = action
    self.children = []

    self.total_reward = 0
    self.visits = 0
    
  
  def is_expanded(self):
    env = Env.from_state(self.dots,self.state)
    return (len(env.action_space()) == len(self.children))
  
  def expand(self,flag=False):
    env = Env.from_state(self.dots,self.state)
    if flag: 
      print(self.dots)
      env.render()
    #print(len(env.action_space()))
    for action in env.action_space():
      new_env = env.clone()
      reward = new_env.step(action=action,turn=self.turn)/len(env.boxes)
      if reward == 0:
        new_turn = -self.turn
      else:
        new_turn = self.turn
      new_state = tuple(new_env.grid + new_env.boxes + [new_turn])
      new_node = Node(self.dots,new_state,new_turn,self,action)
      new_node.total_reward = reward
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
    self.root = Node(dots,tuple(env.grid + env.boxes +[turn]),turn)
    for episode in range(epochs):  
      node = self.root
      #reward = node.total_reward
      gameover = Env.Gameover(dots,node.state)
      while not gameover:
        while node.children:
          child = node.best_child()
          node = child
          gameover = Env.Gameover(dots,node.state)

        if gameover:
          break

        if not node.children:
          node.expand()

      total_reward = sum(Env.from_state(dots,node.state).boxes)
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
    
  def simulate(self,env,turn):
    import random
    while not env.gameover():
      action = random.choice(env.action_space())
      value = env.step(action,turn)
      if value == 0:
        turn *=-1
    reward = sum(env.boxes)
    return 1 if reward > 0 else -1 if reward < 0 else 0
  
  def backpropagate(self,node,reward,discount = 0.95):
    while(node):
      node.visits += 1
      node.total_reward += reward
      reward *= discount
      if not node.parent:
        break
      node = node.parent
      
  def think(self,dots,secs,root):
    start_time = time.time()
    end_time = time.time()
    node = root
    #node.expand()
    while(end_time - start_time < secs):
      node = root
      #reward = node.total_reward
      gameover = Env.Gameover(dots,node.state)
      while not gameover:
        while node.children:
          #print(len(node.children))
          child = node.best_child()
          node = child
          gameover = Env.Gameover(dots,node.state)
        
        if gameover:
          break

        if not node.children:
          node.expand()
          value = self.simulate(Env.from_state(dots,node.state),node.state[-1])
          node.total_reward += value
          

      total_reward = sum(Env.from_state(dots,node.state).boxes)
      reward = 0
      if total_reward > 0:
        reward = 1 
      elif total_reward < 0:
        reward = -1 
      else :
        reward = 0
      self.backpropagate(node, reward)
      #env.reset()
      end_time = time.time()
    return root

  


