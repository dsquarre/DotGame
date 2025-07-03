#now monte carlo tree search
import math
import time
import pickle
import random
from env.env import Env
class Play:
  def play(self,env,turn,sims=5):
    #print('recieved this')
    dots = env.dots
    state = tuple(env.grid + env.boxes + [turn])
    #print(env.action_space())
    livenode = Node(env.dots,state,turn)
    root = livenode    
    thinker = mcts()
    positions = {}
    try:
      with open(f"trained_models/mcts{dots}.pkl","rb") as f:
        positions = pickle.load(f)
        print("model loaded")
        root = positions[state]
        if root is not None:
          livenode = root
          #print(livenode.state)
        #thinker.show(livenode)    
    except Exception as e:
      print(e)
      print("Train model, playing online now")
    #livenode.expand()
    #print(len(livenode.children))
    action = thinker.think(env.dots,livenode,positions,sims)
    #print(len(livenode.children))
    return action
  
class Node:
  def __init__(self,dots,state,turn,parent=None):
    self.state = state
    self.dots = dots
    self.turn = turn
    self.parent = parent
    self.children = {}
    self.total_reward = 0
    self.visits = 0
    
  
  def is_expanded(self):
    env = Env.from_state(self.dots,self.state)
    return (len(env.action_space()) == len(self.children))
  
  def expand(self,positions,flag=False):
    env = Env.from_state(self.dots,self.state)
    if flag: 
      print(env.action_space())
      env.render()
    #print(len(env.action_space()))
    for action in env.action_space():
      new_env = env.clone()
      reward = new_env.step(action=action,turn=self.turn)
      if reward == 0:
        new_turn = -self.turn
      else:
        new_turn = self.turn
      new_state = tuple(new_env.grid + new_env.boxes + [new_turn])
      new_node = Node(self.dots,new_state,new_turn,self)
      self.children[action] = new_node
      positions[new_state] = new_node

  def best_child(self,c_param=1.4,thinking=True):
    excepted_reward = {}
    env = Env.from_state(self.dots,self.state)
    #print(len(env.action_space()))
    for action in env.action_space():
      if thinking:
        excepted_reward[action] = self.children[action].total_reward/(self.children[action].visits+1e-8) + c_param*math.sqrt(math.log(self.visits + 1))/(self.children[action].visits +1e-8)
      else:
        excepted_reward[action] = self.children[action].total_reward/self.children[action].visits
    reward = 0
    if self.turn == 1:
      reward = max(excepted_reward.values())
    elif self.turn == -1:
      reward =  min(excepted_reward.values())
    for key in excepted_reward:
      if reward == excepted_reward[key]:
        return key
    return None

class mcts:
  def __init__(self):
    self.root = []

  def simulate(self,node):
    from agents.minmax import Play as greed
    greedy = greed()
    env = Env.from_state(node.dots,node.state)
    turn = node.state[-1]
    #print("received this")
    #env.render()
    while not env.gameover():
      epsilon = 0.8
      if random.random() > epsilon:
        action = random.choice(env.action_space())
      else:
        action = greedy.play(env,turn)
      value = env.step(action,turn)
      if value == 0:
        turn *=-1
    boxes = sum(env.boxes)
    #print(f"epsilon greedy value {boxes}")
    #env.render()
    if boxes > 0: return 1
    elif boxes <0: return -1
    return 0
    
  
  def backpropagate(self,node,reward,discount = 1):
    while(node):
      node.visits += 1
      node.total_reward += reward
      reward *= discount
      if not node.parent:
        break
      node = node.parent
    return node
  
  def train(self,dots,simulations=200000):
    env = Env(dots)
    state = tuple(env.grid + env.boxes + [1])
    node = Node(dots,state,1)
    positions = {}
    positions[state] = node
    #node = self.root
    threshold = 100
    for i in range(simulations):
      gameover = Env.Gameover(dots,node.state)
      while not gameover:
        while node.children:
          min_visits = 100
          min_action = -9
          for action in list(node.children.keys()):
            if node.children[action].visits < min_visits:
              min_visits = node.children[action].visits
              min_action = action
          if min_visits < threshold:
            action = min_action
          else:
            action = node.best_child()
          #print(f"choosing best child {action}")
          node = node.children[action]
          gameover = Env.Gameover(dots,node.state)
          if gameover:
            break

        if gameover:
          break

        node.expand(positions)
        child = random.choice(list(node.children.values()))
        value = self.simulate(child)
        node = child
        node = self.backpropagate(node,value)
        #print(f"reward backpropagated {value}")
        #self.show(node)

      total_reward = sum(Env.from_state(dots,node.state).boxes)
      reward = 0
      if total_reward > 0:
        reward = 1 
      elif total_reward < 0:
        reward = -1 
      else :
        reward = 0
      node = self.backpropagate(node, reward)  
    print(f"training done, saving model at trained_models/mcts{dots}.pkl")   
    with open (f"trained_models/mcts{dots}.pkl","wb") as f:
      pickle.dump(positions,f)

  def think(self,dots,root,positions,simulations):
    node = root
    node.parent = None
    #print(f"recieved {node.state}")
    threshold = 100
    start_time = time.time()
    for i in range(simulations):
      print(i)
      gameover = Env.Gameover(dots,node.state)
      while not gameover:
        while node.children:
          
          min_visits = 101
          min_action = -9
          for action in list(node.children.keys()):
            if node.children[action].visits < min_visits:
              min_visits = node.children[action].visits
              min_action = action
          if min_visits < threshold:
            action = min_action
          else:
            action = node.best_child()
          print(f"choosing best child {action}")
          node = node.children[action]
          gameover = Env.Gameover(dots,node.state)
          if gameover:
            break

        if gameover:
          break

        node.expand(positions)
        child = random.choice(list(node.children.values()))
        value = self.simulate(child)
        node = child
        node = self.backpropagate(node,value)
        print(f"expanding and reward backpropagated {value}")
        #self.show(node)

      total_reward = sum(Env.from_state(dots,node.state).boxes)
      reward = 0
      if total_reward > 0:
        reward = 1 
      elif total_reward < 0:
        reward = -1 
      else :
        reward = 0
      node = self.backpropagate(node, reward)
      end_time = time.time()
    t = end_time-start_time
    print(f'time {t}')
    for action in node.children.keys():
        print(f"action {action} value: {node.children[action].total_reward/node.children[action].visits}") 

    action =  node.best_child(thinking=False) 
    print(f"expected reward {node.children[action].total_reward/node.children[action].visits}") 
    #print(node.state)
    #self.show(node.children[action])  
    return action
  
  def show(self, node, depth=0):
      indent = "  " * depth
      print(f"{indent}Node state: {node.state}, visits: {node.visits}, total reward: {node.total_reward:.2f}")

      for action, child in node.children.items():
          expected_reward = child.total_reward / (child.visits + 1e-8)
          print(f"{indent}├── Action {action} → visits: {child.visits}, total reward: {child.total_reward:.2f}, expected reward: {expected_reward:.4f}")
          self.show(child, depth + 1)  # recursive call for the child node

      if not node.children:
          print(f"{indent}└── [Leaf node]")
  
 


