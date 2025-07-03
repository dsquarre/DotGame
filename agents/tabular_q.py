import random
from collections import defaultdict
import numpy as np
import pickle
from env.env import Env
from agents.minmax import Play as greed
class qt:
    def train(self,dots,epochs):
        env = Env(dots)
        Q = defaultdict(lambda: np.zeros(len(env.grid)))
        discount_factor = 0.7
        learning_rate = 0.1
        #epochs  = 1000000
        for episode in range(epochs+1):
            turn = random.choice([1,-1])
            state = tuple(env.grid + env.boxes + [turn])
            while not env.gameover():   
                valid_actions = env.action_space()
                epsilon = max(0.01,0.995**(episode))  # Exploration rate
                q_state_action = 0
                if random.random() < epsilon:
                    action = random.choice(valid_actions) # Explore
                else:
                    if turn == 1:
                        valid_q_values = [Q[state][a] for a in valid_actions]
                        action = valid_actions[np.argmax(valid_q_values)] # Exploit
                        q_state_action = Q[state][action]
                    else:
                        q_values = Q[state].copy()
                        valid_q_values = [Q[state][a] for a in valid_actions]
                        action = valid_actions[np.argmin(valid_q_values)]
                        q_state_action = q_values[action]

                reward= float(env.step(action,turn))
                if(reward == 0):
                    turn = -turn
                if(env.gameover()):
                    reward = sum(env.boxes)
                    best_q_next = reward
                else:
                    next_state = tuple(env.grid + env.boxes+[turn])
                    valid_actions_next = env.action_space()
                    best_q_next = 0
                    if turn == 1:
                        valid_q_values = [Q[next_state][a] for a in valid_actions_next]
                        best_q_next = np.max(valid_q_values) # Exploit
                    else:
                        valid_q_values = [Q[next_state][a] for a in valid_actions_next]
                        best_q_next = np.min(valid_q_values) # Exploit

                Q[state][action] += learning_rate * (reward + discount_factor * best_q_next - q_state_action)
                #print(best_q_next)
                
            env.reset()
            if(episode%1000 == 0):
                print(f"no of q values stored: {len(Q)} ,episodes : {episode}")

        with open(f"trained_models/Q{dots}.pkl", "wb") as f:
            pickle.dump(dict(Q),f)
        #print(Q)
        print(f"training finished, model stored in trained_models/Q{dots}.pkl")
        
class Play:
   def play(self,env,turn,secs=0):
        dots = env.dots
        Q = defaultdict(lambda: np.zeros(len(env.grid)))
        try:
            with open(f"trained_models/Q{dots}.pkl","r") as f:
                Q = defaultdict(lambda: np.zeros(len(env.grid)), pickle.load(f))
        except Exception:
            #print("Please train the model first, playing randomnly now")
            greedy = greed()
            return greedy.play(env,turn)
        state = tuple(env.grid + env.boxes + [turn])
        valid_actions = env.action_space()
        valid_q_values = [Q[state][a] for a in valid_actions]
        action = valid_actions[np.argmax(valid_q_values)]
        return action
