class Play:
    def play(self,env,turn,secs=0):
        import random
        ok_actions = []
        for action in env.action_space():
            new_env = env.clone()
            reward = new_env.step(action,turn)
            if reward * turn > 0:
                return action
            elif reward == 0:
                ok_actions.append(action)
        draws = []
        for action in ok_actions:
            new_env = env.clone()
            new_env.step(action,turn)
            draw = True
            for action1 in new_env.action_space():
                new_env1 = new_env.clone()
                reward = new_env1.step(action1,-turn)
                if reward * (-turn) == 1:
                    draw = False
                    break
            if draw:
                draws.append(action)
        #print(draws)
        if draws:
            return random.choice(draws)
        else:
            return random.choice(env.action_space())
	