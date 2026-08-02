import time
import random
import numpy as np
import env.gameenv as gameenv

def minimax_search(state, depth, alpha, beta, maximizing_player, start_time, time_limit):
    """Recursive minimax search with alpha-beta pruning."""
    if time.time() - start_time >= time_limit:
        raise TimeoutError()

    if depth == 0 or gameenv.is_gameover(state):
        p1_boxes = np.sum(state[-1][3])
        p2_boxes = np.sum(state[-1][4])
        return p1_boxes - p2_boxes

    actions = gameenv.get_action_space(state)

    if maximizing_player:
        max_eval = -float('inf')
        for action in actions:
            next_state, reward = gameenv.step(state, action)
            next_turn = 1 if next_state[-1][0][0][0] > 0 else -1
            eval_val = minimax_search(next_state, depth - 1, alpha, beta, next_turn == 1, start_time, time_limit)
            max_eval = max(max_eval, eval_val)
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for action in actions:
            next_state, reward = gameenv.step(state, action)
            next_turn = 1 if next_state[-1][0][0][0] > 0 else -1
            eval_val = minimax_search(next_state, depth - 1, alpha, beta, next_turn == 1, start_time, time_limit)
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval

class Play:
    """Adversarial Minimax search player interface for Dots and Boxes."""
    def __init__(self, h, v):
        self.h = h
        self.v = v

    def play(self, state, turn, time_limit=1.0):
        start_time = time.time()
        actions = gameenv.get_action_space(state)
        if not actions:
            return None

        best_action = random.choice(actions)

        for depth in range(1, 100):
            try:
                best_val = -float('inf') if turn == 1 else float('inf')
                current_best_action = best_action

                for action in actions:
                    if time.time() - start_time >= time_limit:
                        raise TimeoutError()

                    next_state, reward = gameenv.step(state, action)
                    next_turn = 1 if next_state[-1][0][0][0] > 0 else -1

                    eval_val = minimax_search(
                        next_state,
                        depth - 1,
                        -float('inf'),
                        float('inf'),
                        next_turn == 1,
                        start_time,
                        time_limit
                    )

                    if turn == 1:
                        if eval_val > best_val:
                            best_val = eval_val
                            current_best_action = action
                    else:
                        if eval_val < best_val:
                            best_val = eval_val
                            current_best_action = action

                best_action = current_best_action

            except TimeoutError:
                break

        return best_action