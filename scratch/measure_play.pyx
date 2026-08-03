# cython: language_level=3
import time
import numpy as np
import env.gameenv as gameenv

cdef dict best_actions = {}
cdef dict memo_values = {}

def load_db(dict db):
    global best_actions
    best_actions = db
    memo_values.clear()

cdef unsigned long long compress_state(object state):
    cdef unsigned long long bitmask = 0
    cdef unsigned int bit_idx = 0
    cdef int r, c
    
    for r in range(7):
        for c in range(7):
            if (r + c) % 2 == 1:
                if state[1][r][c] > 0.0 or state[2][r][c] > 0.0:
                    bitmask |= (<unsigned long long>1 << bit_idx)
                bit_idx += 1
                
    cdef unsigned long long box_code = 0
    cdef unsigned int box_idx = 0
    cdef unsigned long long code
    for br in [1, 3, 5]:
        for bc in [1, 3, 5]:
            if state[3][br][bc] > 0.0:
                code = 1
            elif state[4][br][bc] > 0.0:
                code = 2
            else:
                code = 0
            box_code |= (code << (2 * box_idx))
            box_idx += 1
            
    cdef unsigned long long turn_bit = 1 if state[0][0][0] > 0.0 else 0
    cdef unsigned long long bonus_bit = 1 if state[5][0][0] > 0.0 else 0
    
    return (bitmask << 20) | (box_code << 2) | (turn_bit << 1) | bonus_bit

cdef int get_outcome_score(object state):
    cdef int p1_boxes = 0
    cdef int p2_boxes = 0
    cdef int r, c
    for r in range(7):
        for c in range(7):
            if state[3][r][c] > 0.0:
                p1_boxes += 1
            if state[4][r][c] > 0.0:
                p2_boxes += 1
    return p1_boxes - p2_boxes

cdef int evaluate_state_value(object state, int alpha, int beta):
    cdef unsigned long long c_state, c_curr
    cdef object curr, act, next_state, action
    cdef int val, best_val, turn, act_idx
    cdef list actions
    cdef int step_num = 0
    
    if gameenv.is_gameover(state):
        return get_outcome_score(state)
        
    c_state = compress_state(state)
    if c_state in memo_values:
        return memo_values[c_state]
        
    if c_state in best_actions:
        # Follow best actions to the end
        curr = state
        while not gameenv.is_gameover(curr):
            c_curr = compress_state(curr)
            if c_curr in best_actions:
                act_idx = best_actions[c_curr]
                act = (act_idx // 7, act_idx % 7)
                curr, _ = gameenv.step(curr, act)
                step_num += 1
            else:
                return evaluate_state_value(curr, alpha, beta)
        val = get_outcome_score(curr)
        memo_values[c_state] = val
        return val
        
    turn = 1 if state[0][0][0] > 0 else -1
    actions = gameenv.get_action_space(state)
    
    if turn == 1:
        best_val = -100
        for action in actions:
            next_state, _ = gameenv.step(state, action)
            val = evaluate_state_value(next_state, alpha, beta)
            if val > best_val:
                best_val = val
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        memo_values[c_state] = best_val
        return best_val
    else:
        best_val = 100
        for action in actions:
            next_state, _ = gameenv.step(state, action)
            val = evaluate_state_value(next_state, alpha, beta)
            if val < best_val:
                best_val = val
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        memo_values[c_state] = best_val
        return best_val

def run_analysis():
    memo_values.clear()
    state = gameenv.get_initial_state((4, 4))
    move_count = 1
    
    print("=" * 60)
    print("EMPERICAL MARGIN OF ERROR ANALYSIS (4x4 Dots / 3x3 Boxes)")
    print("=" * 60)
    
    while not gameenv.is_gameover(state):
        turn = 1 if state[0][0][0] > 0 else -1
        player_name = "Player 1" if turn == 1 else "Player 2"
        
        curr_optimal_val = evaluate_state_value(state, -100, 100)
        
        actions = gameenv.get_action_space(state)
        optimal_actions = []
        blunder_actions = []
        
        for action in actions:
            next_state, _ = gameenv.step(state, action)
            
            # Zero-window search in Cython
            if turn == 1:
                val = evaluate_state_value(next_state, curr_optimal_val - 1, curr_optimal_val)
                if val >= curr_optimal_val:
                    optimal_actions.append(action)
                else:
                    blunder_actions.append(action)
            else:
                val = evaluate_state_value(next_state, curr_optimal_val, curr_optimal_val + 1)
                if val <= curr_optimal_val:
                    optimal_actions.append(action)
                else:
                    blunder_actions.append(action)
                    
        total_legal = len(actions)
        num_optimal = len(optimal_actions)
        pct_optimal = (num_optimal / total_legal) * 100
        
        print(f"\nMove {move_count}: {player_name}'s Turn")
        print(f"  Current Board Value (P1 - P2): {curr_optimal_val}")
        print(f"  Legal Moves: {total_legal} | Optimal: {num_optimal} ({pct_optimal:.1f}%) | Blunders: {total_legal - num_optimal}")
        if blunder_actions:
            print(f"  Sample Blunders: {', '.join(str(a) for a in blunder_actions[:5])}")
            
        # Follow best action
        c_state = compress_state(state)
        best_act_idx = best_actions[c_state]
        best_act = (best_act_idx // 7, best_act_idx % 7)
        state, _ = gameenv.step(state, best_act)
        move_count += 1
        
    print("\n" + "=" * 60)
    print(f"Game finished! Final score: {get_outcome_score(state)}")
    print("=" * 60)
