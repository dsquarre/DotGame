# cython: language_level=3
import numpy as np
import env.gameenv as gameenv

cdef dict memo = {}
cdef dict best_actions = {}

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
    
    cdef unsigned long long compressed = (bitmask << 20) | (box_code << 2) | (turn_bit << 1) | bonus_bit
    return compressed

cdef int count_boxes(object state):
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

cdef int alpha_beta(object state, int alpha, int beta):
    if gameenv.is_gameover(state):
        return count_boxes(state)
        
    cdef unsigned long long c_state = compress_state(state)
    cdef int stored_val, flag
    if c_state in memo:
        stored_val, flag = memo[c_state]
        if flag == 0:
            return stored_val
        elif flag == 1:
            alpha = max(alpha, stored_val)
        elif flag == 2:
            beta = min(beta, stored_val)
        if alpha >= beta:
            return stored_val
            
    cdef int turn = 1 if state[0][0][0] > 0 else -1
    cdef list actions = gameenv.get_action_space(state)
    cdef int best_val, val, action_idx, alpha_orig, beta_orig
    cdef object best_act = None
    cdef object next_state
    
    alpha_orig = alpha
    beta_orig = beta
    
    if turn == 1:
        best_val = -100
        for action in actions:
            next_state, _ = gameenv.step(state, action)
            val = alpha_beta(next_state, alpha, beta)
            if val > best_val:
                best_val = val
                best_act = action
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
    else:
        best_val = 100
        for action in actions:
            next_state, _ = gameenv.step(state, action)
            val = alpha_beta(next_state, alpha, beta)
            if val < best_val:
                best_val = val
                best_act = action
            beta = min(beta, best_val)
            if beta <= alpha:
                break
                
    # Save to transposition table
    if best_val <= alpha_orig:
        flag = 2 # Upperbound
    elif best_val >= beta_orig:
        flag = 1 # Lowerbound
    else:
        flag = 0 # Exact
        
    memo[c_state] = (best_val, flag)
    if best_act is not None:
        best_actions[c_state] = best_act[0] * 7 + best_act[1]
        
    return best_val

def solve(state):
    return alpha_beta(state, -100, 100)

def get_memo():
    # Return dictionary of exact values
    cdef dict exact_memo = {}
    for k, (v, flag) in memo.items():
        if flag == 0:
            exact_memo[k] = v
    return exact_memo

def get_best_actions():
    return best_actions

def clear_cache():
    memo.clear()
    best_actions.clear()
