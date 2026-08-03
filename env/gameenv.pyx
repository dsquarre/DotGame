import numpy as np

def _get_hw(dots):
    """Helper to extract board H and W from dots configuration (int or tuple)."""
    if isinstance(dots, tuple):
        dh, dv = dots
        return 2 * dh - 1, 2 * dv - 1
    else:
        return 2 * dots - 1, 2 * dots - 1

def array_to_nested_tuple(arr):
    """Converts a 3D numpy array to a nested tuple representation."""
    return tuple(
        tuple(
            tuple(row) for row in channel
        ) for channel in arr
    )

def get_initial_state(dots):
    """Returns the initial state tuple of shape (6, H, W)."""
    H, W = _get_hw(dots)
    obs = np.zeros((6, H, W), dtype=np.float32)
    obs[0, :, :] = 1.0  # Turn starts at 1
    # Channel 5 starts at 0.0 (no bonus move yet)
    return array_to_nested_tuple(obs)

def get_action_space(state):
    """Returns the list of legal coordinate actions (r, c) directly from a state tuple."""
    H = len(state[0])
    W = len(state[0][0])
    actions = []
    latest_p1 = state[1]
    latest_p2 = state[2]
    for r in range(H):
        for c in range(W):
            if (r + c) % 2 == 1:
                if latest_p1[r][c] == 0.0 and latest_p2[r][c] == 0.0:
                    actions.append((r, c))
    return actions

def is_gameover(state):
    """Returns whether a given state tuple represents a finished game."""
    return len(get_action_space(state)) == 0

def step(state, action):
    """
    Pure state transition function: computes the next state and reward given a parent state and action.
    Returns (new_state_tuple, reward).
    """
    H = len(state[0])
    W = len(state[0][0])
    latest_obs = np.array(state, dtype=np.float32)
    turn = 1 if latest_obs[0, 0, 0] > 0 else -1
    
    r, c = action
    if (r + c) % 2 != 1 or latest_obs[1, r, c] != 0.0 or latest_obs[2, r, c] != 0.0:
        raise ValueError(f"Invalid move at coordinate {action}")
        
    new_obs = np.copy(latest_obs)
    line_channel = 1 if turn == 1 else 2
    new_obs[line_channel, r, c] = 1.0
    
    completed_boxes = 0
    if r % 2 == 0:
        adjacent_boxes = []
        if r > 0:
            adjacent_boxes.append((r - 1, c))
        if r < H - 1:
            adjacent_boxes.append((r + 1, c))
    else:
        adjacent_boxes = []
        if c > 0:
            adjacent_boxes.append((r, c - 1))
        if c < W - 1:
            adjacent_boxes.append((r, c + 1))
            
    for br, bc in adjacent_boxes:
        if new_obs[3, br, bc] == 0.0 and new_obs[4, br, bc] == 0.0:
            l1 = new_obs[1, br - 1, bc] != 0.0 or new_obs[2, br - 1, bc] != 0.0
            l2 = new_obs[1, br + 1, bc] != 0.0 or new_obs[2, br + 1, bc] != 0.0
            l3 = new_obs[1, br, bc - 1] != 0.0 or new_obs[2, br, bc - 1] != 0.0
            l4 = new_obs[1, br, bc + 1] != 0.0 or new_obs[2, br, bc + 1] != 0.0
            
            if l1 and l2 and l3 and l4:
                box_channel = 3 if turn == 1 else 4
                new_obs[box_channel, br, bc] = 1.0
                completed_boxes += 1
                
    next_turn = turn if completed_boxes > 0 else -turn
    new_obs[0, :, :] = float(next_turn)
    # Set the 6th channel to all 1.0 if this is a consecutive/bonus move
    new_obs[5, :, :] = 1.0 if completed_boxes > 0 else 0.0
    
    next_state = array_to_nested_tuple(new_obs)
    
    if is_gameover(next_state):
        p1_boxes = np.sum(new_obs[3])
        p2_boxes = np.sum(new_obs[4])
        if p1_boxes > p2_boxes:
            reward = 1.0
        elif p2_boxes > p1_boxes:
            reward = -1.0
        else:
            reward = 0.0
    else:
        reward = 0.0
        
    return next_state, reward

def render(state):
    """Renders the board to the console matching the original Env.py layout with rc coordinates."""
    H = len(state[0])
    W = len(state[0][0])
    latest = state
    
    for r in range(H):
        for c in range(W):
            p1_line = latest[1][r][c] == 1.0
            p2_line = latest[2][r][c] == 1.0
            p1_box = latest[3][r][c] == 1.0
            p2_box = latest[4][r][c] == 1.0
            
            if r % 2 == 0:
                if c % 2 == 0:
                    print("\033[42m  \033[m", end='')
                else:
                    if p1_line or p2_line:
                        print("\033[47m  \033[m", end='')
                    else:
                        print(f"\033[40m{r}{c}\033[m", end='')
            else:
                if c % 2 == 0:
                    if p1_line or p2_line:
                        print("\033[47m  \033[m", end='')
                    else:
                        print(f"\033[40m{r}{c}\033[m", end='')
                else:
                    if p1_box:
                        print("\033[41m  \033[m", end='')
                    elif p2_box:
                        print("\033[44m  \033[m", end='')
                    else:
                        print("\033[100m  \033[m", end='')
        print()
    print()
