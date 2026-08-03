import numpy as np
import random

def get_symmetry_count(h, v):
    """Returns 8 symmetries for square grids, or 4 for rectangular grids."""
    H = 2 * h - 1
    W = 2 * v - 1
    return 8 if H == W else 4

def transform_state_and_policy(state_np, policy_np, sym_idx):
    """
    Transforms state_np of shape (channels, H, W) and policy_np of shape (H, W) or None
    using the specified symmetry index. Swaps horizontal/vertical channels (1 and 2)
    when sym_idx >= 4 (axes-swapping transformations).
    """
    if sym_idx == 0:  # Identity
        return state_np, policy_np
    
    s = state_np.copy()
    p = policy_np.copy() if policy_np is not None else None
    
    if sym_idx == 1:  # Flip horizontal
        s = np.flip(s, axis=2)
        p = np.flip(p, axis=1) if p is not None else None
        
    elif sym_idx == 2:  # Flip vertical
        s = np.flip(s, axis=1)
        p = np.flip(p, axis=0) if p is not None else None
        
    elif sym_idx == 3:  # Rotate 180
        s = np.rot90(s, k=2, axes=(1, 2))
        p = np.rot90(p, k=2) if p is not None else None
        
    elif sym_idx == 4:  # Transpose (Diagonal reflection)
        s = np.transpose(s, axes=(0, 2, 1))
        p = np.transpose(p) if p is not None else None
        # Swap horizontal and vertical line channels
        s[[1, 2]] = s[[2, 1]]
        
    elif sym_idx == 5:  # Anti-transpose
        s = np.flip(np.transpose(s, axes=(0, 2, 1)), axis=(1, 2))
        p = np.flip(np.transpose(p), axis=(0, 1)) if p is not None else None
        # Swap horizontal and vertical line channels
        s[[1, 2]] = s[[2, 1]]
        
    elif sym_idx == 6:  # Rotate 90 CCW
        s = np.rot90(s, k=1, axes=(1, 2))
        p = np.rot90(p, k=1) if p is not None else None
        # Swap horizontal and vertical line channels
        s[[1, 2]] = s[[2, 1]]
        
    elif sym_idx == 7:  # Rotate 270 CCW
        s = np.rot90(s, k=3, axes=(1, 2))
        p = np.rot90(p, k=3) if p is not None else None
        # Swap horizontal and vertical line channels
        s[[1, 2]] = s[[2, 1]]

    return s, p

def transform_action(action, sym_idx, H, W):
    """
    Transforms action coordinate tuple (r, c) based on the specified symmetry index.
    """
    r, c = action
    if sym_idx == 0:  # Identity
        return (r, c)
    elif sym_idx == 1:  # Flip horizontal
        return (r, W - 1 - c)
    elif sym_idx == 2:  # Flip vertical
        return (H - 1 - r, c)
    elif sym_idx == 3:  # Rotate 180
        return (H - 1 - r, W - 1 - c)
    elif sym_idx == 4:  # Transpose (Diagonal reflection)
        return (c, r)
    elif sym_idx == 5:  # Anti-transpose
        return (H - 1 - c, W - 1 - r)
    elif sym_idx == 6:  # Rotate 90 CCW
        return (H - 1 - c, r)
    elif sym_idx == 7:  # Rotate 270 CCW
        return (c, W - 1 - r)
    return (r, c)
