import os
import json
import math

def _get_filenames(h, v, agent=None):
    if agent:
        return f"elo_ratings_{h}x{v}_{agent}.json", f"match_history_{h}x{v}_{agent}.csv"
    return f"elo_ratings_{h}x{v}.json", f"match_history_{h}x{v}.csv"

def _load_ratings(h, v, agent=None):
    """Loads player ratings from the JSON database file."""
    ratings_file, _ = _get_filenames(h, v, agent)
    if not os.path.exists(ratings_file):
        return {}
    try:
        with open(ratings_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def _save_ratings(ratings, h, v, agent=None):
    """Saves player ratings to the JSON database file."""
    ratings_file, _ = _get_filenames(h, v, agent)
    with open(ratings_file, 'w') as f:
        json.dump(ratings, f, indent=4)

def _init_csv(h, v, agent=None):
    """Initializes the CSV match log file with a header if it does not exist."""
    _, log_file = _get_filenames(h, v, agent)
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("p1,p2,outcome,p1_elo_before,p2_elo_before,p1_elo_after,p2_elo_after\n")

def log_match(player1, player2, outcome, h, v, agent=None, K=32):
    """
    Logs a match outcome and updates Elo ratings sequentially.
    outcome is 1.0 (player1 win), 0.5 (draw), or 0.0 (player2 win).
    """
    _init_csv(h, v, agent)
    ratings = _load_ratings(h, v, agent)

    for p in [player1, player2]:
        if p not in ratings:
            initial_rating = 1000.0 if p == 'minmax' else 0.0
            ratings[p] = {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "rating": initial_rating,
                "rating_history": [initial_rating]
            }

    p1_stats = ratings[player1]
    p2_stats = ratings[player2]

    R1 = p1_stats["rating"]
    R2 = p2_stats["rating"]

    diff1 = max(-100.0, min(100.0, (R2 - R1) / 400.0))
    diff2 = max(-100.0, min(100.0, (R1 - R2) / 400.0))

    E1 = 1.0 / (1.0 + math.exp(diff1))
    E2 = 1.0 / (1.0 + math.exp(diff2))

    R1_new = R1 + K * (outcome - E1)
    R2_new = R2 + K * ((1.0 - outcome) - E2)

    if player1 == 'minmax':
        R1_new = 1000.0
    if player2 == 'minmax':
        R2_new = 1000.0

    p1_stats["rating"] = round(R1_new, 2)
    p2_stats["rating"] = round(R2_new, 2)

    p1_stats["rating_history"].append(p1_stats["rating"])
    p2_stats["rating_history"].append(p2_stats["rating"])

    if outcome == 1.0:
        p1_stats["wins"] += 1
        p2_stats["losses"] += 1
    elif outcome == 0.0:
        p1_stats["losses"] += 1
        p2_stats["wins"] += 1
    else:
        p1_stats["draws"] += 1
        p2_stats["draws"] += 1

    _save_ratings(ratings, h, v, agent)

    _, log_file = _get_filenames(h, v, agent)
    with open(log_file, 'a') as f:
        f.write(f"{player1},{player2},{outcome},{R1:.2f},{R2:.2f},{R1_new:.2f},{R2_new:.2f}\n")
