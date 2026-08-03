import os
import sys
import pickle

# Add workspace to path
sys.path.append(os.getcwd())
import scratch.measure_play as mp

if __name__ == "__main__":
    db_path = "trained_models/solved3x3_compact.pkl"
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found! Please run the solver first.")
        sys.exit(1)
        
    print("Loading perfect compact 4x4 Dots & Boxes database...")
    with open(db_path, "rb") as f:
        best_actions = pickle.load(f)
        
    print("Injecting database into Cython analyzer and starting analysis...")
    mp.load_db(best_actions)
    mp.run_analysis()
