import os
import sys
import time
import pickle

# Add workspace to path
sys.path.append(os.getcwd())
import env.gameenv as gameenv
import scratch.solve_4x4 as solver

if __name__ == "__main__":
    print("Initializing 4x4 Dots and Boxes environment...")
    initial_state = gameenv.get_initial_state((4, 4))
    
    print("Solving 4x4 Dots and Boxes using Cython transposition table solver...")
    start_time = time.time()
    
    # Run the compiled recursive solver
    outcome = solver.solve(initial_state)
    elapsed = time.time() - start_time
    
    memo = solver.get_memo()
    best_actions = solver.get_best_actions()
    
    print("\n--- Solving Complete ---")
    print(f"Time Taken: {elapsed:.4f} seconds ({elapsed/60.0:.2f} minutes)")
    print(f"Total Unique States Memoized: {len(memo)}")
    print(f"Perfect Play Outcome (P1 boxes - P2 boxes): {outcome}")
    
    # Save output to pickle file
    os.makedirs("trained_models", exist_ok=True)
    save_path = "trained_models/solved3x3_compact.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(best_actions, f)
        
    print(f"Compact solved 4x4 database saved to {save_path} (size: {os.path.getsize(save_path)/(1024*1024):.2f} MB)")
