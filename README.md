# DotGame (Dots and Boxes)

Creating a popular game we play on paper and creating agents to play it perfectly.

---

## 🧠 Agents Implemented

Agents are modular and located in `agents/`:
- **Tabular Q-Learning** (`tabular_q.py`)
- **Deep Q-Network (DQN)** (`dqn.py`)
- **Minimax** (`minmax.py`)
- **Monte Carlo Tree Search (MCTS)** (`mcts.py`)
- **AlphaZero-style Agent** (`alphazero.py`)  
  Uses MCTS guided by policy/value neural networks.


---


## Directory Structure

DotGame/<br/>
├── agents/<br/># All agent implementations<br/>
├── env/<br/># DotGame environment<br/>
├── notebook/<br/># Colabs file<br/>
├── legacy/<br/># Old C implementation<br/>
├── trained_models/<br># Saved model files<br/>
├── plots/<br/># Plots saved here after play<br/>
├── train.py<br/># Main training entry point<br/>
├── play.py<br/># Main evaluation entry point<br/>
└── requirements.txt<br/># Libraries used<br/> 
└── README.md<br/># This file<br/>


---


## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train an agent 

```bash
python3 -m train --dots 3 --agent mcts --epochs 400000
```

Supported agents: qt, dqn, mcts, alphazero

### Play bw agents

```bash
python3 -m play --dots 4 --agent1 self --agent2 alphazero --games 1
```

self = user input<br/>
win-loss rate would automatically be plotted after games at plots/


---


Legacy: legacy/dotgame.c

```bash
gcc dotgame.c
./a.out
```

![](dotgame.png)
