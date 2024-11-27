# Q-Learning for Grid Navigation: 8x8 Environment

## Introduction
The **Q-Learning Grid Game** is a reinforcement learning project designed for grid-based navigation using Markov Decision Processes (MDPs). The project features a graphical user interface (GUI) that visualizes the agent's journey across an 8x8 grid. The objective is to guide the agent from the bottom-left corner to the top-right corner while maximizing rewards and avoiding deaths.

This project provides an implementation of reinforcement learning concepts and offers a comparative analysis of four algorithms: Value Iteration, Policy Iteration, Q-Learning, and State-Action-Reward-State-Action (SARSA).

---

## Setup Instructions

### Step 1: Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/y00628/q-learning-grid-game
cd q-learning-grid-game
```

### Step 2: Activate the Virtual Environment
Activate the Python virtual environment to isolate project dependencies:
```bash
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
```

### Step 3: Install Dependencies
Install the required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Running the GUI
To start the Markov Decision Process GUI and interact with the grid environment:
1. Navigate to the project directory.
2. Run the following command:
   ```bash
   py mdp_gui.py
   ```

The GUI will launch, allowing you to interact with the grid environment and observe the paths of the agent given specified algorithms.

---

## Training and Testing Algorithms

The training and testing of algorithms are controlled through specific key bindings in the GUI. Once the application is running, use the following keys to interact with the agent and initiate various algorithms:

- **Movement Controls**:
  - `Arrow Keys (↑, ↓, ←, →)`: Move the agent up, down, left, or right.
  - `Space`: Reset the game to its initial state.

- **Q-Learning**:
  - `Q`: Train the agent using Q-Learning.
  - `T`: Test the agent's performance using a pre-trained Q-Learning agent.

- **SARSA**:
  - `S`: Train the agent using the SARSA algorithm.
  - `D`: Test the agent's performance using a pre-trained SARSA agent.

- **Policy Iteration**:
  - `P`: Train the agent using Policy Iteration.
  - `Y`: Test the policy derived from Policy Iteration.

- **Value Iteration**:
  - `V`: Train the agent using Value Iteration.
  - `B`: Test the policy derived from Value Iteration.

- **Environment Controls**:
  - `R`: Change the map layout and start a new game.

Use the above instructions to experiment with different reinforcement learning algorithms and train/test agents.

---

## Features
- **GUI-based Visualization**: Observe the agent's learning process on an 8x8 grid.
- **Training Flexibility**: Configure hyperparameters for algorithms.
- **Testing and Evaluation**: Test trained models and visualize their performance.
- **Multiple Algorithms**: Expandable to include additional reinforcement learning methods.

---

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request for bug fixes or new features.

---

For questions or issues, feel free to open an issue on the [GitHub repository](https://github.com/y00628/q-learning-grid-game).
