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

### Training
To train an agent using Q-Learning or other implemented algorithms:
1. Open the `train.py` file in the project directory.
2. Configure parameters such as learning rate, discount factor, and exploration strategy.
3. Run the training script:
   ```bash
   python train.py
   ```

The trained model will be saved in the specified directory or displayed in the terminal.

### Testing
To test the performance of a trained agent:
1. Open the `test.py` file.
2. Ensure the model path is set to the trained model file.
3. Run the testing script:
   ```bash
   python test.py
   ```

The test results, including the policy learned and performance metrics, will be displayed in the terminal or plotted for visualization.

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

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

For questions or issues, feel free to open an issue on the [GitHub repository](https://github.com/y00628/q-learning-grid-game).
