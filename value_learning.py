import numpy as np
import random
import matplotlib.pyplot as plt

def train_value_iteration(grid_game):
    """Trains the agent using Value Iteration."""
    grid_game.create_transition()

    grid_size = grid_game.grid_size
    num_states = grid_size * grid_size
    num_actions = len(grid_game.actions)
    grid_game.policy = np.zeros((grid_size, grid_size), dtype=int)

    discount_factor = 0.95
    threshold = 0.001  # Convergence threshold
    max_iterations = 1000  # Safety limit to prevent infinite loops

    # Initialize value function for all states
    rewards_per_iteration = []

    for iteration in range(max_iterations):
        delta = 0  # Tracks the maximum change in V
        
        # Loop through all states
        for y in range(grid_size):
            for x in range(grid_size):
                # State format: (x, y)
                state = (x, y)
                
                # Skip terminal states
                if state == grid_game.goal_position:
                    continue
                
                # Compute the Bellman update for state (x, y)
                v = grid_game.V[y, x]  # Current value of the state
                action_values = []  # Stores values for each action
                
                for a in range(num_actions):
                    value = 0
                    for new_y in range(grid_size):
                        for new_x in range(grid_size):
                            prob = grid_game.P[new_y * grid_size + new_x, y * grid_size + x, a]
                            reward = grid_game.R[new_y, new_x]
                            
                            # Check for death states
                            if grid_game.D[new_y, new_x]:
                                reward = -1  # Large negative reward for death
                                grid_game.game_over = True

                            value += prob * (reward + discount_factor * grid_game.V[new_y, new_x])
                    
                    action_values.append(value)
                
                # Update the value for this state using the best action
                grid_game.V[y, x] = max(action_values)
                delta = max(delta, abs(v - grid_game.V[y, x]))

        # Track the total reward (sum of V values) for visualization
        total_reward = np.sum(grid_game.V)
        rewards_per_iteration.append(total_reward)
        
        print(f"Iteration {iteration}: Delta = {delta}, Total Reward = {total_reward}")
        
        # Check for convergence
        if delta < threshold:
            print("Value iteration converged!")
            break

    # Plot rewards
    plt.plot(rewards_per_iteration)
    plt.xlabel("Iteration")
    plt.ylabel("Total Reward")
    plt.title("Value Iteration Training Progress")
    plt.savefig("value_iteration_training_progress.png", dpi=300)
    plt.show()

    # Derive the optimal policy
    for y in range(grid_size):
        for x in range(grid_size):
            # State format: (x, y)
            state = (x, y)
            action_values = []
            
            for a in range(num_actions):
                value = 0
                for new_y in range(grid_size):
                    for new_x in range(grid_size):
                        prob = grid_game.P[new_y * grid_size + new_x, y * grid_size + x, a]
                        reward = grid_game.R[new_y, new_x]
                        value += prob * (reward + discount_factor * grid_game.V[new_y, new_x])
                action_values.append(value)
            
            grid_game.policy[y, x] = np.argmax(action_values)
    
    print("Optimal policy derived!")


def test_value_iteration_policy(grid_game):
    """Tests the agent's policy in the grid game."""
    grid_game.reset_player_position()
    state = grid_game.player_position
    print("Starting testing...")
    print("Policy Matrix:\n", grid_game.policy)
    grid_game.game_over = False

    while not grid_game.game_over:
        x, y = state
        action_index = grid_game.policy[y, x]
        action = grid_game.actions[action_index]

        # Update the state
        new_x = max(0, min(x + (-action[0]), grid_game.grid_size - 1))
        new_y = max(0, min(y + (-action[1]), grid_game.grid_size - 1))
        new_state = (new_x, new_y)
        print(f"Action taken: {action}, State: {state} -> {new_state}")

        # Draw agent movement
        grid_game.player_position = new_state
        grid_game.draw_player()
        grid_game.root.update()

        # Check for game over conditions
        death = np.random.binomial(1, grid_game.D[new_state[0], new_state[1]])
        print(death)
        if death:
            print(f"Agent hit a death state at {new_state}!")
            grid_game.game_over = True
        elif new_state == grid_game.goal_position:
            print(f"Goal reached at {new_state}!")
            grid_game.game_over = True

        state = new_state
