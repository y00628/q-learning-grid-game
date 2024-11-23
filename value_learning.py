import numpy as np
import time
import matplotlib.pyplot as plt
def get_reward_function(grid_game):
    """Creates a reward table based on the death matrix and goal state."""
    reward_table = np.full(grid_game.grid_size * grid_game.grid_size, -0.02)  # Default reward for normal states

    for r in range(grid_game.grid_size):
        for c in range(grid_game.grid_size):
            state = grid_game.get_state_from_pos((r, c))
            if (r, c) == grid_game.goal_position:
                reward_table[state] = 10  # Goal state reward
            elif grid_game.D[c, r] > 0.8:  # Death state probability
                reward_table[state] = -1  # Death state penalty

    return reward_table

def get_transition_model(grid_game, random_rate=0.2):
    """Creates a transition probability model for the grid game."""
    num_states = grid_game.grid_size * grid_game.grid_size
    num_actions = len(grid_game.actions)
    transition_model = np.zeros((num_states, num_actions, num_states))

    start_state = grid_game.get_state_from_pos(grid_game.start_position)

    for r in range(grid_game.grid_size):
        for c in range(grid_game.grid_size):
            state = grid_game.get_state_from_pos((r, c))

            if grid_game.D[c, r] > 0.8:  # Current state is a death state
                # Redirect all actions to the starting state
                for a in range(num_actions):
                    transition_model[state, a, :] = 0
                    transition_model[state, a, start_state] = 1
                continue

            # For valid states, calculate normal transitions
            neighbors = np.zeros(num_actions)
            for a, (dr, dc) in enumerate(grid_game.actions):
                new_r = max(0, min(r + dr, grid_game.grid_size - 1))
                new_c = max(0, min(c + dc, grid_game.grid_size - 1))
                neighbors[a] = grid_game.get_state_from_pos((new_r, new_c))

            for a in range(num_actions):
                # Main transition to the intended state
                transition_model[state, a, int(neighbors[a])] += (1 - random_rate)

                # Random transitions to left and right neighbors
                transition_model[state, a, int(neighbors[(a + 1) % num_actions])] += (random_rate / 2.0)
                transition_model[state, a, int(neighbors[(a - 1) % num_actions])] += (random_rate / 2.0)

            # Normalize probabilities
            for a in range(num_actions):
                total_prob = np.sum(transition_model[state, a, :])
                if total_prob > 0:
                    transition_model[state, a, :] /= total_prob

    return transition_model

class ValueIteration:
    def __init__(self, grid_game, gamma=0.95):
        """
        Initialize the ValueIteration class.

        Parameters:
            grid_game: The GridGame object containing the game state, rewards, and transitions.
            gamma: Discount factor for future rewards.
        """
        self.grid_game = grid_game
        self.num_states = grid_game.grid_size * grid_game.grid_size
        self.num_actions = len(grid_game.actions)
        self.reward_function = get_reward_function(grid_game)  # Provided function
        self.transition_model = get_transition_model(grid_game)  # Provided function
        self.gamma = gamma
        self.values = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)

    def one_iteration(self):
        """Perform one iteration of the value iteration algorithm."""
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            action_values = np.zeros(self.num_actions)

            # Compute the value of each action
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                action_values[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)

            # Update the value for the state
            self.values[s] = np.max(action_values)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def get_policy(self):
        """Extract the optimal policy based on the current value function."""
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            action_values = np.zeros(self.num_actions)

            # Compute the value of each action
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                action_values[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)

            # Choose the action with the maximum value
            policy[s] = np.argmax(action_values)
        return policy

    def train(self, tol=1e-3):
        """Train the value iteration algorithm until convergence."""
        delta_history = []
        while True:
            delta = self.one_iteration()
            delta_history.append(delta)
            if delta < tol:
                break
        self.policy = self.get_policy()  # Compute the optimal policy after convergence

        # Plot convergence history
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
        ax.plot(range(1, len(delta_history) + 1), delta_history, marker='o', color='blue')
        ax.set_title("Value Iteration Convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Delta (Change in Values)")
        plt.tight_layout()
        plt.savefig("value_iteration_convergence.png")
        print("Convergence plot saved as value_iteration_convergence.png")

    def test_policy(self):
        """Test the learned policy by navigating the grid based on the optimal policy."""
        position = self.grid_game.start_position  # Start at the initial position
        total_reward = 0
        steps = 0
        visited_positions = set()

        print("Testing the learned policy...")
        while position != self.grid_game.goal_position and steps < 100:
            state = self.grid_game.get_state_from_pos(position)
            action = self.policy[state]
            dr, dc = self.grid_game.actions[action]
            new_position = (
                max(0, min(position[0] + dr, self.grid_game.grid_size - 1)),
                max(0, min(position[1] + dc, self.grid_game.grid_size - 1))
            )

            # Check if the new position is a death zone
            if self.grid_game.D[new_position[1], new_position[0]] > 0.8:
                print(f"Hit a death zone at {new_position}. Resetting to start.")
                position = self.grid_game.start_position
                total_reward -= 10  # Penalty for hitting a death state
                visited_positions.clear()  # Reset visited positions
                steps = 0  # Restart step count
                continue

            if new_position in visited_positions:
                total_reward -= 0.1  # Penalize revisiting states
            visited_positions.add(new_position)

            position = new_position
            total_reward += self.grid_game.R[position[1], position[0]] - 0.04  # Add reward for moving
            steps += 1

            # Update and display the grid
            self.grid_game.player_position = position
            self.grid_game.draw_player()
            self.grid_game.root.update()
            time.sleep(0.5)

            print(f"Step {steps}, Position: {position}, Total Reward: {total_reward}")

        # Print final result
        if position == self.grid_game.goal_position:
            print("Goal reached!")
        else:
            print("Test ended. Maximum steps reached.")
        print(f"Final Reward: {total_reward}")