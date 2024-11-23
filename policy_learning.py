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

class PolicyIteration:
    def __init__(self, grid_game, gamma=0.95, tol=1e-3):
        """
        Initialize the PolicyIteration class.

        Parameters:
            grid_game: The GridGame object containing the game state, rewards, and transitions.
            gamma: Discount factor.
            tol: Tolerance for policy evaluation convergence.
        """
        self.grid_game = grid_game
        self.num_states = grid_game.grid_size * grid_game.grid_size
        self.num_actions = len(grid_game.actions)
        self.gamma = gamma
        self.tol = tol

        # Initialize rewards and transitions
        self.reward_function = self.get_reward_function(grid_game)
        self.transition_model = self.get_transition_model(grid_game)

        # Initialize values and policy
        self.values = np.zeros(self.num_states)
        self.policy = np.random.randint(0, self.num_actions, self.num_states)

    def get_reward_function(self, grid_game):
        """Generate the reward function based on the GridGame's structure."""
        reward_function = np.full(self.num_states, -0.02)
        for r in range(grid_game.grid_size):
            for c in range(grid_game.grid_size):
                state = grid_game.get_state_from_pos((r, c))
                if (r, c) == grid_game.goal_position:
                    reward_function[state] = 10  # Large reward for goal
                elif grid_game.D[c, r] > 0.8:  # Death zone probability check
                    reward_function[state] = -1  # Penalty for death zones
        return reward_function

    def get_transition_model(self, grid_game, random_rate=0.2):
        """Generate the transition model for the GridGame."""
        num_states = grid_game.grid_size * grid_game.grid_size
        num_actions = len(grid_game.actions)
        transition_model = np.zeros((num_states, num_actions, num_states))

        start_state = grid_game.get_state_from_pos(grid_game.start_position)

        for r in range(grid_game.grid_size):
            for c in range(grid_game.grid_size):
                s = grid_game.get_state_from_pos((r, c))
                if grid_game.D[c, r] > 0.8:  # Death state
                    for a in range(num_actions):
                        transition_model[s, a, :] = 0
                        transition_model[s, a, start_state] = 1  # Reset to start position
                else:
                    neighbors = np.zeros(num_actions)
                    for a, (dr, dc) in enumerate(grid_game.actions):
                        new_r = max(0, min(r + dr, grid_game.grid_size - 1))
                        new_c = max(0, min(c + dc, grid_game.grid_size - 1))
                        neighbors[a] = grid_game.get_state_from_pos((new_r, new_c))
                    
                    for a in range(num_actions):
                        transition_model[s, a, int(neighbors[a])] += (1 - random_rate)
                        transition_model[s, a, int(neighbors[(a + 1) % num_actions])] += (random_rate / 2.0)
                        transition_model[s, a, int(neighbors[(a - 1) % num_actions])] += (random_rate / 2.0)
                    
                    for a in range(num_actions):
                        total_prob = np.sum(transition_model[s, a, :])
                        if total_prob > 0:
                            transition_model[s, a, :] /= total_prob

        return transition_model

    def one_policy_evaluation(self):
        """Perform one sweep of policy evaluation."""
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            a = self.policy[s]
            p = self.transition_model[s, a]
            self.values[s] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def run_policy_evaluation(self):
        """Run policy evaluation until convergence."""
        while True:
            delta = self.one_policy_evaluation()
            if delta < self.tol:
                break

    def run_policy_improvement(self):
        """Run one sweep of policy improvement."""
        policy_stable = True
        for s in range(self.num_states):
            old_action = self.policy[s]
            action_values = [
                np.sum(self.transition_model[s, a] * 
                       (self.reward_function + self.gamma * self.values))
                for a in range(self.num_actions)
            ]
            self.policy[s] = np.argmax(action_values)
            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable

    def train(self, tol=1e-3, plot=True):
        """Train the policy using policy iteration and optionally plot statistics."""
        eval_count_history = []
        policy_change_history = []

        while True:
            # Run policy evaluation
            eval_count = 0
            while True:
                delta = self.one_policy_evaluation()
                eval_count += 1
                if delta < tol:
                    break
            eval_count_history.append(eval_count)

            # Run policy improvement
            policy_changes = 0
            for s in range(self.num_states):
                old_action = self.policy[s]
                action_values = [
                    np.sum(self.transition_model[s, a] *
                           (self.reward_function + self.gamma * self.values))
                    for a in range(self.num_actions)
                ]
                self.policy[s] = np.argmax(action_values)
                if old_action != self.policy[s]:
                    policy_changes += 1
            policy_change_history.append(policy_changes)

            # Stop if the policy is stable
            if policy_changes == 0:
                break

        print("Training complete. Optimal policy found.")

        # Plot training graphs if requested
        if plot:
            self.plot_training_history(eval_count_history, policy_change_history)

    def plot_training_history(self, eval_count_history, policy_change_history, save_path="value_iteration_policy_updates_sweeps.png"):
        """Plots and saves training statistics for policy evaluation and improvement."""
        epochs = range(1, len(eval_count_history) + 1)

        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, dpi=150)

        # Plot policy evaluation sweeps
        axes[0].plot(epochs, eval_count_history, marker='o', color='green', label='Policy Evaluation Sweeps')
        axes[0].set_ylabel('Number of Sweeps')  # Meaningful label for y-axis
        axes[0].legend()

        # Plot policy improvement changes
        axes[1].plot(epochs, policy_change_history, marker='o', color='red', label='Policy Updates')
        axes[1].set_xlabel('Epochs')  # x-axis for both plots
        axes[1].set_ylabel('Number of Updates')  # Meaningful label for y-axis
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path)  # Save the figure instead of showing it
        print(f"Graph saved to {save_path}")


    def test_policy(self):
        """Test the trained policy on the grid game."""
        position = self.grid_game.start_position
        steps = 0
        while position != self.grid_game.goal_position and steps < 100:
            state = self.grid_game.get_state_from_pos(position)
            action = self.policy[state]
            dr, dc = self.grid_game.actions[action]
            new_position = (
                max(0, min(position[0] + dr, self.grid_game.grid_size - 1)),
                max(0, min(position[1] + dc, self.grid_game.grid_size - 1)),
            )
            if self.grid_game.D[new_position[1], new_position[0]] > 0.8:
                print(f"Hit a death zone at {new_position}, resetting to start.")
                position = self.grid_game.start_position
            else:
                position = new_position
            steps += 1
            self.grid_game.player_position = position
            self.grid_game.draw_player()

            # this root.update() is to show the learned path by running q-learning
            self.grid_game.root.update()
            self.grid_game.check_position()
            print(f"Step {steps}: {position}")
            time.sleep(0.2)
