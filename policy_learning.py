import numpy as np
import random
import matplotlib.pyplot as plt

def get_reward_function(grid_game):
    """
    Computes a reward table for each state in the grid.
    Takes into account the reward matrix (grid_game.R), death matrix (grid_game.D),
    and additional reward dynamics.
    """
    # Initialize the reward table
    reward_table = np.zeros(grid_game.grid_size * grid_game.grid_size)
    
    for r in range(grid_game.grid_size):
        for c in range(grid_game.grid_size):
            # Get the state index from the position
            state_index = grid_game.get_state_from_pos((r, c))
            
            # Base reward from the reward matrix
            base_reward = grid_game.R[r, c]
            
            # Death penalty from the death matrix
            death_penalty = -2 if grid_game.D[r, c] > 0.5 else 0  # Heavy penalty if death probability is high
            
            # Slight decrease in reward based on position (encourages exploration)
            # distance_from_goal = abs(grid_game.goal_position[0] - r) + abs(grid_game.goal_position[1] - c)
            # exploration_penalty = 0.1 * distance_from_goal  # Reduce slightly as distance increases
            
            # Calculate the final reward for the state
            reward_table[state_index] = base_reward + death_penalty #- exploration_penalty
    
    return reward_table

def get_transition_model(grid_game, random_rate=0.001):
    """
    Constructs a transition model for the grid game.
    Transition probabilities account for random movements (with probability `random_rate`) and deterministic behavior.
    """
    num_states = grid_game.grid_size * grid_game.grid_size
    num_actions = len(grid_game.actions)
    transition_model = np.zeros((num_states, num_actions, num_states))
    
    for r in range(grid_game.grid_size):
        for c in range(grid_game.grid_size):
            # Get the current state index
            current_state = grid_game.get_state_from_pos((r, c))
            
            # Neighboring states for each action
            neighbor_states = np.zeros(num_actions)
            
            for a, (dr, dc) in enumerate(grid_game.actions):
                new_r, new_c = r + dr, c + dc
                
                # Boundary handling: ensure within grid limits
                if 0 <= new_r < grid_game.grid_size and 0 <= new_c < grid_game.grid_size:
                    # Check if the new state is a death square
                    if grid_game.D[new_r, new_c] > 0.5:
                        new_r, new_c = r, c  # Stay in the same position if it's a death square
                else:
                    # If out of bounds, stay in the same position
                    new_r, new_c = r, c
                
                # Map (new_r, new_c) to the corresponding state index
                neighbor_states[a] = grid_game.get_state_from_pos((new_r, new_c))
            
            # Assign probabilities for each action
            for a in range(num_actions):
                # Deterministic transition to intended neighbor
                transition_model[current_state, a, int(neighbor_states[a])] += 1 - random_rate
                
                # Random transition to adjacent states
                transition_model[current_state, a, int(neighbor_states[(a + 1) % num_actions])] += random_rate / 2.0
                transition_model[current_state, a, int(neighbor_states[(a - 1) % num_actions])] += random_rate / 2.0

    # Store the transition model in grid_game.P for future use
    grid_game.P = transition_model
    return transition_model


class PolicyIteration:
    def __init__(self, reward_function, transition_model, gamma, init_policy=None, num_states=None, num_actions=None):
        """
        Initializes the PolicyIteration class.

        Parameters:
        - reward_function: A 1D numpy array representing the reward for each state.
        - transition_model: A 3D numpy array with dimensions (num_states, num_actions, num_states)
                            representing the transition probabilities.
        - gamma: Discount factor (float between 0 and 1).
        - init_policy: Optional initial policy as a 1D numpy array of actions for each state.
        - num_states: Number of states in the environment (int).
        - num_actions: Number of actions available (int).
        """
        self.num_states = num_states if num_states is not None else transition_model.shape[0]
        self.num_actions = num_actions if num_actions is not None else transition_model.shape[1]
        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma

        # Initialize policy: either provided or random
        self.policy = init_policy if init_policy is not None else np.random.randint(0, self.num_actions, self.num_states)

        # Initialize value function for each state
        self.values = np.zeros(self.num_states)

    def one_policy_evaluation(self):
        """
        Performs one sweep of policy evaluation, updating the value of each state based on the current policy.
        """
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            a = self.policy[s]
            p = self.transition_model[s, a]
            self.values[s] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def run_policy_evaluation(self, tol=1e-3):
        """
        Repeatedly performs policy evaluation until the value function converges (change < tol).
        """
        epoch = 0
        delta = self.one_policy_evaluation()
        delta_history = [delta]
        while epoch < 100:  # Limit iterations to avoid infinite loops
            delta = self.one_policy_evaluation()
            delta_history.append(delta)
            if delta < tol:
                break
            epoch += 1
        return len(delta_history)

    def run_policy_improvement(self):
        """
        Updates the policy based on the current value function.
        Returns the number of states for which the policy has changed.
        """
        update_policy_count = 0
        for s in range(self.num_states):
            temp = self.policy[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = np.sum(p * self.values)
            self.policy[s] = np.argmax(v_list)
            if temp != self.policy[s]:
                update_policy_count += 1
        return update_policy_count

    def train(self, tol=1e-3, plot=True):
        """
        Trains the policy using Policy Iteration (alternating policy evaluation and improvement).
        """
        epoch = 0
        eval_count = self.run_policy_evaluation(tol=tol)
        eval_count_history = [eval_count]
        policy_change = self.run_policy_improvement()
        policy_change_history = [policy_change]
        while epoch < 500:  # Limit epochs
            epoch += 1
            new_eval_count = self.run_policy_evaluation(tol)
            new_policy_change = self.run_policy_improvement()
            eval_count_history.append(new_eval_count)
            policy_change_history.append(new_policy_change)
            if new_policy_change == 0:  # Stop when policy converges
                break

        print(f"# Epochs: {len(policy_change_history)}")
        print(f"Policy Evaluation Sweeps: {eval_count_history}")
        print(f"Policy Updates: {policy_change_history}")

        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(4, 5), sharex='all', dpi=150)
            axes[0].plot(eval_count_history, marker='o', markersize=4, color='#2ca02c', alpha=0.8)
            axes[0].set_title("Policy Evaluation Sweeps")
            axes[0].set_ylabel("Sweeps")

            axes[1].plot(policy_change_history, marker='o', markersize=4, color='#d62728', alpha=0.8)
            axes[1].set_title("Policy Updates")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Updates")

            plt.tight_layout()
            plt.show()

def train_policy_iteration(grid_game, gamma=0.99, tol=1e-3, random_rate=0.5, plot=True):
    """
    Trains a policy using Policy Iteration for the provided GridGame instance.

    Parameters:
    - grid_game: Instance of the GridGame class.
    - gamma: Discount factor (float).
    - tol: Convergence tolerance for policy evaluation.
    - random_rate: Probability of random movement in the transition model.
    - plot: Whether to plot training statistics (boolean).
    """
    # Generate the reward function and transition model
    reward_function = get_reward_function(grid_game)
    transition_model = get_transition_model(grid_game, random_rate=random_rate)

    # Initialize Policy Iteration solver
    solver = PolicyIteration(
        reward_function=reward_function,
        transition_model=transition_model,
        gamma=gamma,
        num_states=grid_game.grid_size * grid_game.grid_size,
        num_actions=len(grid_game.actions)
    )

    # Train the policy
    solver.train(tol=tol, plot=plot)

    # Update the grid game's policy
    grid_game.policy = solver.policy.reshape(grid_game.grid_size, grid_game.grid_size)

    print("Policy Iteration training complete!")
    print("Optimal Policy:")
    print(grid_game.policy)


def test_policy_iteration(grid_game, is_training=True):
    """Tests the trained Policy Iteration agent."""
    grid_game.reset_game()
    state = grid_game.player_position
    grid_game.total_reward = 0
    
    while not grid_game.game_over:
        # Use the trained policy to select the best action for the current state
        action_index = grid_game.policy[state[1], state[0]]  # Using policy instead of Q-table
        action = grid_game.actions[action_index]
        
        print('action: ', action)
        
        # Update the state based on the chosen action, ensuring the player stays within grid bounds
        new_state = (max(0, min(state[0] + action[0], grid_game.grid_size - 1)),
                     max(0, min(state[1] + action[1], grid_game.grid_size - 1)))
        
        state = new_state
        print('state: ', state)
        
        grid_game.player_position = state
        grid_game.draw_player()  # Re-render the player at the new position
        
    
            # This updates the UI to show the agent's movement and the learned path
        grid_game.root.update()
        
        # Check for any game over conditions
        grid_game.check_position()