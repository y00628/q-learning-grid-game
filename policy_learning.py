import numpy as np
import time
def create_reward_matrix(grid_game):
    """Creates a reward matrix based on the grid states."""
    reward_matrix = np.full((grid_game.grid_size, grid_game.grid_size), -0.04)

    # Assign rewards based on goal and death states
    reward_matrix[grid_game.goal_position] = 10  # Large reward for the goal state

    for row in range(grid_game.grid_size):
        for col in range(grid_game.grid_size):
            if grid_game.D[row, col] > 0.8:  # Threshold for a death state
                reward_matrix[row, col] = -10  # Severe penalty for death state

    return reward_matrix







def create_transition_matrix(grid_game, random_rate=0.1):
    """Creates a transition probability matrix for the entire grid."""
    num_states = grid_game.grid_size * grid_game.grid_size
    num_actions = len(grid_game.actions)
    transition_model = np.zeros((num_states, num_actions, num_states))

    start_state = grid_game.get_state_from_pos(grid_game.start_position)

    for r in range(grid_game.grid_size):
        for c in range(grid_game.grid_size):
            s = grid_game.get_state_from_pos((r, c))

            if grid_game.D[r, c] > 0.8:  # Death state
                # All transitions lead to the starting state
                for a in range(num_actions):
                    transition_model[s, a, :] = 0
                    transition_model[s, a, start_state] = 1
                continue

            # Valid state: Set transitions to neighbors
            for a, (dr, dc) in enumerate(grid_game.actions):
                new_r = max(0, min(r + dr, grid_game.grid_size - 1))
                new_c = max(0, min(c + dc, grid_game.grid_size - 1))
                s_prime = grid_game.get_state_from_pos((new_r, new_c))

                if grid_game.D[new_r, new_c] > 0.8:  # Neighbor is a death zone
                    transition_model[s, a, s_prime] = 0  # No transition into death state
                else:
                    transition_model[s, a, s_prime] += (1 - random_rate)
                    transition_model[s, a, grid_game.get_state_from_pos(
                        (max(0, min(r + grid_game.actions[(a + 1) % num_actions][0], grid_game.grid_size - 1)),
                         max(0, min(c + grid_game.actions[(a + 1) % num_actions][1], grid_game.grid_size - 1)))
                    )] += (random_rate / 2.0)
                    transition_model[s, a, grid_game.get_state_from_pos(
                        (max(0, min(r + grid_game.actions[(a - 1) % num_actions][0], grid_game.grid_size - 1)),
                         max(0, min(c + grid_game.actions[(a - 1) % num_actions][1], grid_game.grid_size - 1)))
                    )] += (random_rate / 2.0)

            # Normalize probabilities for each action
            for a in range(num_actions):
                total_prob = np.sum(transition_model[s, a, :])
                if total_prob > 0:
                    transition_model[s, a, :] /= total_prob

    return transition_model










def train_policy_iteration(grid_game, discount_factor=0.95, theta=1e-6):
    """Performs policy iteration to find the optimal policy."""
    num_states = grid_game.grid_size * grid_game.grid_size
    num_actions = len(grid_game.actions)

    rewards = create_reward_matrix(grid_game).flatten()
    transition_model = create_transition_matrix(grid_game)

    # Initialize random policy and value function
    policy = np.zeros(num_states, dtype=int)
    value_function = np.zeros(num_states)

    is_policy_stable = False
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(num_states):
                v = value_function[s]
                a = policy[s]
                value_function[s] = sum(
                    transition_model[s, a, s_prime] *
                    (rewards[s_prime] + discount_factor * value_function[s_prime])
                    for s_prime in range(num_states)
                )
                delta = max(delta, abs(v - value_function[s]))
            if delta < theta:
                break

        # Policy Improvement
        is_policy_stable = True
        for s in range(num_states):
            old_action = policy[s]
            action_values = [
                sum(
                    transition_model[s, a, s_prime] *
                    (rewards[s_prime] + discount_factor * value_function[s_prime])
                    for s_prime in range(num_states)
                )
                for a in range(num_actions)
            ]
            policy[s] = np.argmax(action_values)
            if old_action != policy[s]:
                is_policy_stable = False

    grid_game.policy = policy.reshape((grid_game.grid_size, grid_game.grid_size))
    print(grid_game.policy)






def test_policy_iteration(grid_game):
    """Tests the policy obtained from policy iteration."""
    position = grid_game.start_position
    total_reward = 0
    steps = 0
    visited_positions = set()

    while position != grid_game.goal_position and steps < 100:
        state = grid_game.get_state_from_pos(position)
        action = grid_game.policy[state // grid_game.grid_size, state % grid_game.grid_size]
        dr, dc = grid_game.actions[action]
        new_position = (
            max(0, min(position[0] + dr, grid_game.grid_size - 1)),
            max(0, min(position[1] + dc, grid_game.grid_size - 1))
        )

        if new_position in visited_positions:
            total_reward -= 0.1  # Penalize revisiting states
        visited_positions.add(new_position)

        position = new_position
        total_reward += grid_game.R[position[1], position[0]] - 0.04
        steps += 1

        grid_game.player_position = position
        grid_game.draw_player()
        time.sleep(0.5)
        grid_game.root.update()
        grid_game.check_position()
        print(f"Step {steps}, Position: {position}, Total Reward: {total_reward}")

    return total_reward, position == grid_game.goal_position


