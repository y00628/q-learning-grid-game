import numpy as np
import random
import matplotlib.pyplot as plt


def train_q_learning(grid_game):
    """Trains the agent using Q-learning."""
    rewards_per_episode = []
    grid_game.q_table = np.random.uniform(low=0, high=0.01, size=(grid_game.grid_size, grid_game.grid_size, 4))
    grid_game.learning_rate = 0.05
    grid_game.discount_factor = 0.95
    grid_game.exploration_prob = 1.0
    grid_game.exploration_decay = 0.99
    grid_game.min_exploration_prob = 0.1
    grid_game.num_episodes = 5000
    grid_game.reached_goal = False
    for episode in range(grid_game.num_episodes):
        grid_game.reset_player_position()
        visited_count = np.zeros((grid_game.grid_size, grid_game.grid_size), dtype=int)
        state = grid_game.player_position
        max_steps = 100
        step_count = 0
        total_reward = 0
        while not grid_game.game_over and step_count < max_steps:
            step_count += 1
            # Choose action (epsilon-greedy)
            if random.uniform(0, 1) < grid_game.exploration_prob:
                action_index = random.randint(0, 3)
            else:
                max_q_value = np.max(grid_game.q_table[state[1], state[0], :])
                # Choose randomly among the best actions in case there is a tie
                best_actions = np.where(grid_game.q_table[state[1], state[0], :] == max_q_value)[0]
                action_index = np.random.choice(best_actions)
            action = grid_game.actions[action_index]

            # Take action
            new_state = (max(0, min(state[0] + action[0], grid_game.grid_size - 1)),
                            max(0, min(state[1] + action[1], grid_game.grid_size - 1)))
            
            # Get reward and check for death
            reward = np.random.binomial(1, grid_game.R[new_state[1], new_state[0]])
            death = np.random.binomial(1, grid_game.D[new_state[1], new_state[0]])

            if death:
                reward = -1
                grid_game.game_over = True
            if step_count >= max_steps:
                grid_game.game_over = True
                total_reward -= 20

            # Penalize for visted state (to prevent just going back and forth)
            # TODO need to confirm whether it is okay to add this part
            visited_count[new_state[1], new_state[0]] += 1
            if visited_count[new_state[1], new_state[0]] > 1:
                reward -= (1 * visited_count[new_state[1], new_state[0]])

            # Update Q-value
            best_future_q = np.max(grid_game.q_table[new_state[1], new_state[0], :])
            grid_game.q_table[state[1], state[0], action_index] += grid_game.learning_rate * (
                    reward + grid_game.discount_factor * best_future_q - grid_game.q_table[state[1], state[0], action_index])

            # Update GUI to show agent movement
            grid_game.player_position = new_state
            grid_game.draw_player()
            grid_game.root.update()

            state = new_state
            total_reward += reward

            # Check if reached goal
            if state == grid_game.goal_position:
                print('Goal reached during training')
                test_q_learning(grid_game)
                break

        # if num_goal_reached >= self.max_goal_to_reach:
        #     break
        if grid_game.reached_goal:
            break

        progress = int((episode + 1) / grid_game.num_episodes * 100)
        grid_game.progress_bar["value"] = progress
        grid_game.root.update()

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")
            print(grid_game.exploration_prob)

        # Decay exploration prob
        grid_game.exploration_prob = max(grid_game.min_exploration_prob, grid_game.exploration_prob * grid_game.exploration_decay)
        rewards_per_episode.append(total_reward)

        grid_game.game_over = False

    grid_game.progress_bar["value"] = 100
    grid_game.root.update()

    # Plot rewards
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Training Progress")
    plt.savefig("q-learning training progress.png", dpi=300)
    plt.show()

def test_q_learning(grid_game):
    """Tests the trained Q-learning agent."""
    grid_game.reset_game()
    state = grid_game.player_position
    grid_game.total_reward = 0
    while not grid_game.game_over:
        action_index = np.argmax(grid_game.q_table[state[1], state[0], :])
        action = grid_game.actions[action_index]
        new_state = (max(0, min(state[0] + action[0], grid_game.grid_size - 1)),
                        max(0, min(state[1] + action[1], grid_game.grid_size - 1)))
        state = new_state
        grid_game.player_position = state
        grid_game.draw_player()
        grid_game.root.update()
        grid_game.check_position()
