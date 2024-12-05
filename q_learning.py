import numpy as np
import random
import matplotlib.pyplot as plt


def train_q_learning(grid_game):
    """Trains the agent using Q-learning."""
    rewards_per_episode = []
    # Parameters for updating the Q-values
    learning_rate = 0.05
    discount_factor = 0.95
    exploration_prob = 1.0
    exploration_decay = 0.995
    min_exploration_prob = 0.1

    num_episodes = 5000
    grid_game.reached_goal = False
    max_steps = 64
    for episode in range(num_episodes):
        grid_game.reset_player_position()
        visited_count = np.zeros((grid_game.grid_size, grid_game.grid_size), dtype=int)
        state = grid_game.player_position
        step_count = 0
        total_reward = 0
        while not grid_game.game_over and step_count < max_steps:
            step_count += 1
            # Choose action (epsilon-greedy)
            if random.uniform(0, 1) < exploration_prob:
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
            # Another way to stop the game by setting the max step threshold since our grid is 8*8
            if step_count >= max_steps:
                grid_game.game_over = True
                total_reward -= 10

            # Penalize for visted state (to prevent just going back and forth)
            visited_count[new_state[1], new_state[0]] += 1
            if visited_count[new_state[1], new_state[0]] > 1:
                reward -= (0.5 * visited_count[new_state[1], new_state[0]])

            # Update Q-value (assuming the optimal action is chosen --> which is why it requires fewer epochs than SARSA)
            best_future_q = np.max(grid_game.q_table[new_state[1], new_state[0], :])
            grid_game.q_table[state[1], state[0], action_index] += learning_rate * (
                    reward + discount_factor * best_future_q - grid_game.q_table[state[1], state[0], action_index])

            # Update GUI to show agent movement
            grid_game.player_position = new_state
            grid_game.draw_player()
            # NOTE Commenting this line makes training faster
            #      If you want to see the training process, uncomment the line below.
            #      But it will make training slow
            # grid_game.root.update()

            state = new_state
            total_reward += reward

            # Check if reached goal
            if state == grid_game.goal_position:
                print('Goal reached during training')
                # Test the most recent policy to see if it can reach the goal.
                # If not, train more episodes
                test_q_learning(grid_game)
                break

        if grid_game.reached_goal:
            break

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")
            print(exploration_prob)

        # Decay exploration prob
        exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)
        rewards_per_episode.append(total_reward)

        grid_game.game_over = False

    # Plot rewards
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Training Progress")
    plt.savefig("figures/q-learning training progress.png", dpi=300)
    plt.show()


def test_q_learning(grid_game, is_training=True):
    """Tests the trained Q-learning agent."""
    grid_game.reset_game()
    state = grid_game.player_position
    grid_game.total_reward = 0
    while not grid_game.game_over:
        action_index = np.argmax(grid_game.q_table[state[1], state[0], :])
        action = grid_game.actions[action_index]
        print('action: ', action)
        new_state = (max(0, min(state[0] + action[0], grid_game.grid_size - 1)),
                        max(0, min(state[1] + action[1], grid_game.grid_size - 1)))
        state = new_state
        print('state: ', state)
        grid_game.player_position = state
        grid_game.draw_player()
        if not is_training:
            # this root.update() is to show the learned path by running q-learning
            grid_game.root.update()
        grid_game.check_position()



