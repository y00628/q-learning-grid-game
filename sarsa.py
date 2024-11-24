import numpy as np
import random
import matplotlib.pyplot as plt


def train_sarsa(grid_game, learning_rate = 0.05, discount_factor = 0.95, exploration_prob = 1.0, exploration_decay = 0.995, \
        min_exploration_prob = 0.1, num_episodes = 5000, reached_goal = False, max_steps = 64):
    """Trains the agent using State-Action-Reward-State-Action model."""
    
    rewards_per_episode = []
    # save q_table to the GridGame class o that we can access it from test_sarsa
    grid_game.q_table = np.random.uniform(low=0, high=0.01, size=(grid_game.grid_size, grid_game.grid_size, 4))

    grid_game.reached_goal = reached_goal
    
    for episode in range(num_episodes):
        # Initialization
        grid_game.reset_player_position()
        visited_count = np.zeros((grid_game.grid_size, grid_game.grid_size), dtype=int)
        state = grid_game.player_position
        step_count = 0
        total_reward = 0
        
        # Choose action a based on Q(s,a). Method used: epsilon-greedy
        if random.uniform(0, 1) < exploration_prob:
            action_index = random.randint(0, 3)
        else:
            max_q_value = np.max(grid_game.q_table[state[1], state[0], :])
            # Choose randomly among the best actions in case there is a tie
            best_actions = np.where(grid_game.q_table[state[1], state[0], :] == max_q_value)[0]
            action_index = np.random.choice(best_actions)
        action = grid_game.actions[action_index]
        
        while not grid_game.game_over and step_count < max_steps:
            step_count += 1

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
                total_reward -= 10
                

            # Penalize for visted state (to prevent just going back and forth)
            # TODO need to confirm whether it is okay to add this part
            visited_count[new_state[1], new_state[0]] += 1
            if visited_count[new_state[1], new_state[0]] > 1:
                reward -= (0.5 * visited_count[new_state[1], new_state[0]])
                
            # Choose action a based on Q(s',a'). Method used: epilson-greedy method
            if random.uniform(0, 1) < exploration_prob:
                new_action_index = random.randint(0, 3)
            else:
                max_q_value = np.max(grid_game.q_table[new_state[1], new_state[0], :])
                # Choose randomly among the best actions in case there is a tie
                best_actions = np.where(grid_game.q_table[new_state[1], new_state[0], :] == max_q_value)[0]
                new_action_index = np.random.choice(best_actions)
            new_action = grid_game.actions[new_action_index]

            # Update Q-value
            grid_game.q_table[state[1], state[0], action_index] += learning_rate * (
                    reward + discount_factor * grid_game.q_table[new_state[1], new_state[0], new_action_index] - grid_game.q_table[state[1], state[0], action_index])

            # Update GUI to show agent movement
            grid_game.player_position = new_state
            grid_game.draw_player()
            # NOTE Commenting this line makes training faster
            #      If you want to see the training process, uncomment the line below.
            #      But it will make training slow
            grid_game.root.update()

            state = new_state
            action_index = new_action_index
            action = new_action
            total_reward += reward

            # Check if reached goal
            if state == grid_game.goal_position:
                print('Goal reached during training')
                test_sarsa(grid_game)
                break

        # if num_goal_reached >= self.max_goal_to_reach:
        #     break
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
    plt.title("SARSA Training Progress")
    plt.savefig("sarsa training progress.png", dpi=300)
    plt.show()

def test_sarsa(grid_game, is_training=True):
    """Tests the trained SARSA agent."""
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
        if not is_training:
            # this root.update() is to show the learned path by running q-learning
            grid_game.root.update()
        grid_game.check_position()
