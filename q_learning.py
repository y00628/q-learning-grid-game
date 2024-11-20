import numpy as np
import random
import matplotlib.pyplot as plt


def train_q_learning(grid_game): # this is uses value iteration alone the process is the same
    """Trains the agent using Q-learning."""
    rewards_per_episode = []
    # save q_table to the GridGame class since we need to access it from test_q_learning
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
            if step_count >= max_steps:
                grid_game.game_over = True
                total_reward -= 10

            # Penalize for visted state (to prevent just going back and forth)
            # TODO need to confirm whether it is okay to add this part
            visited_count[new_state[1], new_state[0]] += 1
            if visited_count[new_state[1], new_state[0]] > 1:
                reward -= (0.5 * visited_count[new_state[1], new_state[0]])

            # Update Q-value
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
                test_q_learning(grid_game)
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
    plt.title("Q-Learning Training Progress")
    plt.savefig("q-learning training progress.png", dpi=300)
    plt.show()

def train_q_learning_policy(grid_game):
    policy = np.random.choice(len(grid_game.actions), size = (grid_game.grid_size, grid_game.grid_size))#random policy for entire board
    V_values = np.zeroes((grid_game.grid_size, grid_game.grid_size)) #empty v value 
    discount_factor = 0.95
    rewards_per_policy_eval = []
    stable = False
    h = 0
    while not stable:
        h += 1
        while True:
            delta_v = 0 #change in v values declaration
            for y in range(grid_game.grid_size):
                for x in range(grid_game.grid_size):#loop through all spaces
                    prev_value = V_values[y,x]
                    action_index = policy[y,x]
                    action = grid_game.actions[action_index]
                    new_state = (max(0, min(x + action[0], grid_game.grid_size - 1)),
                            max(0, min(y + action[1], grid_game.grid_size - 1))) #just took this from above tbh
                    grid_game.player_position = new_state
                    grid_game.draw_player()
                    
                    reward = grid_game.R[new_state[1], new_state[0]]
                    V_values[y,x] = reward + discount_factor * V_values[new_state[1], new_state[0]]
                    delta_v = max(delta_v, abs(prev_value = V_values[y,x]))
            if delta_v < 0.01: #random choice for threshold, it usually solves before this
                break
        

        #to improve the policy
        stable = True
        for y in range(grid_game.size):
            for x in range(grid_game.size):# im sures theres a way to vectorize but for simplicity sake
                prev_action = policy[y, x]
                action_results = []

                for action_index, action in enumerate(grid_game.actions):#enumerate the goat for loops
                    new_state = (max(0, min(x + action[0], grid_game.grid_size - 1)),
                            max(0, min(y + action[1], grid_game.grid_size - 1)))
                    
                    grid_game.player_position = new_state
                    grid_game.draw_player()
                    
                    reward = grid_game.R[new_state[1], new_state[0]]
                    action_result = reward + discount_factor * V_values[new_state[1], new_state[0]]
                    action_results.append(action_result)

                    opt_action = np.argmax(action_results)
                    policy[y,x] = opt_action

                    if prev_action != opt_action: # checking if same
                        stable = False
        total_reward = np.sum(V_values)  # Example reward calculation
        rewards_per_policy_eval.append(total_reward)
    


    # Plot rewards
    plt.plot(rewards_per_policy_eval)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning 'Policy' Training Progress")
    plt.savefig("q-learning training 'Policy' progress.png", dpi=300)
    plt.show()
    return policy
def test_q_learning(grid_game, is_training=True):
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
        if not is_training:
            # this root.update() is to show the learned path by running q-learning
            grid_game.root.update()
        grid_game.check_position()

#ripped ur idea above tbh
def test_policy(grid_game, policy):

    grid_game.reset_game()
    state = grid_game.player_position
    grid_game.total_reward = 0
    while not grid_game.game_over:
        action_index = policy[state[1], state[0]]
        action = grid_game.actions[action_index]
        new_state = (max(0, min(state[0] + action[0], grid_game.grid_size - 1)),
                        max(0, min(state[1] + action[1], grid_game.grid_size - 1)))
        state = new_state
        grid_game.player_position = state
        grid_game.draw_player()
        grid_game.root.update()
        grid_game.check_position()