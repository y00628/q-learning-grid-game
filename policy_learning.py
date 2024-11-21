import numpy as np
import random
import matplotlib.pyplot as plt

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