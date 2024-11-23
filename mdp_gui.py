import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from q_learning import test_q_learning, train_q_learning
from policy_learning import train_policy_iteration, test_policy_iteration


class GridGame:
    def __init__(self, root, fixed_path=False, path_death_prob=0.3, max_reward_prob=0.3):
        self.root = root
        self.root.title("8x8 Grid Game")

        # Game settings
        self.grid_size = 8
        self.start_position = (0, 7)
        self.goal_position = (7, 0)
        self.total_reward = 0
        self.current_reward = 0
        self.game_over = False
        self.max_reward_prob = max_reward_prob
        self.fixed_path = fixed_path
        self.path_death_prob = path_death_prob
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

        # Version and color map selection
        self.version = tk.IntVar(value=1)
        self.show_reward_map = tk.BooleanVar(value=True)

        # Reward and death matrices
        self.R = (np.random.rand(self.grid_size, self.grid_size) * 0.3 + 0.7) * max_reward_prob  # Low reward probability
        self.D = self.create_death_matrix(fixed=fixed_path, path_death_prob=path_death_prob)  # Custom death matrix with a complex path
        self.P =   np.zeros((self.grid_size ** 2, len(self.actions), self.grid_size ** 2))#transition matrix
        # Player position initialization
        print(self.D)
        print(self.R)
        self.reset_player_position()

        # Game UI setup
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()
        self.cell_size = 400 // self.grid_size
        self.canvas.bind("<Button-1>", self.set_start_position)  # Bind left-click for setting start position in Version 2

        # Control panel for version and color map selection
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=5)

        # Display reward information
        self.current_reward_label = tk.Label(control_frame, text="Current Reward: 0")
        self.current_reward_label.grid(row=0, column=0, sticky="w")
        self.total_reward_label = tk.Label(control_frame, text="Cumulative Reward: 0")
        self.total_reward_label.grid(row=1, column=0, sticky="w")

        # End game message
        self.end_message_label = tk.Label(control_frame, text="", font=("Arial", 12, "bold"))
        self.end_message_label.grid(row=2, column=0, columnspan=3, sticky="w")

        # Version selection radio buttons
        tk.Label(control_frame, text="Select Game Version:").grid(row=3, column=0, columnspan=3, sticky="w")
        versions = [("Version 1: Show probabilities", 1), ("Version 2: Random start position", 2), ("Version 3: Fixed start position", 3)]
        for i, (text, val) in enumerate(versions):
            tk.Radiobutton(control_frame, text=text, variable=self.version, value=val, command=self.change_version).grid(row=i + 4, column=0, columnspan=3, sticky="w")

        # Toggle for reward/death probability map
        self.map_toggle = tk.Checkbutton(control_frame, text="Showing Reward Probability Map", variable=self.show_reward_map, command=self.update_map_label)
        self.map_toggle.grid(row=7, column=0, sticky="w")

        self.map_text = tk.Label(control_frame, text='Deeper color means higher reward/death probability')
        self.map_text.grid(row=8, column=0, sticky="w")

        # Restart button
        self.restart_button = tk.Button(control_frame, text="Restart Game (space key)", command=self.reset_game)
        self.restart_button.grid(row=9, column=0, pady=5)

        #I intialize it here just for ease of use
        self.q_table = np.random.uniform(low=0, high=0.01, size=(self.grid_size, self.grid_size, 4))
        self.policy = np.random.randint(0, len(self.actions), (self.grid_size, self.grid_size), dtype=int)


        self.setup_grid()
        self.draw_player()

        # Bind keyboard events
        self.root.bind("<Up>", lambda _: self.move_player(0, -1))
        self.root.bind("<Down>", lambda _: self.move_player(0, 1))
        self.root.bind("<Left>", lambda _: self.move_player(-1, 0))
        self.root.bind("<Right>", lambda _: self.move_player(1, 0))
        self.root.bind("<space>", lambda _: self.reset_game())
        self.root.bind("<q>", lambda _: train_q_learning(self))
        self.root.bind("<t>", lambda _: test_q_learning(self, is_training=False))
        self.root.bind("<r>", lambda _: self.change_map())

        self.root.bind("<p>", lambda _: train_policy_iteration(self))  # Train policy using policy iteration
        self.root.bind("<y>", lambda _: test_policy_iteration(self))

    def create_death_matrix(self, fixed=False, path_death_prob=0.3):
        """Creates a complex path with an specific overall death probability from start to goal."""
        # D = np.ones((self.grid_size, self.grid_size)) * 0.9  # Set high death probabilities for all cells
        D = np.ones((self.grid_size, self.grid_size)) * 0.5 + 0.5  # Set high death probabilities [0.5, 1] for all cells
        
        if fixed:
            # Define a path with cells that will have lower death probabilities
            path = [
                (0, 7), (1, 7), (2, 7), (2, 6), (2, 5),
                (3, 5), (4, 5), (5, 5), (5, 4), (5, 3),
                (6, 3), (7, 3), (7, 2), (7, 1), (7, 0)
            ]
            # Add some "branching" dead ends with lower probabilities to add complexity
            D[2, 4] = 0.1
            D[3, 6] = 0.1
            D[5, 6] = 0.1
            D[6, 1] = 0.1
        else:
            path = [(0, 7)]
            retry = 0
            while path[-1] != (7, 0):
                retry += 1
                cur = path[-1]
                idx = np.random.choice(4, p=[0.15, 0.35, 0.35, 0.15])
                d = [(0, 1), (0, -1), (1, 0), (-1, 0)][idx]
                new = (cur[0] + d[0], cur[1] + d[1])
                if 0 <= new[0] <= 7 and 0 <= new[1] <= 7:# and new not in path:
                    # check if form a block
                    found_block = False
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        if (new[0] + dx, new[1] + dy) in path and (new[0] + dx, new[1]) in path and (new[0], new[1] + dy) in path:
                            found_block = True
                            break
                    if not found_block:
                        path.append(new)
                        retry = 0
                if retry > 100:
                    print("Reinit...")
                    path = [(0, 7)]
                    retry = 0
 
        # Set lower death probabilities along this path
        each_death_prob = 1 - np.exp(np.log(1 - path_death_prob) / len(path))
        print(f"each_death_prob={each_death_prob}")
        for coord in path:
            D[coord] = each_death_prob  # Moderate death probability for path cells

        return D

    def set_start_position(self, event):
        """Sets the start position based on a click in Version 2 only."""
        if self.version.get() == 2:
            # Calculate the grid cell from the click coordinates
            col = event.x // self.cell_size
            row = event.y // self.cell_size
            if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
                self.player_position = (col, row)  # Update player's position to clicked cell
                self.draw_player()  # Redraw the player at the new position
            self.reset_game(reset_position=False)

    def update_map_label(self):
        """Updates the map toggle label and redraws the grid based on the selected map."""
        if self.show_reward_map.get():
            self.map_toggle.config(text="Showing Reward Probability Map")
        else:
            self.map_toggle.config(text="Showing Death Probability Map")
        self.setup_grid()  # Redraw the grid with the new map selection
        self.draw_player()  # Ensure the player position is shown after map update

    def random_start_position(self):
        """Generates a random starting position for the player."""
        return (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

    def reset_player_position(self):
        """Sets the player position based on the selected version."""
        if self.version.get() == 2:
            self.player_position = self.random_start_position()
        else:
            self.player_position = self.start_position

    def setup_grid(self):
        """Draws the grid, optionally showing probabilities based on user choice."""
        r_range = (np.min(self.R.flatten()), np.max(self.R.flatten()))
        r_sz = r_range[1] - r_range[0]
        d_range = (np.min(self.D.flatten()), np.max(self.D.flatten()))
        d_sz = d_range[1] - d_range[0]
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                # Display color map for Version 1 only
                if self.version.get() == 1:
                    if self.show_reward_map.get():
                        color_intensity = (self.R[row, col] - r_range[0]) / r_sz
                        color = plt.cm.Purples(color_intensity)
                        color = cl.rgb2hex(color)
                    else:
                        color_intensity = (self.D[row, col] - d_range[0]) / d_sz
                        color = plt.cm.Reds(color_intensity)
                        color = cl.rgb2hex(color)
                else:
                    color = "white"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

    def draw_player(self):
        """Draws or updates the player's position on the grid."""
        self.canvas.delete("player")
        x, y = self.player_position
        x1 = x * self.cell_size
        y1 = y * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_oval(x1, y1, x2, y2, fill="blue", tags="player")

    def move_player(self, dx, dy):
        """Moves the player in the specified direction, checks for rewards or death."""
        if self.game_over:
            return

        # Calculate new position
        new_x = self.player_position[0] + dx
        new_y = self.player_position[1] + dy

        # Ensure the move is within bounds
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            self.player_position = (new_x, new_y)
            self.draw_player()
            self.check_position()

    def check_position(self):
        """Checks for rewards or death at the current player position."""
        x, y = self.player_position
        reward_prob = self.R[y, x]
        death_prob = self.D[y, x]

        # Reward sampling
        self.current_reward = np.random.binomial(1, reward_prob)
        self.total_reward += self.current_reward
        self.update_reward_labels()

        # Death sampling
        death = np.random.binomial(1, death_prob)

        if death:
            self.end_game(False)
        elif self.player_position == self.goal_position:
            self.total_reward += 10  # Large reward for reaching the goal
            self.update_reward_labels()
            self.end_game(True)

    def update_reward_labels(self):
        """Updates the current and cumulative reward labels on the UI."""
        self.current_reward_label.config(text=f"Current Reward: {self.current_reward}")
        self.total_reward_label.config(text=f"Cumulative Reward: {self.total_reward}")

    def end_game(self, reached_goal):
        """Ends the game and displays a message in the control panel."""
        self.game_over = True
        self.reached_goal = reached_goal
        if reached_goal:
            message = "Congratulations! You reached the goal!"
            self.end_message_label.config(text=f"{message}", fg="green")
        else:
            message = "Game Over! You died!"
            self.end_message_label.config(text=f"{message}", fg="red")

    def reset_game(self, reset_position=True):
        """Resets the game state for a new round."""
        if reset_position:
            self.reset_player_position()  # Set player start position based on the version
        self.total_reward = 0
        self.current_reward = 0
        self.game_over = False
        self.update_reward_labels()
        self.end_message_label.config(text="")  # Clear the end message
        self.setup_grid()  # Redraw the grid to reflect the map and version selection
        self.draw_player()  # Ensure the player position is shown after reset

    def change_version(self):
        """Handles version change and redraws the grid and player position."""
        self.reset_game()  # Reset game for the new version
        self.draw_player()  # Ensure the player position is shown after version change

    def change_map(self):
        self.R = (np.random.rand(self.grid_size, self.grid_size) * 0.3 + 0.7) * self.max_reward_prob  # Low reward probability
        self.D = self.create_death_matrix(fixed=self.fixed_path, path_death_prob=self.path_death_prob)  # Custom death matrix with a complex path
        self.reached_goal = False
        self.reset_game()
        # self.root.update()

    def get_state_from_pos(self, cords):
        """
        Converts a 2D grid position (row, col) into a 1D state index.
        Accounts for flipped y-axis between the death matrix and the game grid.
        """
        row, col = cords
        # No flipping of row indices needed for proper correspondence
        return row * self.grid_size + col
    def if_death(self, x, y):
        """Checks if the given position (x, y) is a death state."""
        death_prob = self.D[y, x]  # Access the death probability for the given coordinates
        death = np.random.binomial(1, death_prob)  # Simulate the death probability
        return death == 1




    # def create_transition(self):
    #     """Creates the transition probability matrix for the grid."""

    #     grid_size = self.grid_size
    #     num_states = grid_size * grid_size
    #     num_actions = len(self.actions)

    #     # Initialize transition probabilities
    #     self.P = np.zeros((num_states, num_states, num_actions))

    #     for x in range(grid_size):
    #         for y in range(grid_size):
    #             current_state = x * grid_size + y  # Flattened state index (x, y)

    #             for action_index, action in enumerate(self.actions):
    #                 # Compute new state based on action
    #                 new_x = max(0, min(x + action[0], grid_size - 1))
    #                 new_y = max(0, min(y + action[1], grid_size - 1))
    #                 new_state = new_x * grid_size + new_y  # Flattened new state index

    #                 # Handle death zones
    #                 if self.D[new_x, new_y]:  # Death matrix probability
    #                     self.P[new_state, current_state, action_index] = self.D[new_x, new_y]
    #                     self.P[current_state, current_state, action_index] += (1 - self.D[new_x, new_y])
    #                 else:
    #                     self.P[new_state, current_state, action_index] = 1.0

    #     # Normalize probabilities for each state-action pair
    #     for s in range(num_states):
    #         for a in range(num_actions):
    #             total_prob = np.sum(self.P[:, s, a])
    #             if total_prob > 0:
    #                 self.P[:, s, a] /= total_prob







# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    game = GridGame(root)
    root.mainloop()