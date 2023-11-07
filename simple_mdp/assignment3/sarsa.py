from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from grid_world import GridWorld
from graphic_display import GraphicDisplay

class Sarsa:
    
    def __init__(self, gridworld, gamma, alpha, episodes):
        self.gridworld = gridworld
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = episodes
        size = gridworld.size
        self.q_table =  self.q_table = np.zeros((4,) + size) 
        self.eps = 0.9
        self.episode = 0
        self.sum_rewards = []
        self.path = []
    
    def update(self, cell, action, reward, next_cell, next_action):
        r_t, c_t = cell
        r_tp1, c_tp1 = next_cell  
        q_current_step = self.q_table[action, r_t, c_t] 
        q_next_step = self.q_table[next_action, r_tp1, c_tp1]
        error_term = reward + self.gamma * q_next_step - q_current_step    # TD Error
        self.q_table[action, r_t, c_t] = q_current_step + self.alpha * (error_term)
    
    def choose_action(self, cell):
        r, c = cell
        if np.random.random() > (1 - self.eps):
            action = np.argmax(self.q_table[:, r, c])
        else:
            action = np.random.randint(low=0, high=4)
        return action
    
    def anneal_epsilon(self):
        self.eps = max(0, self.eps * (1 - self.episode / self.episodes * 1.5))
    
    def one_episode(self):  
        first_step = True
        cntr = 0
        self.gridworld.reset()
        self.sum_rewards.append(0)

        while not self.gridworld.in_terminal() and cntr < 5000:
            cntr += 1
            cell = self.gridworld.current_cell
            
            if first_step == True:
                action = self.choose_action(cell)
                first_step = False
            else:
                action = next_action

            reward = self.gridworld.reward(cell, action)
            next_cell = self.gridworld.transition(cell, action) 
            next_action = self.choose_action(next_cell)
            self.update(cell, action, reward, next_cell, next_action)
            self.sum_rewards[-1] += reward

        print("Total Epi: {0: 5} Episode Steps: {1: 5} Reward: {2: 5.4f} ".format(
                    self.episode, cntr, self.sum_rewards[-1]))

        self.episode += 1
        self.anneal_epsilon
    
    def trajectory(self):
        self.gridworld.reset()
        self.path = []
        sum_rewards = 0
        itr = 0
        while not self.gridworld.in_terminal() and itr < 20:
            r, c = self.gridworld.current_cell
            action = np.argmax(self.q_table[:, r, c])
            sum_rewards += self.gridworld.reward((r, c), action)
            self.gridworld.transition((r, c), action)
            itr += 1
            self.path.append((r, c))
        return sum_rewards

    def is_learning_finished(self):
        return self.episode > self.episodes

def plot_learning_curve(ql):
    values = ql.sum_rewards
    x = list(range(len(values)))
    y = values
    plt.plot(x, y, 'ro')
    plt.show()

if __name__ == "__main__":
    size = (5, 5)
    start_cell = (0, 0)
    obstacles = [(2, 1)]
    terminating_state = (4, 4)
    gamma = 0.9
    alpha = 0.1
    episodes = 220

    gw = GridWorld(size, start_cell, obstacles, terminating_state)
    solver = Sarsa(gw, gamma, alpha, episodes)

    while not solver.is_learning_finished():
        solver.one_episode()
        sum_rewards = solver.sum_rewards[-1]
    
    print()
    print("LEFT, UP, RIGHT, DOWN")
    count = 1
    name_action = ["left", "up", "right", "down"]
    for i in range (5):
        for k in range (5):
            action = name_action[np.argmax(solver.q_table[:,i,k])]
            print("state:", count, solver.q_table[:,i,k], action)
            count += 1

    sum_rewards = solver.trajectory()
    plot_learning_curve(solver)
    grid_world = GraphicDisplay(solver.q_table, 2)
    grid_world.mainloop()