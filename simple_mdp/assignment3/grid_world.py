class GridWorld:
    def __init__(self, size, start_cell, obstacles, terminating_state):
        self.size = size
        self.start = start_cell
        self.obstacles = obstacles
        self.termin = terminating_state
        self.current_cell = self.start
    
    def reset(self):
        self.current_cell = self.start
    
    def transition(self, cell, action):

        r_current = cell[0]
        c_current = cell[1]
        
        if action == 0:
          cell = (r_current,c_current-1)
        if action == 1:
          cell = (r_current-1,c_current)
        if action == 2:
          cell = (r_current,c_current+1)
        if action == 3:
          cell = (r_current+1,c_current)

        if (cell[0], cell[1]) in self.obstacles:
          cell = (r_current,c_current)

        if cell[0] < 0 \
             or cell[0] > self.size[0] -1 \
             or cell[1] < 0  \
             or cell[1] > self.size[1] -1 : 
          cell = (r_current, c_current)
        
        self.current_cell = cell
        
        return cell

    def reward(self, cell, action):
        cell_state = cell
        if self.transition(cell, action) != self.termin:
          reward = -1
        else:
          reward = 0
        self.current_cell = cell_state
        return reward

    def in_terminal(self):
        return self.current_cell == self.termin