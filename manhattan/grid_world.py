import numpy as np

class GridWorld:
    def __init__(self, size, start_cell, obstacles, goal_state):
        self.size = size
        self.start = start_cell
        self.obstacles = obstacles
        self.goal_state = goal_state
        self.current_cell = self.start
        self.state = np.zeros([size[0],size[1]])
        self.state_temp = np.zeros([size[0],size[1]])

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.state[i][j]=-20
        
        self.fix_value()
    
    def fix_value(self):
        self.state[self.goal_state]=0
        for i in range(14):
            self.state[self.obstacles[i]]=-20

    def final_value(self):
        self.state[self.goal_state]=0
        for i in range(14):
            self.state[self.obstacles[i]]=20
    
    def update_value(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                r_current = i
                c_current = j

                value = self.state[i,j]
                
                if (value > 0):
                    self.state_temp[i,j] = value
                
                else:
                    cell = (r_current,c_current-1)
                    if self.check_boundary(cell):
                        if self.state[cell] > value:
                          value = self.state[cell] 

                    cell = (r_current-1,c_current)
                    if self.check_boundary(cell):
                        if self.state[cell] > value:
                          value = self.state[cell] 

                    cell = (r_current,c_current+1)
                    if self.check_boundary(cell):
                        if self.state[cell] > value:
                          value = self.state[cell] 

                    cell = (r_current+1,c_current)
                    if self.check_boundary(cell):
                        if self.state[cell] > value:
                          value = self.state[cell] 

                    cell = (r_current-1,c_current-1)
                    if self.check_boundary(cell):
                        if self.state[cell] > value:
                          value = self.state[cell] 

                    cell = (r_current-1,c_current+1)
                    if self.check_boundary(cell):
                        if self.state[cell] > value:
                          value = self.state[cell] 

                    cell = (r_current+1,c_current+1)
                    if self.check_boundary(cell):
                        if self.state[cell] > value:
                          value = self.state[cell] 

                    cell = (r_current+1,c_current-1)
                    if self.check_boundary(cell):
                        if self.state[cell] > value:
                          value = self.state[cell] 

                    self.state_temp[i,j] = value + 1

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.state[i,j] = self.state_temp[i,j]
        self.fix_value()
    
    def check_boundary(self, cell):
        status = True
        if cell[0] < 0 \
            or cell[0] > self.size[0] -1 \
            or cell[1] < 0  \
            or cell[1] > self.size[1] -1 :
          status = False
        return status
    
    