class Env:
    def __init__(self):

        self.width = 5
        self.height = 5
        self.reward = [[0] * self.width for _ in range(self.width)]
        self.reward[4][4] = 10.0 
        self.reward[2][1] = -3.0  
        self.reward[1][3] = -3.0  

    def step(self, state, action):
        next_state = self.get_state(state, action)
        return next_state,self.reward[next_state[0]][next_state[1]]

    def set_reward(self,obstacle,goal):
        self.reward[2][1]=obstacle
        self.reward[1][3]=obstacle
        self.reward[4][4]=goal
    def get_state(self,state, action):
        action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        state[0] += action_grid[action][0]
        state[1] += action_grid[action][1]

        if state[0] < 0:
            state[0] = 0
        elif state[0] > 4:
            state[0] = 4

        if state[1] < 0:
            state[1] = 0
        elif state[1] > 4:
            state[1] = 4

        return [state[0], state[1]]
