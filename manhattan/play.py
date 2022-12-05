from grid_world import GridWorld
from graphic_display import GraphicDisplay

if __name__ == "__main__":
    size = (10, 10)
    start_cell = (9, 0)
    obstacles = [(3, 2), (4,2), (5,2), (6,2), (8,3), (9,3), (4,5), (5,5), (6,5), (7,5), (8,5), (9,5), (4,6), (4,7)]
    goal_state = (6, 6)

    gw = GridWorld(size, start_cell, obstacles, goal_state)
    for i in range(15):
      gw.update_value()
    
    gw.final_value()

    grid_world = GraphicDisplay(gw.state)
    grid_world.mainloop()