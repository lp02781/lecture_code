import tkinter as tk
import numpy as np
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100  # 픽셀 수
HEIGHT = 5  # 그리드월드 세로
WIDTH = 5  # 그리드월드 가로
POSSIBLE_ACTIONS = [0, 1, 2, 3]

class Env:
    def __init__(self):
        self.width = WIDTH  # Width of Grid World
        self.height = HEIGHT  # Height of GridWorld
        self.reward = [[0] * WIDTH for _ in range(HEIGHT)]
        self.possible_actions = POSSIBLE_ACTIONS
        self.all_state = []

        for x in range(WIDTH):
            for y in range(HEIGHT):
                state = [x, y]
                self.all_state.append(state)

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_index):
        action = ACTIONS[action_index]
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1
        if state[0] > WIDTH - 1 else state[0])
        state[1] = (0 if state[1] < 0 else HEIGHT - 1
        if state[1] > HEIGHT - 1 else state[1])
        return state

    def get_transition_prob(self, state, action):
        return self.transition_probability

    def get_all_states(self):
        return self.all_state

class GraphicDisplay(tk.Tk):
    def __init__(self, q_table, number):
        super(GraphicDisplay, self).__init__()
        self.title('grid world')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT + 50))
        self.texts = []
        self.arrows = []
        self.env = Env()
        self.number=number
        self.q_table = q_table
        self.iteration_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        (self.up, self.down, self.left,
         self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.text_reward(4, 4, "R : 1.0")
        self.print_box_number()
        self.print_optimal_policy()

    def print_optimal_policy(self):
        for i in self.arrows:
            self.canvas.delete(i)
        for state in self.env.get_all_states():
            i = state[0]
            j = state[1]
            action = np.argmax(self.q_table[:,i,j])
            self.draw_from_values(state, action)

    def draw_from_values(self, state, action):
        i = state[0]
        j = state[1]
        self.draw_one_arrow(i, j, action)

    def draw_one_arrow(self, col, row, action):
        if col == 4 and row == 4:
            return
        if action == 1:  # up
            origin_x, origin_y = 50 + (UNIT * row), 10 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.up))
        elif action == 3:  # down
            origin_x, origin_y = 50 + (UNIT * row), 90 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.down))
        elif action == 2:  # right
            origin_x, origin_y = 90 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.right))
        elif action == 0:  # left
            origin_x, origin_y = 10 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.left))

    def print_box_number(self):
        haha = 1
        for i in range(WIDTH):
            for j in range(HEIGHT):
                self.text_box(i, j, haha)                                                  
                haha += 1    

    def text_box(self, row, col, contents, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
        origin_x, origin_y = 50, 40
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)

        # 그리드 생성
        for col in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, HEIGHT * UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        if self.number == 1:
            canvas.create_image(150, 250, image=self.shapes[1])
            canvas.create_image(350, 150, image=self.shapes[1])
            canvas.create_image(450, 450, image=self.shapes[2])
        elif self.number == 2:
            canvas.create_image(150, 250, image=self.shapes[1])
            canvas.create_image(450, 450, image=self.shapes[2])

        canvas.pack()

        return canvas
    
    def text_reward(self, row, col, contents, font='Helvetica', size=10,
                    style='normal', anchor="nw"):
        origin_x, origin_y = 30, 25
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def load_images(self):
        PhotoImage = ImageTk.PhotoImage
        up = PhotoImage(Image.open("img/up.png").resize((13, 13)))
        right = PhotoImage(Image.open("img/right.png").resize((13, 13)))
        left = PhotoImage(Image.open("img/left.png").resize((13, 13)))
        down = PhotoImage(Image.open("img/down.png").resize((13, 13)))
        rectangle = PhotoImage(
            Image.open("img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("img/circle.png").resize((65, 65)))
        return (up, down, left, right), (rectangle, triangle, circle)