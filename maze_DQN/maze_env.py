import numpy as np
import time
import tkinter as tk

UNIT = 40
MAZE_H = 5
MAZE_W = 5
radius = UNIT / 2 - 5


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()  # 用于继承tk类方法
        self.action_type = ['u', 'd', 'l', 'r']  # 四个方向的移动
        self.n_action = len(self.action_type)  # 神经网络输入端参数数量
        self.n_features = 2  # 神经网络输入端参数数量，为探险者当前位置x,y坐标
        self.title('Maze')  # 画布名称
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._build_maze()

        # 初始化画布

    def _build_maze(self):
        # 基本背景
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT
                                )

        # 画线
        for i in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = i, 0, i, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for i in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, i, MAZE_W * UNIT, i
            self.canvas.create_line(x0, y0, x1, y1)

        # 创建基本元素
        origin = np.array([20, 20])  # 画布的基点

        # 创建两个hell
        hell_1_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.hell_1 = self.canvas.create_rectangle(
            hell_1_center[0] - radius, hell_1_center[1] - radius,
            hell_1_center[0] + radius, hell_1_center[1] + radius,
            fill='black'
        )
        hell_2_center = origin + np.array([UNIT, UNIT * 3])
        self.hell_2 = self.canvas.create_rectangle(
            hell_2_center[0] - radius, hell_2_center[1] - radius,
            hell_2_center[0] + radius, hell_2_center[1] + radius,
            fill='black'
        )

        # 创建目标
        oval_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.oval = self.canvas.create_oval(
            oval_center[0] - radius, oval_center[1] - radius,
            oval_center[0] + radius, oval_center[1] + radius,
            fill='yellow'
        )

        # 创建冒险家
        self.adventurer = self.canvas.create_rectangle(
            origin[0] - radius, origin[1] - radius,
            origin[0] + radius, origin[1] - radius,
            fill='red'
        )

        self.canvas.pack()

    # 初始化环境,返回初始状态
    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.adventurer)
        origin = np.array([20, 20])
        self.adventurer = self.canvas.create_rectangle(
            origin[0] - radius, origin[1] - radius,
            origin[0] + radius, origin[1] + radius,
            fill='red'
        )
        return (np.array(self.canvas.coords(self.adventurer)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(UNIT*MAZE_H)

    # 返回下一步的state, reward, done(是否终止)
    def next_state(self, action):
        s = self.canvas.coords(self.adventurer)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # move
        self.canvas.move(self.adventurer, base_action[0], base_action[1])
        next_adventurer = self.canvas.coords(self.adventurer)

        # reward function
        if next_adventurer == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_adventurer in [self.canvas.coords(self.hell_1), self.canvas.coords(self.hell_2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        next_state = (np.array(next_adventurer[0:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return next_state, reward, done
