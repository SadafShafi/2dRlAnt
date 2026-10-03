from __future__ import division

import numpy as np


ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


class GridWorld2DConfig:
    def __init__(
        self,
        width=7,
        height=7,
        max_steps=45,
        food_reward=1.0,
        poison_reward=-1.0,
        step_penalty=-0.02,
        wall_penalty=-0.05,
        smell_lambda=2.2,
        seed=None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.food_reward = float(food_reward)
        self.poison_reward = float(poison_reward)
        self.step_penalty = float(step_penalty)
        self.wall_penalty = float(wall_penalty)
        self.smell_lambda = float(smell_lambda)
        self.seed = seed


class GridWorld2D:
    """2D grid world with one food and one poison at random positions each episode."""

    def __init__(self, config=None):
        self.config = config if config is not None else GridWorld2DConfig()
        if self.config.width < 3 or self.config.height < 3:
            raise ValueError("Grid dimensions must be at least 3x3.")
        self.rng = np.random.RandomState(self.config.seed)
        self.food_pos = (0, 0)
        self.poison_pos = (self.config.width - 1, self.config.height - 1)
        self.agent_pos = (self.config.width // 2, self.config.height // 2)
        self.steps = 0
        self.done = False

    @property
    def width(self):
        return self.config.width

    @property
    def height(self):
        return self.config.height

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        all_positions = [(x, y) for y in range(self.height) for x in range(self.width)]
        idx = self.rng.choice(len(all_positions), size=3, replace=False)
        self.food_pos = all_positions[int(idx[0])]
        self.poison_pos = all_positions[int(idx[1])]
        self.agent_pos = all_positions[int(idx[2])]

        self.steps = 0
        self.done = False
        return self.observe()

    def _single_smell_field(self, target_pos):
        tx, ty = target_pos
        xs = np.arange(self.width, dtype=np.float32)
        ys = np.arange(self.height, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        dist = np.sqrt((xx - float(tx)) ** 2 + (yy - float(ty)) ** 2)
        return np.exp(-dist / self.config.smell_lambda)

    def smell_fields(self):
        return self._single_smell_field(self.food_pos), self._single_smell_field(self.poison_pos)

    @staticmethod
    def _gradient_dirs(field, agent_pos):
        x, y = agent_pos
        h, w = field.shape

        left_x = max(0, x - 1)
        right_x = min(w - 1, x + 1)
        up_y = max(0, y - 1)
        down_y = min(h - 1, y + 1)

        dx = float(field[y, right_x] - field[y, left_x])
        dy = float(field[down_y, x] - field[up_y, x])

        x_dir = 1 if dx > 1e-9 else (-1 if dx < -1e-9 else 0)
        y_dir = 1 if dy > 1e-9 else (-1 if dy < -1e-9 else 0)
        return x_dir, y_dir

    def observe(self):
        food_field, poison_field = self.smell_fields()
        food_x_dir, food_y_dir = self._gradient_dirs(food_field, self.agent_pos)
        poison_x_dir, poison_y_dir = self._gradient_dirs(poison_field, self.agent_pos)
        return food_x_dir, food_y_dir, poison_x_dir, poison_y_dir


    def step(self, action):
        if self.done:
            raise RuntimeError("step() called after done=True. Call reset().")

        if action not in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            raise ValueError("Invalid action: {0}".format(action))

        x, y = self.agent_pos
        nx, ny = x, y
        if action == ACTION_UP:
            ny = y - 1
        elif action == ACTION_DOWN:
            ny = y + 1
        elif action == ACTION_LEFT:
            nx = x - 1
        elif action == ACTION_RIGHT:
            nx = x + 1

        wall_bump = not (0 <= nx < self.width and 0 <= ny < self.height)
        if wall_bump:
            nx, ny = x, y  # stay in place on a wall bump

        self.agent_pos = (nx, ny)
        self.steps += 1

        reward = self.config.step_penalty
        terminal = "timeout"
        if wall_bump:
            reward += self.config.wall_penalty

        if self.agent_pos == self.food_pos:
            reward += self.config.food_reward
            self.done = True
            terminal = "food"
        elif self.agent_pos == self.poison_pos:
            reward += self.config.poison_reward
            self.done = True
            terminal = "poison"
        elif self.steps >= self.config.max_steps:
            self.done = True

        obs = self.observe()
        info = {
            "terminal": terminal,
            "agent_pos": self.agent_pos,
            "food_pos": self.food_pos,
            "poison_pos": self.poison_pos,
            "steps": self.steps,
            "wall_bump": wall_bump,
            "action_requested": action,
            "action_executed": action,
        }
        return obs, float(reward), self.done, info
    # def step(self, action):
    #     if self.done:
    #         raise RuntimeError("step() called after done=True. Call reset().")

    #     if action not in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
    #         raise ValueError("Invalid action: {0}".format(action))

    #     requested_action = action
    #     x, y = self.agent_pos
    #     old_x, old_y = x, y
    #     wall_bump = False
    #     if action == ACTION_UP:
    #         if y == 0:
    #             wall_bump = True
    #             action = ACTION_RIGHT if x < self.width - 1 else ACTION_LEFT
    #         else:
    #             y -= 1
    #     elif action == ACTION_DOWN:
    #         if y == self.height - 1:
    #             wall_bump = True
    #             action = ACTION_RIGHT if x < self.width - 1 else ACTION_LEFT
    #         else:
    #             y += 1
    #     elif action == ACTION_LEFT:
    #         if x == 0:
    #             wall_bump = True
    #             action = ACTION_UP if y > 0 else ACTION_DOWN
    #         else:
    #             x -= 1
    #     elif action == ACTION_RIGHT:
    #         if x == self.width - 1:
    #             wall_bump = True
    #             action = ACTION_UP if y > 0 else ACTION_DOWN
    #         else:
    #             x += 1

    #     if action == ACTION_UP:
    #         y = max(0, y - 1)
    #     elif action == ACTION_DOWN:
    #         y = min(self.height - 1, y + 1)
    #     elif action == ACTION_LEFT:
    #         x = max(0, x - 1)
    #     elif action == ACTION_RIGHT:
    #         x = min(self.width - 1, x + 1)

    #     self.agent_pos = (x, y)
    #     self.steps += 1

    #     reward = self.config.step_penalty
    #     terminal = "timeout"
    #     if wall_bump:
    #         reward += self.config.wall_penalty

    #     if self.agent_pos == self.food_pos:
    #         reward += self.config.food_reward
    #         self.done = True
    #         terminal = "food"
    #     elif self.agent_pos == self.poison_pos:
    #         reward += self.config.poison_reward
    #         self.done = True
    #         terminal = "poison"
    #     elif self.steps >= self.config.max_steps:
    #         self.done = True

    #     obs = self.observe()
    #     info = {
    #         "terminal": terminal,
    #         "agent_pos": self.agent_pos,
    #         "food_pos": self.food_pos,
    #         "poison_pos": self.poison_pos,
    #         "steps": self.steps,
    #         "wall_bump": wall_bump,
    #         "action_requested": requested_action,
    #         "action_executed": action,
    #     }
    #     return obs, float(reward), self.done, info
