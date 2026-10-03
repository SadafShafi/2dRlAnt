from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


ACTION_LEFT = 0
ACTION_RIGHT = 1


class LineWorldConfig:
    def __init__(
        self,
        length=11,
        max_steps=20,
        food_reward=1.0,
        poison_reward=-1.0,
        step_penalty=-0.02,
        smell_scale=1.0,
        seed=None,
    ):
        self.length = length
        self.max_steps = max_steps
        self.food_reward = food_reward
        self.poison_reward = poison_reward
        self.step_penalty = step_penalty
        self.smell_scale = smell_scale
        self.seed = seed


class LineWorld:
    """A tiny 1D world with smell-based perception.

    Food and poison are always placed on opposite ends. The agent starts in a
    random interior cell, so it must rely on the local smell gradient rather
    than memorizing a fixed starting position.
    """

    def __init__(self, config=None):
        self.config = config or LineWorldConfig()
        if self.config.length < 3:
            raise ValueError("LineWorld length must be at least 3.")
        self.rng = np.random.default_rng(self.config.seed)
        self.food_pos = 0
        self.poison_pos = self.config.length - 1
        self.agent_pos = 1
        self.steps = 0
        self.done = False

    @property
    def length(self) -> int:
        return self.config.length

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.rng.random() < 0.5:
            self.food_pos = 0
            self.poison_pos = self.length - 1
        else:
            self.food_pos = self.length - 1
            self.poison_pos = 0

        self.agent_pos = int(self.rng.integers(1, self.length - 1))
        self.steps = 0
        self.done = False
        return self.observe()

    def _smell_field(self, target_pos: int) -> np.ndarray:
        positions = np.arange(self.length, dtype=np.float32)
        distances = np.abs(positions - float(target_pos))
        # Monotonic and smooth enough to make the gradient obvious in renderings.
        return self.config.smell_scale / (1.0 + distances)

    def smell_fields(self):
        return self._smell_field(self.food_pos), self._smell_field(self.poison_pos)

    @staticmethod
    def _direction_from_field(field: np.ndarray, agent_pos: int) -> int:
        left_idx = max(0, agent_pos - 1)
        right_idx = min(len(field) - 1, agent_pos + 1)
        delta = float(field[right_idx] - field[left_idx])
        # Positive means the smell is stronger to the right.
        return 1 if delta >= 0.0 else -1

    def observe(self):
        food_field, poison_field = self.smell_fields()
        food_dir = self._direction_from_field(food_field, self.agent_pos)
        poison_dir = self._direction_from_field(poison_field, self.agent_pos)
        return food_dir, poison_dir

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called after the episode finished; call reset().")

        if action not in (ACTION_LEFT, ACTION_RIGHT):
            raise ValueError(f"Invalid action {action}; expected 0 (left) or 1 (right).")

        if action == ACTION_LEFT:
            self.agent_pos = max(0, self.agent_pos - 1)
        else:
            self.agent_pos = min(self.length - 1, self.agent_pos + 1)

        self.steps += 1

        reward = float(self.config.step_penalty)
        terminal = "timeout"
        if self.agent_pos == self.food_pos:
            reward += float(self.config.food_reward)
            self.done = True
            terminal = "food"
        elif self.agent_pos == self.poison_pos:
            reward += float(self.config.poison_reward)
            self.done = True
            terminal = "poison"
        elif self.steps >= self.config.max_steps:
            self.done = True

        observation = self.observe()
        info = {
            "terminal": terminal,
            "agent_pos": self.agent_pos,
            "food_pos": self.food_pos,
            "poison_pos": self.poison_pos,
            "steps": self.steps,
        }
        return observation, reward, self.done, info
