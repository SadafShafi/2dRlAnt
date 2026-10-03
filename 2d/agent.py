from __future__ import division

from pathlib import Path

import numpy as np


ACTION_VALUES = (0, 1, 2, 3)
DIR_VALUES = (-1, 0, 1)


class QLearning2DConfig:
    def __init__(
        self,
        learning_rate=0.35,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.seed = seed


class QLearning2DAgent:
    def __init__(self, config=None, q_table=None):
        self.config = config if config is not None else QLearning2DConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.q_table = q_table if q_table is not None else np.zeros((3, 3, 3, 3, 4), dtype=np.float32)

    @staticmethod
    def encode_state(state):
        # Map {-1,0,1} -> {0,1,2}
        return tuple(int(v + 1) for v in state)

    def q_values(self, state):
        return self.q_table[self.encode_state(state)]

    def select_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return int(self.rng.randint(0, 4))
        values = self.q_values(state)
        best = np.flatnonzero(values == np.max(values))
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, done):
        s = self.encode_state(state)
        ns = self.encode_state(next_state)
        current = self.q_table[s][action]
        target = float(reward)
        if not done:
            target += self.config.discount_factor * float(np.max(self.q_table[ns]))
        self.q_table[s][action] = current + self.config.learning_rate * (target - current)

    def decay_epsilon(self, epsilon):
        return max(self.config.epsilon_min, epsilon * self.config.epsilon_decay)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.q_table)

    @classmethod
    def load(cls, path, config=None):
        table = np.load(str(path))
        return cls(config=config, q_table=table)
