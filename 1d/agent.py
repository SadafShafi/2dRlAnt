from pathlib import Path

import numpy as np


STATE_VALUES = (-1, 1)
ACTION_VALUES = (0, 1)


class QLearningConfig:
    def __init__(
        self,
        learning_rate=0.35,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed


class QLearningAgent:
    def __init__(self, config=None, q_table=None):
        self.config = config or QLearningConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.q_table = q_table if q_table is not None else np.zeros((2, 2, 2), dtype=np.float32)

    @staticmethod
    def encode_state(state):
        food_dir, poison_dir = state
        food_idx = 1 if food_dir > 0 else 0
        poison_idx = 1 if poison_dir > 0 else 0
        return food_idx, poison_idx

    @staticmethod
    def state_labels():
        return ["(-1, -1)", "(-1, +1)", "(+1, -1)", "(+1, +1)"]

    def q_values(self, state):
        return self.q_table[self.encode_state(state)]

    def select_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, 2))

        values = self.q_values(state)
        best_actions = np.flatnonzero(values == np.max(values))
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        state_idx = self.encode_state(state)
        next_state_idx = self.encode_state(next_state)

        current = self.q_table[state_idx][action]
        target = reward
        if not done:
            target += self.config.discount_factor * float(np.max(self.q_table[next_state_idx]))

        self.q_table[state_idx][action] = current + self.config.learning_rate * (target - current)

    def decay_epsilon(self, epsilon):
        return max(self.config.epsilon_min, epsilon * self.config.epsilon_decay)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    @classmethod
    def load(cls, path, config=None):
        q_table = np.load(path)
        return cls(config=config, q_table=q_table)
