
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class BitSwitchEnv:
    def __init__(self, size: int) -> None:
        self.size = size
        self.reached = False
        self.choices = np.array([0.0, 1.0], dtype=np.float32)

        self.reset()

    def reset(self) -> NDArray:
        self.state = np.random.choice(self.choices, size=self.size)
        self.goal = self.generate_goal()
        self.reached = False

        return self.goal

    def _flip(self, n: int) -> None:
        if self.state[n] == 0.0:
            self.state[n] = 1.0
        else:
            self.state[n] = 0.0

    def generate_goal(self) -> NDArray:
        return np.random.choice(self.choices, size=self.size)

    def calculate_reward(self) -> int:
        return -((self.state - self.goal)**2).mean() if not self.is_done() else 10
        # return 15 if self.is_done() else -1

    def is_done(self) -> int:
        return np.array_equal(self.state, self.goal)

    def step(self, action: int) -> Tuple[NDArray, int, int]:
        if action >= self.size:
            raise Exception("Action is larger than action space")

        self._flip(action)

        reached = 1 if self.reached else 0

        if not self.reached and self.is_done():
            self.reached = True

        return (
            np.copy(self.state),
            self.calculate_reward(),
            reached
        )
