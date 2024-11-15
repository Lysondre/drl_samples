
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ExperienceBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque([], maxlen=capacity)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def __len__(self):
        return len(self.buffer)
