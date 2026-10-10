from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.data = deque(maxlen=capacity)
    def __len__(self):
        return len(self.data)
    def add(self, state, action, reward, next_state, terminated):
        self.data.append((np.asarray(state, dtype=np.float32).copy(), int(action), float(reward), np.asarray(next_state, dtype=np.float32).copy(), bool(terminated)))
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        s, a, r, ns, t = zip(*batch)
        return np.stack(s), np.asarray(a), np.asarray(r, dtype=np.float32), np.stack(ns), np.asarray(t, dtype=np.float32)
