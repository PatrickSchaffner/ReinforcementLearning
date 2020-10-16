import numpy as np


class TransitionModel:
    
    def __init__(self, state_shape, action_shape):
        self._counts = np.zeros(state_shape + action_shape + state_shape, dtype=np.uint)
    
    def update(self, start_state, action, end_state):
        self._counts[start_state + action + end_state] += 1
    
    def predict(self, start_state, action):
        p = self._counts[start_state + action]
        return p.astype(np.float) / np.sum(p)


class RewardModel:
    
    def __init__(self, state_shape, action_shape, learning_rate = 0.2, initial_value = 0):
        self.learning_rate = learning_rate
        self._r = np.full(state_shape + action_shape, initial_value, dtype=np.uint)
    
    def update(self, state, action, reward):
        idx = state + action
        self._r[idx] = (1-self.learning_rate) * self._r[idx] + self.learning_rate * reward
    
    def predict(self, state, action):
        return self._r[state + action]

