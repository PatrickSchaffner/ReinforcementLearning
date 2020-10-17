from abc import ABC, abstractmethod

import numpy as np


class Environment(ABC):
    
    @abstractmethod
    def current_state(self):
        pass
    
    @abstractmethod
    def execute(self, action)
        pass


class QFunction(ABC):
    
    @abstractmethod
    def max_value(self, state):
        pass
    
    @abstractmethod
    def max_action(self, state):
        pass
    
    @abstractmethod
    def evaluate(self, state, action):
        pass
    
    @abstractmethod
    def update(self, state, action, delta):
        pass


class QLAlgo():
    
    def __init__(self, env: Environment, Q: QFunction, learning_rate=0.1, discount_factor=0.95, eps_greedy=0.03):
        self.environment = env
        self.Q = Q
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    def runStep(self):
        
        s0 = self.environment.current_state()
        a = self.Q.max_action(s0)  # TODO epsilon greedy
        (s0, a, s1, r, t) = self.environment.execute(a)
        
        q0 = self.Q.evaluate(s0, a)
        q1 = self.Q.max_value(s1)
        delta = self.learning_rate * (r + self.discount_factor * q1 - q0)
        self.Q.update(s0, a, delta)
        
        # reset if t
        
        return (s0, a, s1, r, t)
    
    def runEpisode(self, max_steps=None):
        t = False
        steps = []
        while not t and (max_steps is None or len(steps) < max_steps):
            step = self.runStep()
            (_,_,_,_,t) = step
            steps.append(step)
        return steps


class TabularQFunction(QFunction):
    
    def __init__(self, state_shape, action_shape, initial_value=0, randomize_equal_actions=True):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.shape = (np.prod(np.array(self.state_shape)), np.prod(np.array(self.action_shape)))
        self.Q = np.full(self.shape, initial_value, dtype=np.float)
        self.randomize = randomize_equal_actions
    
    def evaluate(self, state, action):
        return self.Q[self._idx(state, action)]
    
    def max_value(self, state):
        return self.Q[self._idx(state)].max()
    
    def max_action(self, state):
        q = self.Q[self._idx(state)]
        if not self.randomize:
            return q.argmax()
        else:
            a = np.where(q == q.max())
            a = a if len(a) < 2 else np.random.choice(a, 1)[0]
            return a
    
    def update(self, state, action, delta):
        self.Q[self._idx(state, action)] += delta
    
    def _idx(self, state, action=None):
        idx = (np.ravel_multi_index(state, self.state_shape),)
        if action is not None:
            idx = idx + (np.ravel_multi_index(action, self.action_shape),)
        return idx
