from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Environment(ABC):
    
    @abstractmethod
    def state_shape(self):
        pass
    
    @abstractmethod
    def action_shape(self):
        pass
    
    @abstractmethod
    def current_state(self):
        pass
    
    @abstractmethod
    def execute(self, action):
        pass
    
    @abstractmethod
    def is_terminal(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass


class QFunction(ABC):
    
    @abstractmethod
    def max(self, state):
        pass
    
    def max_value(self, state):
        (val, _) = self.max(state)
        return val
    
    def max_action(self, state):
        (_, action) = self.max(state)
        return action
    
    @abstractmethod
    def evaluate(self, state, action):
        pass
    
    @abstractmethod
    def update(self, state, action, delta):
        pass


def epsilon_greedy(action_generator, epsilon=0.015):
    
    if isinstance(action_generator, tuple):
        n_actions = np.prod(np.array(action_generator))
        a = np.unravel_index([i for i in range(n_actions)], shape=action_generator)
        a = [(a[j][i] for j in range(len(action_generator))) for i in range(n_actions)]
        action_generator = a
    
    if isinstance(action_generator, (np.ndarray, list)):
        options = action_generator
        def random_choice(state, action):
            action = np.random.choice(options)
            if not isinstance(action, tuple):
                action = (action,)
            return action
        action_generator = random_choice
    
    def behavior(state, action):
        if np.random.rand() < epsilon:
            action = action_generator(state, action)
        return action
    return behavior


class Algo(ABC):
    
    def __init__(self, env:Environment, Q:QFunction):
        self.environment = env
        self.Q = Q

    @abstractmethod
    def run_step(self):
        pass
    
    @abstractmethod
    def run_episode(self, max_steps=None, after_step_func=None):
        pass
    

class QLAlgo(Algo):
    
    def __init__(self, env: Environment, Q: QFunction, learning_rate=0.1, discount_factor=0.95, behavior=None):
        super().__init__(env=env, Q=Q)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.behavior = behavior
    
    def act(self):
        if self.environment.is_terminal():
            self.environment.reset()
        s0 = self.environment.current_state()
        a = self.Q.max_action(s0)
        if self.behavior is not None:
            a = self.behavior(s0, a)
        r = self.environment.execute(a)
        s1 = self.environment.current_state()
        t = self.environment.is_terminal()
        return (s0, a, s1, r, t)
    
    def run_step(self):
        step = self.act()
        self.update(*step[:-1])
        return step
        
    def update(self, s0, a, s1, r):
        q0 = self.Q.evaluate(s0, a)
        q1 = self.Q.max_value(s1)
        delta = self.learning_rate * (r + self.discount_factor * q1 - q0)
        self.Q.update(s0, a, delta)
    
    def run_episode(self, max_steps=None, after_step_func=None):
        steps = []
        self.environment.reset()
        while not self.environment.is_terminal() \
              and (max_steps is None or len(steps) < max_steps):
            steps.append(self.run_step())
            if after_step_func is not None:
                after_step_func()
        return steps


class DynaQ(QLAlgo):
    
    def __init__(self, env: Environment, Q: QFunction, batch_size=16, memory_size=256, learning_rate=0.1, discount_factor=0.95, behavior=None):
        super().__init__(env=env, Q=Q, learning_rate=learning_rate, discount_factor=discount_factor, behavior=behavior)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.forget()
    
    def remember(self, s0, a, s1, r, t):
        self.memory[self.memory_idx] = (s0, a, s1, r, t)
        self.memory_idx = (self.memory_idx + 1) % self.memory_size
        if not self.memory_full and self.memory_idx == 0:
                self.memory_full = True

    def forget(self):
        self.memory = np.full((self.memory_size,), None)
        self.memory_idx = 0
        self.memory_full = False
    
    def run_step(self, batch_size=None):
        step = self.learn_step()
        self.remember(*step)
        self.plan_step(batch_size=batch_size)
        return step
    
    def learn_step(self):
        return super().run_step()

    def plan_step(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        memory = self.memory
        if not self.memory_full:
            if self.memory_idx < batch_size:
                return
            memory = memory[:self.memory_idx]
        episodes = np.random.choice(memory, batch_size)
        for step in episodes:
            self.update(*step[:-1])


class TabularQFunction(QFunction):
    
    def __init__(self, env:Environment, initial_value=0, randomize_equal_actions=True):
        self.state_shape = env.state_shape()
        self.action_shape = env.action_shape()
        self.shape = (np.prod(np.array(self.state_shape)), np.prod(np.array(self.action_shape)))
        self.Q = np.full(self.shape, initial_value, dtype=np.float)
        self.randomize = randomize_equal_actions
    
    def evaluate(self, state, action):
        return self.Q[self._idx(state, action)]
    
    def max_value(self, state):
        return self.Q[self._idx(state)].max()
    
    def max(self, state):
        q = self.Q[self._idx(state)]
        if not self.randomize:
            a = q.argmax()
            v = q[a]
        else:
            v = q.max()
            a = np.transpose(np.array(np.where(q == v)))
            a = a[np.random.randint(0, a.shape[0]),:]
            a = tuple(a)
        a = self._coords(action=a)
        return (v, a)
    
    def update(self, state, action, delta):
        self.Q[self._idx(state, action)] += delta
    
    def _idx(self, state, action=None):
        idx = (np.ravel_multi_index(state, self.state_shape),)
        if action is not None:
            idx = idx + (np.ravel_multi_index(action, self.action_shape),)
        return idx
    
    def _coords(self, state=None, action=None):
        if state is not None:
            idx = state
            shape = self.state_shape
        elif action is not None:
            idx = action
            shape = self.action_shape
        else:
            raise ValueError('One of state or action must be given.')
        if state is not None and action is not None:
            raise ValueError('Only one of state and action must be given.')
        coords = tuple([c[0] for c in np.unravel_index(idx, shape)])
        return coords
