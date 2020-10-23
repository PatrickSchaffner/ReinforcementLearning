from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Space(object):
    
    def __init__(self, shape):
        self._shape = shape
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def ndims(self):
        return len(self.shape)
    
    @property
    def nelems(self):
        return np.prod(self.shape)
            

class StateSpace(Space):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ActionSpace(Space):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def random(self):
        raise NotImplementedError


class DiscreteSpace(StateSpace, ActionSpace):
    
    def __init__(self, *args, **kwargs):
        super(ActionSpace, self).__init__(*args, **kwargs)
        #super(StateSpace, self).__init__(*args, **kwargs)
    
    def random(self):
        return np.random.randint(self.shape)
    
    def ravel(self, multi_idx):
        if isinstance(multi_idx, list):
            multi_idx = tuple(np.transpose(np.array(multi_idx)))
        return np.ravel_multi_index(multi_idx, self.shape)

    def unravel(self, idx):
        multi_idx = np.unravel_index(idx, self.shape)
        if isinstance(idx, (list, np.ndarray)):
            multi_idx = np.array(multi_idx)
            multi_idx = [tuple(multi_idx[:,i]) for i in range(len(idx))]
        return multi_idx


class Environment(ABC):
    
    def __init__(self, state_space:StateSpace, action_space:ActionSpace):
        self._states = state_space
        self._actions = action_space
    
    @property
    def state_space(self) -> StateSpace:
        return self._states
    
    @property
    def action_space(self) -> ActionSpace:
        return self._actions
    
    @property
    @abstractmethod
    def current_state(self):
        raise NotImplementedError
    
    @abstractmethod
    def execute(self, action) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def is_terminal(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError


def epsilon_greedy(action_generator, epsilon=0.015):
    
    if isinstance(action_generator, Environment):
        action_generator = action_generator.action_space
    
    if isinstance(action_generator, ActionSpace):
        action_generator = lambda s,a: action_generator.random()
    
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
