from numpy.random import Generator, default_rng

from . import Policy


class Greedy(Policy):
    
    def computeAction(self, avf, state):
        return avf.optAction(state)


class EpsilonGreedy(Policy):
    
    def __init__(self, policy, epsilon, action_generator, rng = None):
        self._policy = policy
        self._eps = epsilon
        self._gen = action_generator
        if rng is int: rng = Generator(rng)
        if rng is None: rng = default_rng()
        self._rng = rng
    
    def computeAction(self, avf, state):
        if self._rng.random() < self._epsilon:
            return self._gen()
        else:
            return self._policy(state)
