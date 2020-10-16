import numpy as np


class ActionValueFunc:
    
    def __init__(self, state_shape, action_shape, initial_value = 0, learning_rate = 0.2):
        self.learning_rate = learning_rate
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._func_shape = self._state_shape + self._action_shape
        self._Q = np.full((np.prod(self._state_shape), np.prod(self._action_shape)), initial_value)
    
    def max_value(self, state=None):
        if state is None:
            return np.reshape(self._Q.max(axis=1), self._state_shape)
        else:
            return np.max(self._Q[np.ravel_multi_index(state, self._state_shape),:])
    
    def max_action(self, state=None):
        if state is None:
            q_max = self._Q.max(axis=1)
            print('q_max[%s]: %s' %(q_max.shape, q_max))
            q_where = self._Q == q_max[:,np.newaxis]
            print(q_where)
            a_max = np.zeros((self._func_shape,))
            for s in range(len(a_max)):
                print(q_where[s,:].shape)
                a = np.argwhere(q_where[s,:])[:,0]
                if len(a) > 1: a = np.random.choice(a, 1)[0]
                else: a = a.flat[0]
                a_max[s] = a
            a_max = np.unravel_index(a_max, self._action_shape)
        else:
            q = self._Q[np.ravel_multi_index(state, self._state_shape),:]
            q_max = np.max(q)
            a_max = np.argwhere(q == q_max)[:,0]
            if len(a_max) == 1: a_max = a_max[0]
            else: a_max = np.random.choice(a_max, 1)[0]
            a_max = np.unravel_index(a_max, self._action_shape)
        return a_max
    
    def evaluate(self, state, action):
        return self._Q.flat[np.ravel_multi_index(state + action, self._func_shape)]
    
    def update(self, state, action, value):
        idx = np.ravel_multi_index(state + action, self._func_shape)
        self._Q.flat[idx] = (1-self.learning_rate) * self._Q.flat[idx] + self.learning_rate * value


class EpsilonGreedyBehavior:
    
    def __init__(self, action_space, epsilon=0.015):
        self.epsilon = epsilon
        self._shape = action_space
    
    def __call__(self, state, action):
        if np.random.random_sample() < self.epsilon: action = tuple(np.random.randint(0, self._shape[d]) for d in range(len(self._shape)))
        return action


class QLAlgo:
    
    def __init__(self, Q: ActionValueFunc, start_func, terminal_func, transition_func, reward_func,
                 behavior=None,
                 discount_rate=0.9):
        self.Q = Q
        self._discount = discount_rate
        self._start = start_func
        self._terminal = terminal_func
        self._transition = transition_func
        self._reward = reward_func
        self._state = self._start()
        if behavior is None: behavior = EpsilonGreedyBehavior(self.Q._action_shape)
        self._behavior = behavior
    
    def run_step(self):
        self._init_state()
        s0 = self._state
        a = self._behavior(s0, self.Q.max_action(s0))
        s1 = self._transition(s0, a)
        r = self._reward(s0, a, s1)
        self.Q.update(s0, a, r + self._discount * self.Q.max_value(s1))
        t = self._terminal(s1)
        self._state = None if t else s1
        return (s0, a, s1, r, t)
    
    def run_episode(self, max_steps=None):
        self._init_state()
        t = self._terminal(self._state)
        history = []
        while not t and (max_steps is None or len(history) < max_steps):
            step = self.run_step()
            history.append(step)
            (_,_,_,_,t) = step
        self._state = None
        return history
    
    def _init_state(self):
        if self._state is None: self._state = self._start()
        
