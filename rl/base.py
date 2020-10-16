from abc import ABC


class Simulator(ABC):
    
    @abstractmethod
    def getState(self):
        pass
    
    @abstractmethod
    def reset(self, state):
        pass
    
    @abstractmethod
    def update(self, action):
        pass
    
    def __call__(self, action):
        return self.update(action)


class ActionValueFunction(ABC):
    
    @abstractmethod
    def getValue(self, state, action):
        pass
    
    @abstractmethod
    def setValue(self, state, action, value):
        pass
    
    @abstractmethod
    def maxValue(self, state):
        pass
    
    @abstractmethod
    def optAction(self, state):
        pass
    
    @abstractmethod
    def meanVal(self):
        pass
    
    def __getitem__(self, key):
        return self.getValue(*key)
    
    def __setitem__(self, key, val):
        return self.setValue(*key, val)


class Policy(ABC):
    
    @abstractmethod
    def computeAction(self, avf: ActionValueFunction, state):
        pass
    
    def __call__(self, avf: ActionValueFunction, state):
        return self.computeAction(avf, state)


class Reward(ABC):
    
    @abstractmethod
    def computeReward(self, previous_state, action, next_state, terminal):
        pass
    
    def __call__(self, s0, a, s1, t):
        return self.computeReward(s0, a, s1, t)


class UpdateRule(ABC):
    
    @abstractmethod
    def update(self, afv: ActionValueFunction, pol: Policy, previuos_state, action, next_state, reward):
        pass
    
    def __call__(self, afv, pol, s0, a, s1, r):
        return self.update(afv, pol, s0, a, s1, r)


class TerminationRule(ABC):

    @abstractmethod
    def is_terminal(self, state):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    def __call__(self, s):
        return self.is_terminal(s)


class Algorithm:
    
    def __init__(self):
        self._avf = None
        self._sim = None
        self._pol = None
        self._behav = None
        self._rew = None
        self._upd = None
        self._ter = None
        self._init_state = None
        self._init = False
    
    def set(self,
            avf: ActionValueFunction = None,
            simulator: Simulator = None,
            policy: Policy = None,
            behavior: Policy = None,
            reward: Reward = None,
            update: UpdateRule = None,
            termination: TerminationRule = None,
            init_state = None):
        if avf is not None: self._avf = avf
        if simulator is not None: self._sim = simulator
        if policy is not None: self._pol = policy
        if behavior is not None: self._behav = behavior
        if reward is not None: self._rew = reward
        if update is not None: self._upd = update
        if termination is not None: self._ter = termination
        if init_state is not None: self._init_state = init_state
    
    def getValue(self):
        return self._avf
    
    def runSteps(self, max_steps=1):
        self._initialize()
        history = []
        terminal = False
        i = 0
        while not terminal and ( i < max_steps or max_steps is None ):
            i += 1
            step, terminal = self._computeStep()
            history.append(step)
        if terminal: self._resetEpisode()
        return history, terminal
    
    def runEpisodes(self, n_episodes = 1, n_collect = 0, max_steps = None):
        if n_collect > n_episodes: n_collect = n_episodes
        n_collect = n_episodes - n_collect
        episodes = []
        steps = 0
        for i in range(n_episodes):
            mx = None if max_steps is None else max_steps - steps
            history = self.runSteps(max_steps = mx)
            if i >= n_collect: episodes.append(history)
            steps += len(history)
            if steps >= max_steps: break
        return episodes
    
    def _initialize(self):
        if not self._init:
            self._resetEpisode()
            self._init = True
    
    def _resetEpisode(self):
        self._ter.reset()
        self._sim.reset(self._init_state if not callable(self._init_state) else self._init_state())
    
    def _computeStep(self):
        s0 = self._sim.getState()
        a = self._behav(self._avf, self._state)
        s1 = self._sim(a)
        t = self._ter(s1)
        r = self._rew(s0, a, s1, t)
        self._avf = self._upd(self._avf, self._pol, s0, a, s1, r)
        return (s0, a, s1, r), t


        
