import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from rl.gridworld import Gridworld, plot_gridworld
from rl.ql import DynaQ, QLAlgo, TabularQFunction, epsilon_greedy


dim = (20, 20)
start = (0, 0)
target = (18, 17)

blocked = np.full(dim, False)
blocked[0:5,[2,16]] = True
blocked[7,4:18] = True
blocked[4:15,10] = True
blocked[12:20,14] = True

epsilon = 0.1

def params(algoType):
    p = {}
    if issubclass(algoType, QLAlgo):
        p = {**p,
             'learning_rate': 0.3,
             'discount_factor': 0.975}
    if issubclass(algoType, DynaQ):
        p = {**p,
             'batch_size': 8,
             'memory_size': 2048}
    return p

algos = [(QLAlgo, {'learning_rate': 0.05}),
         QLAlgo,
         (QLAlgo, {'learning_rate': 0.85}),
         (QLAlgo, {'discount_factor': 0.65}),
         DynaQ,
         (DynaQ, {'learning_rate': 0.05, 'batch_size': 32})]


class AlgoInstance():
    
    def __init__(self, grid, Q, algo):
        
        self.grid = grid
        self.Q = Q
        self.algo = algo
        self.episodes = []
        self.rewards = []
        
        self.name = type(self.algo).__name__
        if isinstance(self.algo, QLAlgo):
            self.name += ' [lr=%.2f' % (self.algo.learning_rate)
            if isinstance(self.algo, DynaQ):
                self.name += ', bs=%d' % (self.algo.batch_size)
        self.name += ']'
    
    def run_episode(self):
        history = self.algo.run_episode()
        self.episodes.append(history)
        r = np.sum([h[3] for h in history])
        self.rewards.append(r)
    
    def plot(self):
        plot_gridworld(self.grid,
                       Q=self.Q,
                       episodes=self.episodes,
                       colorbar=False)
        plt.title(self.name)
        self.episodes = []


def create_instance(algoType):
    
    if isinstance(algoType, list):
        return [create_instance(a) for a in algoType]
    if isinstance(algoType, tuple):
        p = algoType[1]
        algoType = algoType[0]
    else:
        p = {}
        
    grid = Gridworld(dim, start, target, blocked=blocked)
    Q = TabularQFunction(grid, initial_value=0)
    behavior = epsilon_greedy(grid.actions, epsilon=epsilon)
    algo = algoType(env=grid, Q=Q, **{**params(algoType), **p})
    return AlgoInstance(grid, Q, algo)

algos = create_instance(algos)


#matplotlib.use('Qt4agg')
plt.figure()
plt.ion()

def plot_algos():
    
    plt.clf()
    
    N = len(algos)
    P = N + 1
    if P<4:
        dim = (1, P)
    else:
        d = np.sqrt(np.float(P))
        cols = np.int(np.ceil(d))
        rows = np.int(np.floor(d))
        if cols * rows < P: rows = cols
        dim = (rows, cols)
    
    for i in range(N):
        plt.subplot(*dim, i+1)
        algos[i].plot()
    
    plt.subplot(*dim, N + 1)
    plt.title('Performance')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    rewards = [np.array(a.rewards)[np.newaxis,:] for a in algos]
    if rewards[0].size > 0:
        rewards = np.transpose(np.concatenate(tuple(rewards), axis=0))
        plt.plot(np.cumsum(rewards, axis=0))
        plt.legend([a.name for a in algos])
    
    plt.draw()
    plt.pause(0.001)
    
plot_algos()
plt.show()


for e in range(1000):
    for a in algos:
        a.run_episode()
    if e%50 == 0:
        plot_algos()

plot()
input('Press [ENTER] to quit.')
