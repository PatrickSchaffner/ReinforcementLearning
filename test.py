import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from rl.gridworld import Gridworld, plot_gridworld
from rl.ql import QLAlgorithm, TabularQFunction, epsilon_greedy

dim = (20, 20)
start = (0, 0)
target = (18, 17)

b = np.full(dim, False)
b[0:5,[2,16]] = True
b[7,4:18] = True
b[4:15,10] = True
b[12:20,14] = True


class AlgoLab():
    
    def __init__(self, grid, Q, algo):
        self.grid = grid
        self.Q = Q
        self.algo = algo
        self.episodes = []
        self.rewards = []
    
    def run_episode(self):
        history = self.algo.run_episode()
        self.episodes.append(history)
        r = np.sum([h[3] for h in history])
        self.rewards.append(r)
    
    def plot(self):
        plot_gridworld(self.grid,
                       Q=self.Q,
                       episodes=self.episodes)
        self.episodes = []
    
    def create(learning_rate):
        if isinstance(learning_rate, list):
            return [AlgoLab.create(lr) for lr in learning_rate]
        grid = Gridworld(dim, start, target, blocked=b)
        Q = TabularQFunction(dim, (len(grid.actions),), initial_value=0)
        behavior = epsilon_greedy(grid.actions, epsilon=0.015)
        algo = QLAlgorithm(env=grid,
                           Q=Q,
                           behavior=behavior,
                           learning_rate=learning_rate,
                           discount_factor=0.975)
        return AlgoLab(grid, Q, algo)

algos = AlgoLab.create(learning_rate=[0.05, 0.1, 0.2, 0.3])

def plot():
    plt.clf()
    N = len(algos)
    
    plt.subplot(2, 1, 2)
    plt.title('Performance')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    rewards = [np.array(a.rewards)[np.newaxis,:] for a in algos]
    if rewards[0].size > 0:
        rewards = np.transpose(np.concatenate(tuple(rewards), axis=0))
        plt.plot(rewards)
        plt.legend(['lr=%.2f' % (a.algo.learning_rate) for a in algos])
    
    for i in range(N):
        plt.subplot(2, N, i+1)
        algos[i].plot()
    plt.draw()
    plt.pause(0.001)
    
#matplotlib.use('Qt4agg')
plt.figure()
plt.ion()
plot()
plt.show()

for e in range(3000):
    for a in algos:
        a.run_episode()
    if e%100 == 0:
        plot()

plot()
input('Press [ENTER] to quit.')
