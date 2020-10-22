import numpy as np

import matplotlib.pyplot as plt

from rl.gym import Cartpole
from rl.ql import DynaQ, QLAlgo, TabularQFunction, epsilon_greedy


epsilon = 0.05
discretize_num = 51

def params(algoType):
    p = {}
    if issubclass(algoType, QLAlgo):
        p = {**p,
             'learning_rate': 0.3,
             'discount_factor': 0.975}
    if issubclass(algoType, DynaQ):
        p = {**p,
             'batch_size': 128,
             'memory_size': 512}
    return p

algos = [(QLAlgo, {'learning_rate': 0}),
         DynaQ,
         QLAlgo]


class AlgoInstance():
    
    def __init__(self, env, Q, algo):
        
        self.env = env
        self.Q = Q
        self.algo = algo
        self.rewards = []
        
        self.name = type(self.algo).__name__
        if isinstance(self.algo, QLAlgo):
            self.name += ' [lr=%.2f' % (self.algo.learning_rate)
            if isinstance(self.algo, DynaQ):
                self.name += ', bs=%d' % (self.algo.batch_size)
        self.name += ']'
    
    def run_episode(self, render=False):
        args = {} if not render else {'after_step_func': lambda: self.render()}
        history = self.algo.run_episode(**args)
        r = np.sum([r for (_, _, _, r, _) in history])
        self.rewards.append(r)
    
    def render(self):
        self.env.render()

def create_instance(algoType):
    
    if isinstance(algoType, list):
        return [create_instance(a) for a in algoType]
    if isinstance(algoType, tuple):
        p = algoType[1]
        algoType = algoType[0]
    else:
        p = {}
        
    env = Cartpole(discretize_num=discretize_num)
    Q = TabularQFunction(env, initial_value=0)
    behavior = epsilon_greedy(env.action_shape(), epsilon=epsilon)
    algo = algoType(env=env, Q=Q, **{**params(algoType), **p})
    return AlgoInstance(env, Q, algo)

algos = create_instance(algos)


#matplotlib.use('Qt4agg')
plt.figure()
plt.ion()
plt.show()

best_i = 0
for e in range(500):
    
    for i in range(len(algos)):
        algos[i].run_episode(render=(i == best_i and e % 10 == 0))
    
    if not (e % 50 == 0):
        continue
    
    plt.clf()
    plt.title('Performance')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    
    rewards = [np.array(a.rewards)[np.newaxis,:] for a in algos]
    if rewards[0].size > 0:
        rewards = np.transpose(np.concatenate(tuple(rewards), axis=0))
        best_i = np.transpose(rewards[-5:, :].mean(axis=0)).argmax()
        plt.plot(np.cumsum(rewards, axis=0))
        plt.legend([a.name for a in algos])
        
    plt.draw()
    plt.pause(0.001)


input("Press [Enter] to quit.")
for a in algos:
    a.env.close()
