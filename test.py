import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from rl.gridworld import Gridworld2D


b = np.full((20,20), False)
b[0:5,[2,16]] = True
b[7,4:18] = True
b[2:15,10] = True
b[12:20,14] = True
env = Gridworld2D(20,20,0,0,18,17,blocked=b)
del b


Q = np.full(env.shape + (len(env.actions),), 0.0)

def greedy_policy(state):
    q = Q[state]
    a, = np.where(q == q.max())
    a = a[0] if len(a)==1 else np.random.choice(a, 1)[0]
    return a

def epsilon_policy(state, eps=0.015):
    if np.random.random_sample() < eps:
        return np.random.choice(env.actions, 1)[0]
    else:
        return greedy_policy(state)

def update_Q(s0, a, s1, r, learning_rate=0.05, discount=0.95):
    idx = s0 + (a,)
    value = r + discount * Q[s1].max()
    Q[idx] = (1-learning_rate) * Q[idx] + learning_rate * value


def plot():
    plt.clf()
    env.plot(Q=Q, visits=False)
    plt.draw()
    plt.pause(0.001)
    
matplotlib.use('Qt4agg')
plt.figure()
plt.ion()
plot()
plt.show()
    

for e in range(3000):
    while not env.is_terminal():
        (s0,a,s1,r,_) = env.walk(epsilon_policy(env.state))
        update_Q(s0,a,s1,r)
    env.reset()
    if e%100 == 0:
        plot()
plot()

input('Press [ENTER] to quit.')
