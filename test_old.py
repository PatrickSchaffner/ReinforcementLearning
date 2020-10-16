import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from rl.ql import ActionValueFunc, QLAlgo
from rl.gridworld import GridWorld

dim = (20, 20)
blocked = np.full(dim, False)
blocked[1:10, 4] = True
blocked[5,8:15] = True
blocked[13, 11:20] = True
blocked[12:19,10] = True
blocked[17, 14:18] = True
blocked[17:20, 12] = True
world = GridWorld(dim, (18,17), blocked=blocked)

def start_state():
    return (0,0)
    #state = None
    #while state is None or world.is_terminal(state):
    #    state = tuple(np.random.randint(0, s) for s in world.get_state_space())
    #return state

def reward(s0, a, s1):
    return float(world.is_terminal(s1)) * 20 - 1

Q = ActionValueFunc(world.get_state_space(), world.get_action_space(), learning_rate=0.20)

matplotlib.use('Qt4agg')
plt.figure()
plt.ion()
world.plot(Q)
#plt.show()



algo = QLAlgo(Q, start_state, world.is_terminal, world.transition, reward, discount_rate=0.95)
episodes = []
counts = np.zeros(world.get_state_space())
steps = []

def plot_world():
    plt.clf()
    
    plt.subplot(1,2,1)
    world.plot(Q, episodes=episodes, counts=counts, start=(0,0))
    plt.title('%d episodes, %d steps' % (i, sum(counts.flat)))
    
    plt.subplot(1,2,2)
    plt.plot(np.cumsum(steps))
    plt.xlabel('Episodes')
    plt.ylabel('Total steps')
    plt.title('Path length')
    
    plt.draw()
    plt.pause(0.001)

for i in range(25000):
    history = algo.run_episode()
    
    episodes.append(history)
    for h in history:
        (x,y) = h[0]
        counts[x,y] += 1
    (x,y) = history[len(history)-1][2]
    counts[x,y] += 1
    
    steps.append(len(history))
    if len(history)<20:
        print(np.transpose(history))
        print('------')
    
    if i < 50 or i % 50 == 0:
        plot_world()
        episodes = []
        counts = np.zeros(world.get_state_space())

plot_world()
input('Press any key to continue.')
