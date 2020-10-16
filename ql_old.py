import numpy as np
from enum import IntEnum, unique
from typing import Tuple, List
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt4agg')

np.set_printoptions(precision=3, suppress=True)

class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

n_actions = len(Action)
n_states = np.array([20, 20])

start_state = np.array([0, 0])
target_state = np.array([18,16])

blocked = np.full(n_states, False)
blocked[1:10, 4] = True
blocked[5,8:15] = True
blocked[13, 11:20] = True
blocked[12:19,10] = True
blocked[17, 14:18] = True
blocked[17:20, 12] = True

def action_delta(action):
    x = y = 0
    if   action == Action.UP:    y +=  1
    elif action == Action.DOWN:  y += -1
    elif action == Action.LEFT:  x += -1
    elif action == Action.RIGHT: x +=  1
    else:
        raise ValueError('Unknown action %d.' % (action))
    return np.array([x, y])

def transition(state, action, chance=True):
    s = state + action_delta(action)
    if chance and np.random.random_sample()<0.2: s[0]+=1
    s = np.maximum(0, np.minimum(n_states-1, s))
    return state if blocked[s[0],s[1]] else s

def is_terminal(state):
    return np.all(state == target_state)

def reward(action, state):
    return float(is_terminal(state)) * 50 - 1

Q = np.zeros((n_actions, n_states[0], n_states[1]))

def greedy_policy(state):
    q = Q[:,state[0],state[1]]
    return np.random.choice(np.argwhere(q == np.amax(q))[:,0], 1)[0]

def eps_policy(state):
    if np.random.random_sample() < 0.015:
        return np.random.randint(n_actions)
    else:
        return greedy_policy(state)

def update_plot(avg_q_hist, n_steps, n_visited, episodes):
    if not update_plot.active:
        plt.figure()
        plt.ion()
        plt.show()
        update_plot.active = True
    plt.clf()
    
    plt.subplot(2,2,1)
    plt.plot(avg_q_hist)
    plt.ylabel('Average Value-Function')
    plt.xlabel('Episode')
    
    plt.subplot(1,2,2)
    img = plt.imshow(np.transpose(n_visited))
    plt.colorbar(img)
    
    b = np.transpose(np.invert(blocked).astype(float))[:,:,np.newaxis]
    b = np.concatenate((b,b,b,1-b),axis=2)
    plt.imshow(b)
    
    x = np.arange(n_states[0])
    y = np.arange(n_states[1])
    
    for i in x:
        for j in y:
            s = np.array([i,j])
            if blocked[i][j] or is_terminal(s):
                continue
            d = action_delta(greedy_policy(s)) * 0.8
            plt.arrow(i-d[0]/2,j-d[1]/2,d[0],d[1],
                      shape='full',
                      color='w',
                      length_includes_head=True,
                      width=0.05,
                      head_width=0.15)

    for i in range(max(0,len(episodes)-16),len(episodes)):
        h = np.array(episodes[i])
        plt.plot(h[:,0],h[:,1], color='r', alpha=0.2)
        
    plt.plot(*start_state, marker='o', color='r', ls='', markersize=7, fillstyle='none')
    plt.plot(*target_state, marker='x', color='r', ls='', markersize=7)
            
    plt.xticks(x-0.5)
    plt.yticks(y-0.5)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.xlim([x[0]-0.5, x[len(x)-1]+0.5])
    plt.ylim([y[0]-0.5, y[len(y)-1]+0.5])
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(2,2,3)
    N = 1
    ma = np.cumsum(np.insert(n_steps, 0, 0)) 
    ma = (ma[N:] - ma[:-N]) / float(N)
    plt.plot(ma)
    plt.ylim([0,250])
    plt.ylabel('Path Length')
    plt.xlabel('Episode')
    
    plt.draw()
    plt.pause(0.001)
update_plot.active = False

n_episodes = 500
learning_rate = 0.05
discount_factor = 0.95
policy = eps_policy
avg_q = []
episodes = []
n_steps = []
n_visited = np.zeros(n_states)
for e in range(n_episodes):
    s0 = start_state
    history = [s0]
    while not is_terminal(s0):
        a = policy(s0)
        s1 = transition(s0, a)
        r = reward(a, s1)
        Q[a,s0[0],s0[1]] += learning_rate * (r + discount_factor * Q[policy(s1),s1[0],s1[1]] - Q[a,s0[0],s0[1]])
        history.append(s1)
        s0 = s1
    episodes.append(history)
    n_steps.append(len(history))
    for s in history:
        n_visited[s[0],s[1]] += 1
    avg_q.append(np.mean(np.amax(Q,axis=0)))
    if e % 100 == 0:
        update_plot(avg_q, n_steps, n_visited, episodes)
        episodes = []
    if e % 2000 == 0:
        n_visited = np.zeros(n_states)
update_plot(avg_q,  n_steps, n_visited, episodes)
input("Press [ENTER] to continue.\n")
