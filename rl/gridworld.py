from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt

from .ql import Environment, QFunction


class Gridworld(Environment):
    
    
    def __init__(self, dim, start, target, step_penalty=1, target_reward=None, blocked=None):
        self.shape = dim
        self.start = start
        self.target = target
        self.actions = list(IntEnum('GridworldAction', ['RIGHT', 'UP', 'LEFT', 'DOWN'], start=0))
        self.step_penalty = step_penalty
        if target_reward is None:
            target_reward = 2*np.sum(np.array(self.shape))
        self.target_reward = target_reward
            
        if blocked is None:
            blocked = np.full(self.shape, False)
        else:
            blocked = np.array(blocked)
        if blocked.dtype == np.int:
            idx = blocked
            blocked = np.full(self.shape, False)
            blocked[idx[:,0], idx[:,1]] = True
        self.blocked = blocked
        
        self.reset()
        
    def state_shape(self):
        return self.shape
    
    def action_shape(self):
        return np.shape(self.actions)
    
    def is_terminal(self, state=None):
        if state is None:
            state = self.state
        return state == self.target
    
    def is_blocked(self, state):
        return self.blocked[state]
    
    
    def walk(self, action):
        
        s0 = self.state
        
        d = self.action_delta(action)
        s1 = s0 + d
        s1 = np.maximum(0, np.minimum(np.array(self.shape)-1, s1))
        s1 = tuple(s1.astype(np.int))
        if self.is_blocked(s1):
            s1 = s0
        
        t = self.is_terminal(s1)
        r = np.float(t) * self.target_reward - self.step_penalty
        
        self.state = s1
        return (s0, action, s1, r, t)
    
    
    def reset(self):
        self.state = self.start
    
    
    def action_delta(self, action):
        a = int(action)*np.pi/2.0
        return np.round([np.cos(a), np.sin(a)])
    
    def current_state(self):
        return self.state
    
    def execute(self, action):
        (_, _, _, r, _) = self.walk(action[0])
        return r
    

def plot_gridworld(grid:Gridworld,
                   Q:QFunction=None,
                   episodes=None,
                   blocked=True,
                   start=True,
                   target=True,
                   state=False,
                   colorbar=True):
        
    (x_dim, y_dim) = grid.shape
    x_axis = np.arange(x_dim)
    y_axis = np.arange(y_dim)
    
    if Q is not None:
        q = np.array([[Q.max_value((x, y)) for y in y_axis] for x in x_axis])
        ax = plt.imshow(np.transpose(q))
        if colorbar:
            plt.colorbar(ax)
    
    if isinstance(blocked, bool):
        blocked = grid.blocked if blocked else None
    if blocked is not None:
        b = np.transpose(np.invert(grid.blocked).astype(float))[:,:,np.newaxis]
        b = np.concatenate((b,b,b,1-b), axis=2)
        plt.imshow(b)
    
    plt.xticks(x_axis - 0.5)
    plt.yticks(y_axis - 0.5)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.xlim([-0.5, x_dim-0.5])
    plt.ylim([-0.5, y_dim-0.5])
    plt.grid(True)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Policy')
    
    if Q is not None:
        policy = np.array([[Q.max_action((x, y))[0] for y in y_axis] for x in x_axis])
        for x in x_axis:
            for y in y_axis:
                s = (x, y)
                if grid.is_blocked(s) or grid.is_terminal(s):
                    continue
                d = grid.action_delta(policy[s]) * 0.8
                plt.arrow(x-d[0]/2, y-d[1]/2, d[0], d[1],
                        shape='full', color='w',
                        width=0.05, head_width=0.15, length_includes_head=True)
    
    def to_path(history):
        if isinstance(history, list):
            history = np.array(history)
        if history.ndim > 1:
            history = np.array([h[0] for h in history] + [history[-1][2]])
        return history
    if episodes is not None and len(episodes) > 0:
        episodes = [to_path(history) for history in episodes]
        alpha = 1.0 / (1 + np.log(len(episodes)))
        for p in episodes:
            plt.plot(p[:,0], p[:,1], color='r', alpha=alpha)
    
    if isinstance(start, bool):
        start = grid.start if start else None
    if start is not None:
        plt.plot(*start, marker='o', color='r', ls='', markersize=7, fillstyle='none')
    
    if isinstance(target, bool):
        target = grid.target if target else None
    if target is not None:
        plt.plot(*target, marker='x', color='r', ls='', markersize=7)
    
    if isinstance(state, bool):
        state = grid.state if state else None
    if state is not None:
        plt.plot(*state, marker='o', color='r', ls='', markersize=5, fillstyle='full')

