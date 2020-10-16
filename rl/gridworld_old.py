import numpy as np

import matplotlib
import matplotlib.pyplot as plt


class GridWorld:
    
    
    def __init__(self, shape, target, blocked=None, wind=(0, np.array([0, 0]))):
        self._shape = shape
        self._target = target
        if blocked is None: blocked = np.full(self._shape, False)
        self._blocked = blocked
        (self._wind_prob, self._wind_dir) = wind
    
    
    def get_state_space(self):
        return self._shape
    
    
    def get_action_space(self):
        return (4,)
    
    
    def transition(self, s0, a):
        s1 = np.array(s0) + self._action_delta(a)
        if self._wind_prob > 0  and np.random.random_sample() < self._wind_prob:
            print('its windy')
            s1 += self._wind_dir
        s1 = np.maximum(0, np.minimum(s1, np.array(self._shape)-1))
        s1 = tuple(s1.astype(np.int))
        if self._blocked[s1]: s1 = s0
        return s1
    
    
    def is_terminal(self, state):
        return state == self._target


    def plot(self, Q, counts=None, episodes=[], start=None):
        x_axis = np.arange(self._shape[0])
        y_axis = np.arange(self._shape[1])
        
        if counts is not None:
            ax = plt.imshow(np.transpose(counts))
            plt.colorbar(ax)
        
        b = np.transpose(np.invert(self._blocked).astype(float))[:,:,np.newaxis]
        b = np.concatenate((b,b,b,1-b), axis=2)
        plt.imshow(b)
        
        plt.xticks(x_axis - 0.5)
        plt.yticks(y_axis - 0.5)
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        plt.xlim([-0.5, self._shape[0]-0.5])
        plt.ylim([-0.5, self._shape[1]-0.5])
        
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        
        arrow_col = 'k' if counts is None else 'w'
        for x in x_axis:
            for y in y_axis:
                s = (x,y)
                if self._blocked[x,y] or self.is_terminal(s): continue
                d = self._action_delta(Q.max_action(s)) * 0.8
                plt.arrow(x-d[0]/2, y-d[1]/2, d[0], d[1],
                          shape='full', color=arrow_col,
                          width=0.05, head_width=0.15, length_includes_head=True)
        
        for e in episodes:
            path = [step[0] for step in e]
            path.append(e[len(e)-1][2])
            path = np.array(path)
            plt.plot(path[:,0], path[:,1], color='r', alpha=0.2)
        
        plt.plot(*self._target, marker='x', color='r', ls='', markersize=7)
        if start is not None: plt.plot(*start, marker='o', color='r', ls='', markersize=7, fillstyle='none')
        
    
    def _action_delta(self, a):
        b = a[0]*np.pi/2
        return np.round([np.cos(b), np.sin(b)])


