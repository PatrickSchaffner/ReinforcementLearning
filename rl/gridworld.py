from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt


class Gridworld2D():
    
    
    def __init__(self, dim_x, dim_y, start_x, start_y, target_x, target_y, step_penalty=1, target_reward=None, memory_size=16, blocked=None):
        
        self.shape = (dim_x, dim_y)
        self.start = (start_x, start_y)
        self.target = (target_x, target_y)
        self.actions = IntEnum('Action2D', ['RIGHT', 'UP', 'LEFT', 'DOWN'], start=0)
        self.step_penalty = step_penalty
        if target_reward is None:
            target_reward = 2*np.sum(np.array(self.shape))
        self.target_reward = target_reward
        
        self.visits = np.zeros(self.shape, dtype=np.int)
        self.paths = np.full((memory_size,), None)
        self.path_idx = 0
        self.current_history = []
        self.rewards = []
        self.path_length = []
            
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
        
        step = (s0, action, s1, r, t)
        self.current_history.append(step)
        return step
    
    
    def reset(self, save_episode=True, only_terminal=True):
        self.state = self.start
        if save_episode and len(self.current_history) > 0 and (not only_terminal or self.current_history[-1][4]):
            self.record_episode(self.current_history)
        self.current_history = []
    
    
    def action_delta(self, action):
        a = int(action)*np.pi/2.0
        return np.round([np.cos(a), np.sin(a)])
    
    
    def record_episode(self, history):
        p = Gridworld2D.to_path(history)
        self.visits[p[:,0], p[:,1]] += 1
        self.paths[self.path_idx] = p
        self.path_idx += 1
        self.path_idx %= len(self.paths)
        self.rewards.append(np.sum([h[3] for h in history]))
        self.path_length.append(p.shape[0]-1)
    
    
    def plot(self, Q=None, policy=None, episodes=True, visits=True, blocked=True, start=True, target=True, state=False, rewards=True):
        
        if rewards:
            plt.subplot(1,2,1)
            plt.title('Performance')
            plt.xlabel('Episodes')
            
            col = 'b'
            ax = plt.gca()
            ax.plot(np.cumsum(self.rewards), color=col)
            ax.set_ylabel('Cumulative Reward', color=col)
            
            col = 'r'
            ax = ax.twinx()
            ax.plot(self.path_length, color=col)
            ax.set_ylabel('Path Length', color=col)
            
            plt.subplot(1,2,2)
        
        if isinstance(visits, bool):
            visits = self.visits if visits else None
        if visits is None and Q is not None:
            visits = Q.max(axis=2)
        if visits is not None:
            ax = plt.imshow(np.transpose(visits))
            plt.colorbar(ax)
        
        if isinstance(blocked, bool):
            blocked = self.blocked if blocked else None
        if blocked is not None:
            b = np.transpose(np.invert(self.blocked).astype(float))[:,:,np.newaxis]
            b = np.concatenate((b,b,b,1-b), axis=2)
            plt.imshow(b)
        
        (x_dim, y_dim) = self.shape
        x_axis = np.arange(x_dim)
        y_axis = np.arange(y_dim)
        plt.xticks(x_axis - 0.5)
        plt.yticks(y_axis - 0.5)
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.xlim([-0.5, x_dim-0.5])
        plt.ylim([-0.5, y_dim-0.5])
        plt.grid(True)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Policy')
        
        if policy is None and Q is not None:
            policy = Q.argmax(axis=-1)
        if policy is not None:
            arrow_col = 'w' if visits is not None else 'k'
            for x in x_axis:
                for y in y_axis:
                    s = (x, y)
                    if self.is_blocked(s) or self.is_terminal(s):
                        continue
                    d = self.action_delta(policy[s]) * 0.8
                    plt.arrow(x-d[0]/2, y-d[1]/2, d[0], d[1],
                            shape='full', color=arrow_col,
                            width=0.05, head_width=0.15, length_includes_head=True)
        
        if isinstance(episodes, bool):
            if episodes:
                episodes = [p for p in self.paths if p is not None]
                episodes = None if len(episodes) == 0 else episodes
            else:
                episodes = None
        elif episodes is not None:
            episodes = [Gridworld2D.to_path(history) for history in episodes]
        if episodes is not None:
            for p in episodes:
                plt.plot(p[:,0], p[:,1], color='r', alpha=0.2)
        
        if isinstance(start, bool):
            start = self.start if start else None
        if start is not None:
            plt.plot(*start, marker='o', color='r', ls='', markersize=7, fillstyle='none')
        
        if isinstance(target, bool):
            target = self.target if target else None
        if target is not None:
            plt.plot(*target, marker='x', color='r', ls='', markersize=7)
        
        if isinstance(state, bool):
            state = self.state if state else None
        if state is not None:
            plt.plot(*state, marker='o', color='r', ls='', markersize=5, fillstyle='full')
    
    
    @staticmethod
    def to_path(history):
        if isinstance(history, list):
            history = np.array(history)
        if history.ndim > 1:
            history = np.array([h[0] for h in history] + [history[-1][2]])
        return history
        
