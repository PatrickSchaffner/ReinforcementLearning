import gym
import numpy as np

from .. import Environment, DiscreteSpace

class Cartpole(Environment):
    
    def __init__(self, discretize_num=11):
        self.env = None
        self.reset()
        super().__init__(DiscreteSpace(tuple(np.full((4,), discretize_num))),
                         DiscreteSpace((2,)))
        self._position = Discretizer(1.5, discretize_num)
        self._velocity = Discretizer(2.0,  discretize_num)
        self._angle    = Discretizer(0.4,  discretize_num)
        self._rotation = Discretizer(2.5,  discretize_num)
    
    def reset(self):
        if self.env is None:
            self.env = gym.make('CartPole-v0')
        self.state = self.env.reset()
        self.terminal = False
    
    def is_terminal(self):
        return self.terminal
    
    @property
    def current_state(self):
        return self._discretize(*self.state)
    
    def execute(self, action):
        (self.state, reward, self.terminal, _) = self.env.step(action[0])
        return reward
    
    def render(self):
        self.env.render()
        
    def close(self):
        self.env.close()
        self.env = None
        self.state = None
        self.terminal = True
    
    def _discretize(self, pos, vel, ang, rot):
        return (self._position(pos),
                self._velocity(vel),
                self._angle(ang),
                self._rotation(rot))


class Discretizer(object):
    
    def __init__(self, mag, num):
        
        self.low = -mag
        self.high = mag
        self.num = num
        
        if self.num % 2 == 0:
            ax = np.power(np.linspace(0, 1, self.num / 2 + 1), 1)
            ax = np.concatenate((-ax[1:][::-1], ax), axis=0)
        else:
            ax = np.power(np.linspace(0, 1, np.int(np.floor(self.num / 2))), 1)
            ax[0] = ax[1] / 3
            ax = np.concatenate((-ax[::-1], ax), axis=0)
        ax -= ax[0]
        ax *= (self.high - self.low) / ax[-1]
        ax += self.low
        self.buckets = ax
    
    def __call__(self, value):
        if value < self.low:
            return 0
        elif value >= self.high:
            return self.num - 1
        else:
            return np.int(np.digitize(value, self.buckets))

            
            
        
