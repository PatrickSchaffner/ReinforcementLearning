from enum import Enum, IntEnum, unique, auto
from typing import Tuple, List
from collections import namedtuple
from math import inf


@unique
class Cell(Enum):
    BLOCKED = auto()
    OPEN = auto()
    TERMINAL = auto()


@unique
class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    WAIT = 4


class Map:
    
    def __init__(self, width: int, height: int, start: Tuple[int, int], target: Tuple[int, int], blocked: List[Tuple[int, int]] = []):
        self.width = width
        self.height = height
        self.start = start
        
        self._grid = [[Cell.OPEN for i in range(self.width)] for j in range(self.height)]
        
        for i in range(len(blocked)):
            self._grid[blocked[i][0]][blocked[i][1]] = Cell.BLOCKED
            
        (r,c) = target
        if self._grid[r][c] is Cell.BLOCKED:
            raise ValueError('Target cell (%d,%d) is blocked.' % target)
        self._grid[r][c] = Cell.TERMINAL
        
        (r,c) = self.start
        if self._grid[r][c] is not Cell.OPEN:
            raise ValueError('Start cell (%d,%d) is %s.' % start + ('blocked' if self._grid[r][c] == Cell.BLOCKED else 'terminal',))
    
    def is_open(self, cell):
        return self._grid[cell[0]][cell[1]] is not Cell.BLOCKED
    
    def is_terminal(self, cell):
        return self._grid[cell[0]][cell[1]] is Cell.TERMINAL
    
    def transition(self, position, action):
        (r,c) = position
        next = None
        if action == Action.WAIT:
            next = position
        elif action is Action.UP:
            next = (r-1, c)
        elif action == Action.DOWN:
            next = (r+1, c)
        elif action == Action.LEFT:
            next = (r, c-1)
        elif action == Action.RIGHT:
            next = (r, c+1)
        else:
            raise ValueError('Action %s is invalid.' % (action))
        next = self._trim(next)
        if not self.is_open(next):
            next = position
        return next
    
    def _trim(self, position):
        return (max(0, min(self.height-1, position[0])), max(0, min(self.width-1, position[1])))


class Reward:
    
    def __init__(self, map: Map, goal_reward: float = 10, step_penalty: float = 1):
        self._map = map
        self._goal_reward = goal_reward
        self._step_penalty = step_penalty
    
    def __call__(self, action: Action, position: Tuple[int, int]):
        return float(self._map.is_terminal(position)) * self._goal_reward - float(action is not Action.WAIT) * self._step_penalty


class ActionValueFunction:
    
    def __init__(self, map: Map, initial: float = 0):
        self._value = [[[initial for i in range(map.width)] for j in range(map.height)] for a in range(len(Action))]
    
    def __getitem__(self, key: Tuple[Action, Tuple[int, int]]):
        return self._value[key[0]][key[1][0]][key[1][1]]
    
    def __setitem__(self, key: Tuple[Action, Tuple[int, int]], value: float):
        self._value[key[0]][key[1][0]][key[1][0]] = value
    
    def policy(self, position):
        action = None
        value = -inf
        for a in Action:
            v = self[a, position]
            if v <= value:
                continue
            action = a
            value = v
        return value, action


Step = namedtuple('Step', ['initial_position', 'action', 'reward', 'end_position'])


class Episode:
    
    def __init__(self):
        self._steps = []
    
    def length():
        return len(self._steps)
    
    def __getitem__(self, key):
        return self._steps[key]
    
    def append(self, step: Step):
        self._steps.append(step)

class QLearning:
    
    def __init__(self, map: Map, reward: Reward, learning_rate: float = 0.05, discount_factor: float = 0.9, avf: ActionValueFunction = None):
        if avf is None:
            avf = ActionValueFunction(map)
        self._map = map
        self._reward = reward
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._avf = avf

    def run_episode(self):
        position = self._map.start
        while not self._map.is_terminal(position):
            (_, action) = self._policy(position)
            next = self._map.transition(position, action)
            reward = self._reward(action, next)
            self._update(action, position, reward, next)
            position = next
    
    def _policy(self, position):
        return self._avf.policy(position)
    
    def _update(self, action, position, reward, next):
        (av_next, _) = self._avf.policy(next)
        self._avf[action, position] += self._learning_rate * (reward + self._discount_factor * av_next - self._avf[action, position])
        

map = Map(width = 5,
          height = 5,
          start = (1, 1),
          target = (4, 4),
          blocked = [(1, 2), (3, 3), (3,4), (0, 3)])

reward = Reward(map)

learner = QLearning(map, reward)
learner.run_episode()

