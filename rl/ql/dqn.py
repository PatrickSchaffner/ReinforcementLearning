import numpy as np

from tensorflow.kears import Model
from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.models import Sequential

from .. import Environment
from . import QFunction


class QNetwork(QFunction):
    
    def __init__(self, state_shape, action_shape):
        super().__init__()
        
        self.model = Sequential([
            Flatten(input_shape=self.state_shape),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(np.prod(self.action_shape), activation='linear'),
            Reshape(target_shape=self.action_shape)
        ])
        self.model.compile(optimizer='adam', loss='mse')
    
    def predict(self, X):
        return np.array(self.model(X))
    
    def train(self, X, y):
        self.model.train_on_batch(X, y)
    
    def max(self, state):
        pass
    
    def evaluate(self, state, action):
        
    
    def update(self, state, action, delta):
        pass
