from . import UpdateRule


class QLearnUpdate(UpdateRule):

    def __init__(self, learning_rate = 0.15, discount_rate = 0.9):
        self.set(learning_rate=learning_rate, discount_rate=discount_rate)

    def set(self, learning_rate=None, discount_rate=None):
        if learning_rate is not None: self._learn = learning_rate
        if discount_rate is not None: self._discount = discount_rate

    def update(self, avf, s0, a, s1, r):
        v0 = avf[s0,a]
        v1 = avf.maxValue(s1)
        avf[s0,a] = (1-self._learn)*v0 + self._learn*(r + self._discount*v1)
        return avf

