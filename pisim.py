import numpy as np
import matplotlib.pyplot as plt

class GARCH:
    
    def __init__(self,w,a,b,dist=lambda n: np.random.default_rng().standard_normal(n)):
        self._p = len(a)
        self._q = len(b)
        self._w = w
        self._a = np.array(a)
        self._b = np.array(b)
        self._dist = dist
        self._n = max(self._p, self._q)
        lr_var = self._w / (1 - np.sum(self._a) - np.sum(self._b))
        self._sqres = np.repeat(lr_var, self._p)
        self._var = np.repeat(lr_var, self._q)
        self._cond_var = lr_var
    
    def _updateNextPeriod(self, residual):
        self._sqres = np.concatenate(([residual * residual], self._sqres[:self._p-1]))
        self._var = np.concatenate(([self._cond_var], self._var[:self._q-1]))
        self._cond_var = self._w + np.sum(self._a * self._sqres) + np.sum(self._b * self._var)
    
    def getConditionalVolatility(self):
        return np.sqrt(self._cond_var)
    
    def drawNextResiduals(self, n=1):
        vola = self.getConditionalVolatility()
        z = self._dist(n)
        return (z * vola, vola)
    
    def simulate(self, steps=1):
        residuals = []
        volatility = []
        for s in range(steps):
            z = self._dist(1)
            vol = self.getConditionalVolatility()
            volatility.append(vol)
            res = z * vol
            residuals.append(res)
            self._updateNextPeriod(res)
        return (np.array(residuals), np.array(volatility))
    
    def update(self, residuals):
        volatility = []
        for r in residuals:
            self._updateNextPeriod(r)
            volatility.append(self.getConditionalVolatility())
        return np.array(volatility)


class PriceImpact:
    
    def __init__(self, direction, size, volatility):
        self._direction = direction
        self._size = size
        self._volatility = volatility

    def compute(self, order, volatility):
        if np.abs(order)<=10*np.finfo(float).eps:
            return 0
        sgn = np.sign(order)
        volume = order / sgn
        impact = self._volatility * volatility * ( sgn * self._direction + sgn * np.sqrt(volume) * self._size )
        return impact
    

class MarketSimulation:
    
    def __init__(self, p0=100, return_model=GARCH(0.000001, [0.15, 0.02], [0.3, 0.1, 0.05]), price_impact_model=PriceImpact(0.01,0.001,1)):
        self._return = return_model
        self._price_impact = price_impact_model
        self._p = p0
    
    def getPrice(self):
        return self._p
    
    def computePriceImpact(self, order):
        return self._price_impact.compute(order, self.getVolatility())
    
    def getVolatility(self):
        return self._return.getConditionalVolatility()
    
    def simulate(self, order):
        (r, vola) = self._return.drawNextResiduals()
        dp = self._p * ( np.exp(r) - 1 ) + self._price_impact.compute(order, vola)
        self._return.update(np.log(1 + dp/self._p))
        self._p = self._p + dp
        return dp
        

sim = MarketSimulation(p0=1.08)
n = 1000
orders = np.random.normal(0,10,n)
orders[np.random.standard_normal(n)<-0.5] = 0
orders = np.zeros(n)
p = np.zeros(n)
v = np.zeros(n)
pi = np.zeros(n)
for i in range(n):
    v[i] = sim.getVolatility()
    pi[i] = sim.computePriceImpact(orders[i])
    sim.simulate(orders[i])
    p[i] = sim.getPrice()

plt.subplot(2,2,1)
plt.plot(p)
plt.title('Price')
plt.subplot(2,2,3)
plt.plot(v)
plt.title('Volatility')
plt.subplot(2,2,2)
plt.bar(range(n), orders)
plt.title('Orders')
plt.subplot(2,2,4)
plt.bar(range(n), pi);
plt.title('Price Impact');
plt.show()



        
    
        
        
