import numpy as np
from scipy.signal import iirfilter, lfilter

class LP_delay():
    def __init__(self, wc, ts):
        self._alpha = 1/(wc*ts+1)
        self._yk_ = 0
    
    def update(self, xk):
        yk = self._alpha*self._yk_ + (1-self._alpha)*xk
        self._yk_ = yk
        return yk

lp = LP_delay(25, 0.01)

t = np.linspace(0,2, num=200)
x = np.sin(2*np.pi*5*t)
y = []
for i in range(len(x)):
    y.append(lp.update(x[i]))
y = np.array(y)


# 绘制滤波前后的信号
import matplotlib.pyplot as plt
plt.plot(t, x)
plt.plot(t, y)
plt.show()