import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 设计FIR滤波器
numtaps = 51   # 滤波器阶数
cutoff = 0.1   # 截止频率
nyq_freq = 0.5 # 采样率的一半
taps = signal.firwin(numtaps, cutoff, nyq=nyq_freq)

# 绘制滤波器系数响应
plt.stem(taps)
plt.title('FIR Filter Coefficients')
plt.xlabel('Tap Number')
plt.ylabel('Coefficient Value')
plt.show()

# 生成输入信号
fs = 1000  # 采样率
t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*200*t)

# 使用FIR滤波器处理信号
y = signal.convolve(x, taps, mode='same')

# 绘制输入和输出信号
plt.plot(t, x, label='Input Signal')
plt.plot(t, y, label='Output Signal')
plt.title('FIR Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()