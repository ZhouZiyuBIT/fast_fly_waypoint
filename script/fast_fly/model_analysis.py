import numpy as np
import scipy.spatial.transform as spt
import scipy.optimize as opt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os, sys
BASEPATH = os.path.abspath(__file__).split('script/', 1)[0]+'script/fast_fly/'
sys.path += [BASEPATH]

from trajectory import StateLoader
from lpf import fft_filter


state_ld = StateLoader(BASEPATH+"results/real_flight1.csv")

print("length",state_ld._N)

velocity = state_ld._vel[6000:12000]
velocity_rate = np.linalg.norm(velocity, axis=1)
acceleration = state_ld._acc[6000:12000]
quaternion = state_ld._quaternion[6000:12000]
thrust = state_ld._u[6000:12000,0]
angular_rate_set = state_ld._u[6000:12000,1:4]
angular_rate = state_ld._omega[6000:12000]

# transform to body frame
velocity_body = []
for i in range(6000):
    q = [-quaternion[i,1], -quaternion[i,2], -quaternion[i,3], quaternion[i,0]]
    R = spt.Rotation.from_quat(q).as_matrix()
    velocity_body.append(np.dot(R, velocity[i]))
velocity_body = np.array(velocity_body)

acceleration[:,0] = fft_filter(acceleration[:,0], 2)
velocity_body[:,0] = fft_filter(velocity_body[:,0], 2)

acceleration[:,1] = fft_filter(acceleration[:,1], 1.5)
velocity_body[:,1] = fft_filter(velocity_body[:,1], 1.5)

acceleration[:,2] = fft_filter(acceleration[:,2], 8)
thrust = fft_filter(thrust, 8)
velocity_body[:,2] = fft_filter(velocity_body[:,2], 8)

plt.figure("angular_rate vs angular_rate_set")
plt.plot(angular_rate[:,0], 'r.')
plt.plot(angular_rate_set[:,0], 'g.')

plt.figure("velocity")
plt.plot(velocity_rate)

plt.figure("acc_x about vel_x")
plt.scatter(velocity_body[:,0], acceleration[:,0], alpha=0.2, c='r')
def func(x, a):
    return a*x
popt, pcov = opt.curve_fit(func, velocity_body[:,0], acceleration[:,0])
print("x drag coeff", popt)
plt.plot(velocity_body[:,0], func(velocity_body[:,0], *popt))

plt.figure("acc_y about vel_y")
plt.scatter(velocity_body[:,1], acceleration[:,1], alpha=0.2, c='r')
popt, pcov = opt.curve_fit(func, velocity_body[:,1], acceleration[:,1])
print("y drag coeff", popt)
plt.plot(velocity_body[:,1], func(velocity_body[:,1], *popt))

fig = plt.figure("acc_z about vel_z, thrust")
plt3d = fig.add_subplot(111, projection='3d')
plt3d.set_xlabel('z velocity')
plt3d.set_ylabel('thrust')
plt3d.set_zlabel('z acceleration')
plt3d.scatter(velocity_body[:,2], thrust , acceleration[:,2], alpha=0.2, c='r')
def func2(x, a, b):
    return a*x[0] + b*x[1]
popt, pcov = opt.curve_fit(func2, np.array([velocity_body[:,2], thrust]), acceleration[:,2])
print("z drag coeff", popt[0])
print("thrust coeff", popt[1:])
X = np.linspace(velocity_body[:,2].min(), velocity_body[:,2].max(), 100)
Y = np.linspace(thrust.min(), thrust.max(), 100)
X, Y = np.meshgrid(X, Y)
Z = func2([X, Y], *popt)
plt3d.plot_surface(X, Y, Z, alpha=0.8)

plt.show()
