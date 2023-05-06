import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np

import os, sys
BASEPATH = os.path.abspath(__file__).split("fast_fly/", 1)[0]+"fast_fly/"
from trajectory import Trajectory
from gates.gates import Gates
from plotting import plot_gates_3d, plot_traj_3d

traj_topt = Trajectory(BASEPATH+"results/res_t_real1.csv")
traj_track = Trajectory(BASEPATH+"results/real_flight2023-04-23_15:42-1.0.csv")

# gates = Gates()
# gates.add_gate([ 4.38, -1.52, -1.86], 90)
# # gates.add_gate([ 0.05, -0.02, -0.76], 90)
# gates.add_gate([-2.60,  1.38, -1.75], 90)
# gates.add_gate([-1.93, -1.44, -1.62], 90)
# gates.add_gate([ 0.62,  0.46, -1.72], 90)
# gates.add_gate([ 4.61,  1.36, -1.22], 90)

traj_track = traj_track[5985+140:7262]
traj_track._pos[1027-140] = (traj_track._pos[1026-140]+traj_track._pos[1028-140])/2

ts = np.array([t for t in range(traj_track._N+1)])
ts = ts/100

# vel, err
fig = plt.figure(figsize=(6,5))
ax_pos = fig.add_subplot(311)

# ax_pos.get_xaxis().set_visible(False)
# ax_pos.set_xlim([0, 12])
ax_pos.set_ylabel("Position Y [m]")
ax_pos.set_xlabel("Position X [m]")
# ax_pos.set_ylim([-4, 11])
ax_pos.plot(traj_topt._pos[:,0], traj_topt._pos[:,1], 'r', label="Optimal")
plot_track, = ax_pos.plot(traj_track._pos[:0,0], traj_track._pos[:0,1], label="Tracked")
# plot_y, = ax_pos.plot(ts[:0], traj_track._pos[:0,1], label="Py")
# plot_z, = ax_pos.plot(ts[:0], traj_track._pos[:0,2], label="Pz")
ax_pos.legend(loc="upper left")

ax_vel = fig.add_subplot(312)
# ax_vel.get_xaxis().set_visible(False)
ax_vel.set_xlabel("Time [s]", labelpad= 3)
ax_vel.set_xlim([0,12])
ax_vel.set_ylabel("Velocity [m/s]")
ax_vel.set_ylim([1.5, 12])
# ax_vel.plot(ts, traj_track._vel[:,0], label="Vx")
# ax_vel.plot(ts, traj_track._vel[:,1], label="Vy")
# ax_vel.plot(ts, traj_track._vel[:,2], label="Vz")
plot_v, = ax_vel.plot(ts[:0], traj_track._vel[:0,3])
# ax_vel.plot(traj_track._vel[:,3])
# ax_vel.legend(loc="upper left")
ax_vel.axvline(365/100, color="k", linestyle="--")
ax_vel.axvline(755/100, color="k", linestyle="--")
ax_vel.axvline(1137/100, color="k", linestyle="--")

ax_err = fig.add_subplot(313)
ax_err.set_xlim([0,12])
ax_err.set_xlabel("Time [s]", labelpad= 3)
ax_err.set_ylim([0,0.25])
ax_err.set_ylabel("Error [m]")
track_errors = []
for i in range(traj_track._N+1):
    track_errors.append(traj_topt.distance(traj_track._pos[i]))
plot_err, = ax_err.plot(ts[:0], track_errors[:0], color="gray")
ax_err.axvline(365/100, color="k", linestyle="--")
ax_err.axvline(755/100, color="k", linestyle="--")
ax_err.axvline(1137/100, color="k", linestyle="--")

fig.align_ylabels()
fig.tight_layout(pad=0, rect=[0.02, 0.02, 0.95, 0.98])


cnt = 1
def update_plot(*arg):
    global cnt
    
    plot_track.set_data(traj_track._pos[:cnt,0], traj_track._pos[:cnt,1])

    
    plot_v.set_data(ts[:cnt], traj_track._vel[:cnt,3])
    
    plot_err.set_data(ts[:cnt], track_errors[:cnt])
    
    cnt += 1
    return [plot_track, plot_v, plot_err]

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, update_plot, interval=10, save_count=len(ts), blit=True, repeat=False)
# ani.save(BASEPATH+'figs/animation.gif', writer='imagemagick')
ani.save(BASEPATH+'figs/animation1.mp4', writer='ffmpeg', fps=50)

# plt.savefig(BASEPATH+"figs/tttt.eps")
plt.show()
