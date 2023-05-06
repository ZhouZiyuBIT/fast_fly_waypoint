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
traj_track = Trajectory(BASEPATH+"results/real_flight2023-04-23_18:33:55.csv")

# gates = Gates()
# gates.add_gate([ 4.38, -1.52, -1.86], 90)
# # gates.add_gate([ 0.05, -0.02, -0.76], 90)
# gates.add_gate([-2.60,  1.38, -1.75], 90)
# gates.add_gate([-1.93, -1.44, -1.62], 90)
# gates.add_gate([ 0.62,  0.46, -1.72], 90)
# gates.add_gate([ 4.61,  1.36, -1.22], 90)

traj_track = traj_track[4345:7500]
# traj_track._pos[1027-140] = (traj_track._pos[1026-140]+traj_track._pos[1028-140])/2

ts = np.array([t for t in range(traj_track._N+1)])
ts = ts/100

# vel, err
fig = plt.figure(figsize=(6,3.5))
# ax_pos = fig.add_subplot(311)

# # ax_pos.get_xaxis().set_visible(False)
# # ax_pos.set_xlim([0, 12])
# ax_pos.set_ylabel("Position Y [m]")
# ax_pos.set_xlabel("Position X [m]")
# # ax_pos.set_ylim([-4, 11])
# ax_pos.plot(traj_topt._pos[:,0], traj_topt._pos[:,1], 'r', label="Optimal")
# plot_track, = ax_pos.plot(traj_track._pos[:0,0], traj_track._pos[:0,1], label="Tracked")
# # plot_y, = ax_pos.plot(ts[:0], traj_track._pos[:0,1], label="Py")
# # plot_z, = ax_pos.plot(ts[:0], traj_track._pos[:0,2], label="Pz")
# ax_pos.legend(loc="upper left")

ax_vel = fig.add_subplot(211)
ax_vel.get_xaxis().set_visible(False)
ax_vel.set_xlim([0,32])
ax_vel.set_ylabel("Velocity [m/s]")
ax_vel.set_ylim([-11, 11])
plot_vx, = ax_vel.plot(ts[:0], traj_track._vel[:0,0], label="Vx")
plot_vy, = ax_vel.plot(ts[:0], traj_track._vel[:0,1], label="Vy")
plot_vz, = ax_vel.plot(ts[:0], traj_track._vel[:0,2], label="Vz")
# plot_v, = ax_vel.plot(ts[:], traj_track._vel[:,3])
ax_vel.legend(loc="upper right")

ax_thrust = fig.add_subplot(212)
ax_thrust.set_xlabel("Time [s]", labelpad= 3)
ax_thrust.set_xlim([0,32])
ax_thrust.set_ylabel("Thrust [m/$s^2$]")
ax_thrust.set_ylim([-26, 2])
plot_thrust, = ax_thrust.plot(ts[:0], traj_track._acc[:0,2])
ax_thrust.plot([0,35], [-21, -21], color="k", linestyle="--", label="Max. thrust")
ax_thrust.legend(loc="upper right")

fig.align_ylabels()
fig.tight_layout(pad=0, rect=[0.02, 0.02, 0.95, 0.98])


cnt = 1
def update_plot(*arg):
    global cnt
    
    plot_vx.set_data(ts[:cnt], traj_track._vel[:cnt,0])
    plot_vy.set_data(ts[:cnt], traj_track._vel[:cnt,1])
    plot_vz.set_data(ts[:cnt], traj_track._vel[:cnt,2])

    plot_thrust.set_data(ts[:cnt], traj_track._acc[:cnt,2])


    cnt += 3
    return [plot_vx, plot_vy, plot_vz, plot_thrust]

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, update_plot, interval=10, save_count=int(len(ts)/3)-1, blit=True, repeat=False)
# # ani.save(BASEPATH+'figs/animation.gif', writer='imagemagick')
ani.save(BASEPATH+'figs/animation2.mp4', writer='ffmpeg', fps=50)

plt.show()
