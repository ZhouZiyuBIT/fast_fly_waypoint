import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os, sys
BASEPATH = os.path.abspath(__file__).split("fast_fly/", 1)[0]+"fast_fly/"
from trajectory import Trajectory
from gates.gates import Gates
from plotting import plot_gates_3d, plot_traj_3d

traj_topt1 = Trajectory(BASEPATH+"results/res_t_real_dyn1.csv")
traj_topt2 = Trajectory(BASEPATH+"results/res_t_real_dyn2.csv")
traj_topt3 = Trajectory(BASEPATH+"results/res_t_real_dyn3.csv")
traj_track = Trajectory(BASEPATH+"results/real_flight2023-04-23_18:46:29.csv")

# gates = Gates(BASEPATH+"gates/gates_real1.yaml")
gates = Gates()

# 1.0
gates.add_gate([ 6.70,  1.32, -1.43], 90)
gates.add_gate([ 5.14, -1.56, -1.51], 90)

gates.add_gate([ 2.35, -1.40, -1.19], 90)
gates.add_gate([ 2.44, -1.25, -1.75], 90)
gates.add_gate([ 2.46, -0.70, -1.70], 90)

gates.add_gate([-2.12,  1.20, -1.87], 90)
gates.add_gate([-2.14, -1.23, -1.83], 90)

# 0.9
# gates.add_gate([ 6.70,  1.32, -1.43], 90)
# gates.add_gate([ 5.14, -1.56, -1.51], 90)

# gates.add_gate([ 2.45, -1.43, -1.18], 90)
# gates.add_gate([ 2.45, -1.37, -1.80], 90)
# gates.add_gate([ 2.45, -0.72, -1.72], 90)

# gates.add_gate([-2.12,  1.20, -1.87], 90)
# gates.add_gate([-2.14, -1.23, -1.83], 90)

traj_track = traj_track[4650:5750]
ts = np.array([t for t in range(traj_track._N+1)])
ts = ts/100

# pos, vel, err
fig2 = plt.figure(figsize=(6,4.5))
ax_pos = fig2.add_subplot(311)
ax_pos.get_xaxis().set_visible(False)
ax_pos.set_ylabel("Position [m]")
ax_pos.set_ylim([-4, 11])
ax_pos.plot(ts, traj_track._pos[:,0], label="Px")
ax_pos.plot(ts, traj_track._pos[:,1], label="Py")
ax_pos.plot(ts, traj_track._pos[:,2], label="Pz")
ax_pos.legend(loc="upper left")
vline1 = ax_pos.axvline(341/100, color="k", linestyle="--")
vline1 = ax_pos.axvline(696/100, color="k", linestyle="--")
vline1 = ax_pos.axvline(1062/100, color="k", linestyle="--")
ax_pos.text(346/100, 9, "Loop 1", fontsize=10)
ax_pos.text(701/100, 9, "Loop 2", fontsize=10)
ax_pos.text(940/100, 9, "Loop 3", fontsize=10)

ax_vel = fig2.add_subplot(312)
ax_vel.get_xaxis().set_visible(False)
ax_vel.set_ylabel("Velocity [m/s]")
# ax_vel.set_ylim([-12, 12])
# ax_vel.plot(ts, traj_track._vel[:,0], label="Vx")
# ax_vel.plot(ts, traj_track._vel[:,1], label="Vy")
# ax_vel.plot(ts, traj_track._vel[:,2], label="Vz")
ax_vel.plot(ts, traj_track._vel[:,3])
# ax_vel.legend(loc="upper left")
ax_vel.axvline(341/100, color="k", linestyle="--")
ax_vel.axvline(696/100, color="k", linestyle="--")
ax_vel.axvline(1062/100, color="k", linestyle="--")

ax_thrust = fig2.add_subplot(313)
ax_thrust.set_xlabel("Time [s]", labelpad= 3)
ax_thrust.set_ylabel("Thrust [m/$s^2$]")
# ax_thrust.plot(ts, traj_track._u[:,1])
# ax_thrust.plot(ts, traj_track._u[:,2])
# ax_thrust.plot(ts, traj_track._u[:,3])
# ax_thrust.plot(ts, traj_track._acc[:,0])
# ax_thrust.plot(ts, traj_track._acc[:,1])
ax_thrust.plot(ts, traj_track._acc[:,2])
ax_thrust.axvline(341/100, color="k", linestyle="--")
ax_thrust.axvline(696/100, color="k", linestyle="--")
ax_thrust.axvline(1062/100, color="k", linestyle="--")


# ax_err.set_ylabel("Error [m]")
# track_errors = []
# for i in range(traj_track._N+1):
#     track_errors.append(traj_topt.distance(traj_track._pos[i]))
# ax_err.plot(ts, track_errors, color="gray")
# ax_err.axvline(365/100, color="k", linestyle="--")
# ax_err.axvline(755/100, color="k", linestyle="--")
# ax_err.axvline(1137/100, color="k", linestyle="--")

fig2.align_ylabels()
fig2.tight_layout(pad=0, rect=[0.02, 0.02, 0.95, 0.98])
plt.savefig(BASEPATH+"figs/track_real_dyn.eps")
plt.show()

#
fig = plt.figure(figsize=(6,5))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.set_xlabel("X [m]", labelpad= 8)
ax3d.set_ylabel("Y [m]", labelpad= 3)
ax3d.set_zlabel("Z [m]", labelpad= 3)
ax3d.set_xlim([-4, 9])
ax3d.set_ylim([-3, 3])
ax3d.set_zlim([-2, 0])
ax3d.set_zticks([-2, -1])
ax3d.view_init(elev=220, azim=25)
ax3d.set_aspect("equal")

# plot_gates_3d(ax3d, gates)

plot_traj_3d(ax3d, traj_topt1, color=None, linewidth=1, linestyle="--", alpha=1, label="Optimal: Loop1")
plot_traj_3d(ax3d, traj_topt2, color=None, linewidth=1, linestyle="--", alpha=1, label="Optimal: Loop2")
plot_traj_3d(ax3d, traj_topt3, color=None, linewidth=1, linestyle="--", alpha=1, label="Optimal: Loop3")

v_max = traj_track._vel[:,3].max()
v_min = traj_track._vel[:,3].min()
norm = plt.Normalize(vmin=v_min, vmax=v_max)
segments = np.stack((traj_track._pos[:-1], traj_track._pos[1:]), axis=1)

line_segments = Line3DCollection(segments, cmap="jet", norm=norm, linewidths=1, alpha=0.5, label="Tracked")
line_segments.set_array(traj_track._vel[:,3])
ax3d.add_collection(line_segments)

cb = fig.colorbar(line_segments, ax=ax3d, label="Velocity [m/s]", pad=0.1)
ax3d.legend(loc="upper right", bbox_to_anchor=(1.1, 1.), ncol=1, fontsize=None, frameon=True)

plot_gates_3d(ax3d, gates)
ax3d.text(gates._pos[0][0], gates._pos[0][1], gates._pos[0][2]-0.4, "C1", fontsize=8)
ax3d.text(gates._pos[1][0], gates._pos[1][1], gates._pos[1][2]-0.4, "C2", fontsize=8)
ax3d.text(gates._pos[2][0], gates._pos[2][1]-1.5, gates._pos[2][2]-0.4, "C3_1", fontsize=8)
ax3d.text(gates._pos[3][0], gates._pos[3][1]-0.6, gates._pos[3][2]-0.7, "C3_2", fontsize=8)
ax3d.text(gates._pos[4][0]+0.1, gates._pos[4][1], gates._pos[4][2]-0.5, "C3_3", fontsize=8)
ax3d.text(gates._pos[5][0], gates._pos[5][1], gates._pos[5][2]-0.4, "C4", fontsize=8)
ax3d.text(gates._pos[6][0], gates._pos[6][1], gates._pos[6][2]-0.4, "C5", fontsize=8)

fig.tight_layout(pad=0, rect=[0.0, 0.03, 0.95, 0.95])
plt.savefig(BASEPATH+"figs/track_real_dyn_3d.eps")

plt.show()
