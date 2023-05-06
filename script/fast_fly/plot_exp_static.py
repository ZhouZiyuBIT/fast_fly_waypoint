import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os, sys
BASEPATH = os.path.abspath(__file__).split("fast_fly/", 1)[0]+"fast_fly/"
from trajectory import Trajectory
from gates.gates import Gates
from plotting import plot_gates_3d, plot_traj_3d

traj_topt = Trajectory(BASEPATH+"results/res_t_real1.csv")
traj_track = Trajectory(BASEPATH+"results/real_flight2023-04-23_15:42-1.0.csv")

# gates = Gates(BASEPATH+"gates/gates_real1.yaml")
gates = Gates()
gates.add_gate([ 4.38, -1.52, -1.86], 90)
# gates.add_gate([ 0.05, -0.02, -0.76], 90)
gates.add_gate([-2.60,  1.38, -1.75], 90)
gates.add_gate([-1.93, -1.44, -1.62], 90)
gates.add_gate([ 0.62,  0.46, -1.72], 90)
gates.add_gate([ 4.61,  1.36, -1.22], 90)

traj_track = traj_track[5985+140:7262]
traj_track._pos[1027-140] = (traj_track._pos[1026-140]+traj_track._pos[1028-140])/2

ts = np.array([t for t in range(traj_track._N+1)])
ts = ts/100

# vel, err
fig2 = plt.figure(figsize=(6,4.5))
ax_pos = fig2.add_subplot(311)
ax_pos.get_xaxis().set_visible(False)
ax_pos.set_ylabel("Position [m]")
ax_pos.set_ylim([-4, 11])
ax_pos.plot(ts, traj_track._pos[:,0], label="Px")
ax_pos.plot(ts, traj_track._pos[:,1], label="Py")
ax_pos.plot(ts, traj_track._pos[:,2], label="Pz")
ax_pos.legend(loc="upper left")
vline1 = ax_pos.axvline(365/100, color="k", linestyle="--")
vline1 = ax_pos.axvline(755/100, color="k", linestyle="--")
vline1 = ax_pos.axvline(1137/100, color="k", linestyle="--")
ax_pos.text(370/100, 9, "Loop 1", fontsize=10)
ax_pos.text(760/100, 9, "Loop 2", fontsize=10)
ax_pos.text(1010/100, 9, "Loop 3", fontsize=10)

ax_vel = fig2.add_subplot(312)
ax_vel.get_xaxis().set_visible(False)
ax_vel.set_ylabel("Velocity [m/s]")
ax_vel.set_ylim([-12, 12])
ax_vel.plot(ts, traj_track._vel[:,0], label="Vx")
ax_vel.plot(ts, traj_track._vel[:,1], label="Vy")
ax_vel.plot(ts, traj_track._vel[:,2], label="Vz")
# ax_vel.plot(traj_track._vel[:,3])
ax_vel.legend(loc="upper left")
ax_vel.axvline(365/100, color="k", linestyle="--")
ax_vel.axvline(755/100, color="k", linestyle="--")
ax_vel.axvline(1137/100, color="k", linestyle="--")

ax_err = fig2.add_subplot(313)
ax_err.set_xlabel("Time [s]", labelpad= 3)
ax_err.set_ylabel("Error [m]")
track_errors = []
for i in range(traj_track._N+1):
    track_errors.append(traj_topt.distance(traj_track._pos[i]))
ax_err.plot(ts, track_errors, color="gray")
ax_err.axvline(365/100, color="k", linestyle="--")
ax_err.axvline(755/100, color="k", linestyle="--")
ax_err.axvline(1137/100, color="k", linestyle="--")

fig2.align_ylabels()
fig2.tight_layout(pad=0, rect=[0.02, 0.02, 0.95, 0.98])
plt.savefig(BASEPATH+"figs/track_real_static.eps")
plt.show()

#
# fig = plt.figure(figsize=(10,10))
# ax3d = fig.add_subplot(111, projection="3d")
# ax3d.set_xlabel("X [m]", labelpad= 8)
# ax3d.set_ylabel("Y [m]", labelpad= 10)
# ax3d.set_zlabel("Z [m]", labelpad= 3)
# ax3d.set_xlim([-4, 9])
# ax3d.set_ylim([-3, 3])
# ax3d.set_zlim([-2, 0])
# ax3d.set_zticks([-2, -1])
# ax3d.view_init(elev=195, azim=-7)
# ax3d.set_aspect("equal")

# plot_gates_3d(ax3d, gates)

# plot_traj_3d(ax3d, traj_track, linewidth=1, linestyle="-", label="Traj: Tracked")
# plot_traj_3d(ax3d, traj_topt, linewidth=1, linestyle="--", label="Traj: Planned")



# ax3d.legend(loc="upper right", bbox_to_anchor=(0.79, 0.78), ncol=1, fontsize=None, frameon=True)
# fig.tight_layout(pad=0, rect=[0.0, 0.0, 1, 1])
# plt.savefig(BASEPATH+"figs/track_real_static.eps")

# plt.show()
