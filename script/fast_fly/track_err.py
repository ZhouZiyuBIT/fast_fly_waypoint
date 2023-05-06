import numpy as np
import matplotlib.pyplot as plt

import os, sys
BASEPATH = os.path.abspath(__file__).split("fast_fly/", 1)[0]+"fast_fly/"
from trajectory import Trajectory
from gates.gates import Gates

traj_topt = Trajectory(BASEPATH+"results/res_t_n8.csv")
traj_track = Trajectory(BASEPATH+"results/sim_flight.csv")
traj_track_mpc = Trajectory(BASEPATH+"results/sim_flight_mpc.csv")
gates = Gates(BASEPATH+"gates/gates_n8.yaml")

loop_idx = traj_track.divide_loops(np.array(gates._pos[0]))
d_err = []
for i in range(loop_idx[2], loop_idx[3]):
    d_err.append(traj_topt.distance(traj_track._pos[i]))
d_err = np.array(d_err)

loop_idx_mpc = traj_track_mpc.divide_loops(np.array(gates._pos[0]))
d_err_mpc = []
for i in range(loop_idx_mpc[2], loop_idx_mpc[3]):
    d_err_mpc.append(traj_topt.distance(traj_track_mpc._pos[i]))
d_err_mpc = np.array(d_err_mpc)

# plt.figure()
# plt.plot(traj_topt._pos[:,0], traj_topt._pos[:,1], 'r')
# plt.plot(traj_track._pos[:,0], traj_track._pos[:,1], 'g')
# plt.plot(traj_track._vel[:,3], 'r')
# plt.plot(d_err)
# plt.plot(d_err_mpc)
# plt.show()

print("Mean error: ", np.mean(d_err))
print("Max error: ", np.max(d_err))
print("Time: ", traj_track._t[loop_idx[3]]-traj_track._t[loop_idx[2]])

print("Mean error mpc: ", np.mean(d_err_mpc))
print("Max error mpc: ", np.max(d_err_mpc))
print("Time mpc: ", traj_track_mpc._t[loop_idx_mpc[3]]-traj_track_mpc._t[loop_idx_mpc[2]])
