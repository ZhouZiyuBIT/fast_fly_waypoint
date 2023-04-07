import matplotlib.pyplot as plt

import os, sys
BASEPATH = os.path.abspath(__file__).split("fast_fly/", 1)[0]+"fast_fly/"
from trajectory import Trajectory
from gates.gates import Gates
from plotting import plot_track_cmp

traj_topt = Trajectory(BASEPATH+"results/res_t_n8.csv")
traj_track = Trajectory(BASEPATH+"results/sim_flight.csv")
traj_track_mpc = Trajectory(BASEPATH+"results/sim_flight_mpc.csv")

gates = Gates(BASEPATH+"gates/gates_n8.yaml")

plot_track_cmp(gates, traj_topt, traj_track, traj_track_mpc)

plt.savefig(BASEPATH+"figs/track_cmp.eps")

plt.show()
