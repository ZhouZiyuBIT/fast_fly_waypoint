import csv
import numpy as np
import yaml

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gates.gates import Gates
from trajectory import Trajectory

# sns.set()

# inear function
def plot_gates_3d(axes:plt.Axes, gates:Gates):
    for idx in range(gates._N):
        axes.plot(gates._shapes[idx][0], gates._shapes[idx][1], gates._shapes[idx][2], linewidth=3, color="dimgray")
        axes.plot(gates._pos[idx][0], gates._pos[idx][1], gates._pos[idx][2], marker="*", markersize=3, color="red")

def plot_gates_2d(axes:plt.Axes, gates:Gates):
    for idx in range(gates._N):
        axes.plot(gates._shapes[idx][0], gates._shapes[idx][1], linewidth=3, color="dimgray")

def plot_traj_xy(axes:plt.Axes, traj:Trajectory, linewidth=1, linestyle="-", label="", color=None, alpha=1):
    axes.plot(traj._pos[:,0], traj._pos[:,1], linewidth=linewidth, linestyle=linestyle, color=color, alpha=alpha, label=label)

def plot_traj_3d(axes3d, traj:Trajectory, linewidth=1, linestyle="-", label="", color=None, alpha=1):
    axes3d.plot(traj._pos[:,0], traj._pos[:,1], traj._pos[:,2], linewidth=linewidth, linestyle=linestyle, color=color, alpha=alpha, label=label)

# plot
def plot_tracked(gates:Gates, traj_planned:Trajectory, traj_tracked:Trajectory):
    fig = plt.figure(figsize=(6,5.5))

    ax = fig.add_subplot(111)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("X [m]", labelpad=0)
    ax.set_ylabel("Y [m]", labelpad=-5)

    ax.set_aspect("equal")

    plot_gates_2d(ax, gates)
    
    ax.scatter(traj_tracked._pos[0,0], traj_tracked._pos[0,1], marker="*")
    plot_traj_xy(ax, traj_tracked, linewidth=2, linestyle="-", color="#2878B5", label="Traj: Tracked")
    # plot_traj_xy(ax, traj_planned, linewidth=3, linestyle="--", color="#F8AC8C", label="Traj: Planned")

    # # inset axes....
    # axins = ax.inset_axes([0.56, 0.22, 0.28, 0.28])
    # plot_traj_xy(axins, traj_tracked, linewidth=2, linestyle="-", color="#2878B5", label="Traj: Tracked")
    # plot_traj_xy(axins, traj_planned, linewidth=3, linestyle="--", color="#F8AC8C", label="Traj: Planned")
    # plot_gates_2d(axins, gates)
    # # subregion of the original image
    # x1, x2, y1, y2 = 7.7, 8.8, 5.2, 6.3
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])

    # ax.indicate_inset_zoom(axins, edgecolor="gray")
    # ax.legend()
    # fig.tight_layout(pad=0, rect=[0.01, 0.0, 0.99, 1])
    # plt.savefig("track.eps")

def plot_track_cmp(gates:Gates, traj_planned:Trajectory, traj_track:Trajectory, traj_track_mpc:Trajectory):
    fig = plt.figure(figsize=(6,4.8))
    
    ax = fig.add_subplot(111)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("X [m]", labelpad=0)
    ax.set_ylabel("Y [m]", labelpad=-5)
    ax.grid(True)

    ax.set_aspect("equal")

    plot_gates_2d(ax, gates)
    
    ax.scatter(traj_track._pos[0,0], traj_track._pos[0,1], marker="*")
    plot_traj_xy(ax, traj_track_mpc, linewidth=1, linestyle="-", label="Traj: Track MPC")
    plot_traj_xy(ax, traj_track, linewidth=1, linestyle="-", label="Traj: Track tMPC")
    plot_traj_xy(ax, traj_planned, linewidth=1, linestyle="--", label="Traj: Planned")
    
    # inset axes....
    axins = ax.inset_axes([0.56, 0.22, 0.28, 0.28])
    plot_traj_xy(axins, traj_track_mpc, linewidth=1, linestyle="-", label="Traj: Track MPC")
    plot_traj_xy(axins, traj_track, linewidth=1, linestyle="-", label="Traj: Tracked")
    plot_traj_xy(axins, traj_planned, linewidth=2, linestyle="--", label="Traj: Planned")
    plot_gates_2d(axins, gates)
    # subregion of the original image
    x1, x2, y1, y2 = 7.7, 9.0, 5.2, 6.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="gray")
    ax.legend()
    fig.tight_layout(pad=0, rect=[0.15, 0.0, 0.85, 1])
    

def plot_track_vel(gates:Gates, traj_planned:Trajectory, traj_tracked:Trajectory, traj_track_mpc:Trajectory, first_gate_pos):

    fig = plt.figure(figsize=(6, 4.5))

    ax1 = fig.add_subplot(221)
    ax1.set_xlim([-10, 10])
    ax1.set_ylim([-10, 10])
    ax1.set_xlabel("X [m]", labelpad=0)
    ax1.set_ylabel("Y [m]", labelpad=-5)
    ax1.set_title("Loop 1")
    ax1.set_aspect("equal")
    plot_gates_2d(ax1, gates)
    plot_traj_xy(ax1, traj_planned, linestyle='--', linewidth=2, color="gray", label="Traj: Planned")

    ax2 = fig.add_subplot(222)
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-10, 10])
    ax2.set_xlabel("X [m]", labelpad=0)
    ax2.set_ylabel("Y [m]", labelpad=-5)
    ax2.set_aspect("equal")
    ax2.set_title("Loop 2")
    plot_gates_2d(ax2, gates)
    plot_traj_xy(ax2, traj_planned, linestyle='--', linewidth=2, color="gray", label=None)
    
    loops_idx = traj_tracked.divide_loops(first_gate_pos)
    print(len(loops_idx))
    vel_min = np.min(traj_tracked._vel[:-1,3])
    vel_max = np.max(traj_tracked._vel[:-1,3])
    norm = plt.Normalize(vel_min, vel_max)

    traj1_seg = np.stack((traj_tracked._pos[0:loops_idx[1]-1, :2], traj_tracked._pos[1:loops_idx[1], :2]), axis=1)
    traj2_seg = np.stack((traj_tracked._pos[loops_idx[1]:loops_idx[2]-1, :2], traj_tracked._pos[loops_idx[1]+1:loops_idx[2], :2]), axis=1)

    traj1_collection = LineCollection(traj1_seg, cmap="jet", linewidth=1, norm=norm, linestyles='-', label="Traj: Tracked")
    traj1_collection.set_array(traj_tracked._vel[0:loops_idx[1],3])
    traj2_collection = LineCollection(traj2_seg, cmap="jet", linewidth=1, norm=norm, linestyles='-')
    traj2_collection.set_array(traj_tracked._vel[loops_idx[1]:loops_idx[2],3])

    ax1.scatter(traj_tracked._pos[0,0], traj_tracked._pos[0,1], marker="*")
    line1 = ax1.add_collection(traj1_collection)
    line2 = ax2.add_collection(traj2_collection)
    # plot_traj_xy(ax1, traj_planned, linestyle=':', linewidth=2, color="black", label="Traj: Planned")
    # plot_traj_xy(ax2, traj_planned, linestyle='--', linewidth=2, color="gray", label="Traj: Planned")

    # divider1 = make_axes_locatable(ax1)
    # cax1 = divider1.append_axes("right", size="5%", pad="5%")
    # fig.colorbar(line1, cax=cax1, label='Velocity [m/s]')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad="10%")
    fig.colorbar(line2, cax=cax2, label='Velocity: tMPC [m/s]', ticks=[2.5, 5.0, 7.5, 10.0, 12.5])
    fig.legend(ncol=2, loc="upper center")
    
    ax3 = fig.add_subplot(223)
    ax3.set_xlim([-10, 10])
    ax3.set_ylim([-10, 10])
    ax3.set_xlabel("X [m]", labelpad=0)
    ax3.set_ylabel("Y [m]", labelpad=-5)
    # ax3.set_title("Loop 1")
    ax3.set_aspect("equal")
    plot_gates_2d(ax3, gates)
    plot_traj_xy(ax3, traj_planned, linestyle='--', linewidth=2, color="gray", label="Traj: Planned")

    ax4 = fig.add_subplot(224)
    ax4.set_xlim([-10, 10])
    ax4.set_ylim([-10, 10])
    ax4.set_xlabel("X [m]", labelpad=0)
    ax4.set_ylabel("Y [m]", labelpad=-5)
    ax4.set_aspect("equal")
    # ax4.set_title("Loop 2")
    plot_gates_2d(ax4, gates)
    plot_traj_xy(ax4, traj_planned, linestyle='--', linewidth=2, color="gray", label=None)

    loops_idx = traj_track_mpc.divide_loops(first_gate_pos)
    print(len(loops_idx))
    vel_min = np.min(traj_track_mpc._vel[:-1,3])
    vel_max = np.max(traj_track_mpc._vel[:-1,3])
    norm = plt.Normalize(vel_min, vel_max)

    traj1_seg = np.stack((traj_track_mpc._pos[0:loops_idx[1]-1, :2], traj_track_mpc._pos[1:loops_idx[1], :2]), axis=1)
    traj2_seg = np.stack((traj_track_mpc._pos[loops_idx[1]:loops_idx[2]-1, :2], traj_track_mpc._pos[loops_idx[1]+1:loops_idx[2], :2]), axis=1)

    traj1_collection = LineCollection(traj1_seg, cmap="jet", linewidth=1, norm=norm, linestyles='-', label="Traj: Track MPC")
    traj1_collection.set_array(traj_track_mpc._vel[0:loops_idx[1],3])
    traj2_collection = LineCollection(traj2_seg, cmap="jet", linewidth=1, norm=norm, linestyles='-')
    traj2_collection.set_array(traj_track_mpc._vel[loops_idx[1]:loops_idx[2],3])

    ax3.scatter(traj_track_mpc._pos[0,0], traj_track_mpc._pos[0,1], marker="*")
    line1 = ax3.add_collection(traj1_collection)
    line2 = ax4.add_collection(traj2_collection)
    # plot_traj_xy(ax1, traj_planned, linestyle=':', linewidth=2, color="black", label="Traj: Planned")
    # plot_traj_xy(ax2, traj_planned, linestyle='--', linewidth=2, color="gray", label="Traj: Planned")

    # divider1 = make_axes_locatable(ax1)
    # cax1 = divider1.append_axes("right", size="5%", pad="5%")
    # fig.colorbar(line1, cax=cax1, label='Velocity [m/s]')
    divider2 = make_axes_locatable(ax4)
    cax2 = divider2.append_axes("right", size="5%", pad="10%")
    fig.colorbar(line2, cax=cax2, label='Velocity: MPC [m/s]', ticks=[2.5, 5.0, 7.5, 10.0, 12.5])

    
    fig.tight_layout(pad=0, rect=[0.1, 0.03, 0.9, 0.9])
    plt.savefig("track_loop.eps")
    # ax1.margins(x=0,y=0)
    # ax1.margins(x=0,y=0)
    # fig.tight_layout()
    # ax1.legend()
    # ax2.legend()


def plot_3d(gates:Gates):
    fig = plt.figure("3d", figsize=(6,3))
    ax_3d = fig.add_subplot([0,0,1,1], projection="3d")
    ax_3d.set_xlim((-10.5,10.5))
    ax_3d.set_ylim((-10.5,10.5))
    ax_3d.set_zlim((-4.5,0.5))
    # ax_3d.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax_3d.set_zticks([-4.0, -2.0, 0.0])
    ax_3d.tick_params(pad=0, labelrotation=0)
    ax_3d.set_xlabel("X [m]", labelpad=8, rotation=0)
    ax_3d.set_ylabel("Y [m]", labelpad=8, rotation=0)
    ax_3d.set_zlabel("Z [m]", labelpad=-2, rotation=0)
    ax_3d.view_init(elev=200, azim=-15)
    ax_3d.set_aspect("equal")
    
    plot_gates_3d(ax_3d, gates)

def plot_planned(gates:Gates, traj_planned:Trajectory):
    fig = plt.figure(figsize=(6,5.5))

    ax = fig.add_subplot(111)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("X [m]", labelpad=0)
    ax.set_ylabel("Y [m]", labelpad=-5)
    ax.set_aspect("equal")
    
    plot_gates_2d(ax, gates)
    plot_traj_xy(ax, traj_planned)

if __name__ == "__main__":

    import os, sys
    BASEPATH = os.path.abspath(__file__).split("fast_fly/", 1)[0]+"fast_fly/"

    traj = Trajectory(BASEPATH+"results/res_n6.csv")
    traj_t = Trajectory(BASEPATH+"results/res_t_n6.csv")
    traj_track = Trajectory(BASEPATH+"results/real_flight1.csv")
    # rpg_n6 = Trajectory("./rpg_results/result_n8.csv")
    gates = Gates(BASEPATH+"gates/gates_n6.yaml")

    # plot_track_vel(gates, traj_t, traj_track, gates._pos[0])
    # plot_tracked(gates, traj_t, traj_track)
    # plot_3d(gates)
    
    plot_planned(gates, traj_t)

    # plt.tight_layout(pad=0)
    plt.show()
