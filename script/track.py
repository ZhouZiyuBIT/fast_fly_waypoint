#! /usr/bin/python3.8

import numpy as np
import casadi as ca
import time

# ROS
import rospy

from fast_fly_waypoint.msg import TrackTraj
from px4_bridge.msg import ThrustRates
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu

# 
import os, sys
BASEPATH = os.path.abspath(__file__).split('script', 1)[0]+'script/fast_fly/'
sys.path += [BASEPATH]

from quadrotor import QuadrotorModel
from tracker import TrackerOpt, TrackerOpt2, TrackerPos, TrackerPosVel2, TrackerMPCC, TrackerP, TrackerMPC
from trajectory import Trajectory, StateSave
from gates.gates import Gates

from plotting import plot_gates_2d, plot_traj_xy

rospy.init_node("track")
rospy.loginfo("ROS: Hello")

# traj = Trajectory(BASEPATH+"results/res_t_n6.csv")
traj = Trajectory()
quad =  QuadrotorModel(BASEPATH+'quad/quad_px4.yaml')

tracker = TrackerPos(quad)
# tracker = TrackerOpt(quad)
# tracker = TrackerMPCC(quad)
# tracker = TrackerP(quad)
# tracker = TrackerMPC(quad)

# tracker.define_opt()
tracker.load_so(BASEPATH+"generated/tracker_pos.so")
# tracker.load_so(BASEPATH+"generated/tracker_opt.so")

state_saver = StateSave(BASEPATH+"results/real_flight1.csv")

imu_data = Imu()

ctrl_pub = rospy.Publisher("~thrust_rates", ThrustRates, tcp_nodelay=True, queue_size=1)

stop_flag = False
def stop_cb(msg: Bool):
    global stop_flag
    stop_flag = msg.data

r_x = []
r_y = []
# last_t = time.time()
cnt = 0
time_factor = 0.5
def odom_cb(msg: Odometry):
    # global last_t 
    # rospy.loginfo(time.time()-last_t)
    # last_t = time.time()
    global cnt, time_factor
    cnt += 1
    if traj._N != 0:
        print("track:", cnt)
        x0 = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                    msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                    msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])

        r_x.append(msg.pose.pose.position.x)
        r_y.append(msg.pose.pose.position.y)
        trjp, trjv, trjdt, ploy = traj.sample(tracker._trj_N, x0[:3])
        # trjp, trjv = traj.sample_dt(2, x0[:3], dt=0.1)
        # print(trjdt)
        if stop_flag:
            time_factor = time_factor*0.998
            # res = tracker.solve(x0, ploy.reshape(-1), trjdt, time_factor=1)
            # res = tracker.solve(x0, trjp.reshape(-1), 20)
            # res = tracker.solve(x0, trjp.reshape(-1), trjv.reshape(-1), trjdt, np.array([-10000,100,-1]), 20)
            # res = tracker.solve(x0, trjp.reshape(-1), trjv.reshape(-1))
            # res = tracker.solve(x0, trjp.reshape(-1), trjv.reshape(-1))

        res = tracker.solve(x0, ploy.reshape(-1), trjdt, time_factor=time_factor)
        
        x = res['x'].full().flatten()
        Tt = 1*(x[tracker._Herizon*13+0]+x[tracker._Herizon*13+1]+x[tracker._Herizon*13+2]+x[tracker._Herizon*13+3])
        
        # thrust to z_accel
        # qw, qx, qy, qz = x0[6], x0[7], x0[8], x0[9]
        # vx, vy, vz = x0[3], x0[4], x0[5]
        # c_n = ( Tt/quad._m + quad._D[2,2]*(2*(qw*qy+qx*qz)*vx+2*(qy*qz-qw*qx)*vy+(qw*qw-qx*qx-qy*qy+qz*qz)*vz) )
        # print(c_n)
        u = ThrustRates()
        u.thrust = Tt/4/quad._T_max
        u.wx = x[10]
        u.wy = x[11]
        u.wz = x[12]
        ctrl_pub.publish(u)

        # data save
        state_saver.log(time.time(), x0, [u.thrust, u.wx, u.wy, u.wz], [imu_data.linear_acceleration.x, imu_data.linear_acceleration.y, imu_data.linear_acceleration.z])

def imu_cb(msg:Imu):
    global imu_data
    imu_data = msg

def track_traj_cb(msg: TrackTraj):
    pos = []
    vel = []
    quat = []
    angular = []
    dt = []
    for i in range(len(msg.dt)):
        pos.append([msg.position[i].x, msg.position[i].y, msg.position[i].z])
        vel.append([msg.velocity[i].x, msg.velocity[i].y, msg.velocity[i].z])
        quat.append([msg.orientation[i].w, msg.orientation[i].x, msg.orientation[i].y, msg.orientation[i].z])
        angular.append([msg.angular[i].x, msg.angular[i].y, msg.angular[i].z])
        dt.append(msg.dt[i])
    pos.append([msg.position[-1].x, msg.position[-1].y, msg.position[-1].z])
    vel.append([msg.velocity[-1].x, msg.velocity[-1].y, msg.velocity[-1].z])
    quat.append([msg.orientation[-1].w, msg.orientation[-1].x, msg.orientation[-1].y, msg.orientation[-1].z])
    angular.append([msg.angular[-1].x, msg.angular[-1].y, msg.angular[-1].z])

    traj.load_data(np.array(pos), np.array(vel), np.array(quat), np.array(angular), np.array(dt))

rospy.Subscriber("~odom", Odometry, odom_cb, queue_size=1, tcp_nodelay=True)
rospy.Subscriber("~imu", Imu, imu_cb, queue_size=1, tcp_nodelay=True)
rospy.Subscriber("~track_traj", TrackTraj, track_traj_cb, queue_size=1, tcp_nodelay=True)
rospy.Subscriber("stop_flag", Bool, stop_cb)

rospy.spin()
rospy.loginfo("ROS: Goodby")
import matplotlib.pyplot as plt
ax = plt.gca()
plt.plot(traj._pos[:,0], traj._pos[:,1])
plt.plot(r_x, r_y)
plt.show()
