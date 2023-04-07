#! /usr/bin/python3.8

import numpy as np
import time

# ROS
import rospy

from px4_bridge.msg import ThrustRates
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

# 
import os, sys
BASEPATH = os.path.abspath(__file__).split('script', 1)[0]+'script/fast_fly/'
sys.path += [BASEPATH]

from quadrotor import QuadrotorModel
from tracker import TrackerPos
from trajectory import Trajectory, StateSave
from gates.gates import Gates

from plotting import plot_gates_2d, plot_traj_xy

rospy.init_node("track_sim")
rospy.loginfo("ROS: Hello")

traj = Trajectory(BASEPATH+"results/res_t_n8.csv")
# traj = Trajectory()
quad =  QuadrotorModel(BASEPATH+'quad/quad_real.yaml')

tracker = TrackerPos(quad)

tracker.define_opt()
# tracker.load_so(BASEPATH+"generated/tracker_pos.so")
# tracker.load_so(BASEPATH+"generated/tracker_opt.so")

state_saver = StateSave(BASEPATH+"results/sim_flight.csv")

ctrl_pub = rospy.Publisher("track/thrust_rates", ThrustRates, tcp_nodelay=True, queue_size=1)
planned_path_pub = rospy.Publisher("planed_path", Path, queue_size=1)

def pub_path_visualization(traj:Trajectory):
    msg = Path()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    for i in range(traj._pos.shape[0]):
        pos = PoseStamped()
        pos.header.frame_id = "world"
        pos.pose.position.y = traj._pos[i, 0]
        pos.pose.position.x = traj._pos[i, 1]
        pos.pose.position.z = -traj._pos[i, 2]

        pos.pose.orientation.w = 1
        pos.pose.orientation.y = 0
        pos.pose.orientation.x = 0
        pos.pose.orientation.z = 0
        msg.poses.append(pos)
    planned_path_pub.publish(msg)

r_x = []
r_y = []
# last_t = time.time()
cnt = 0
time_factor = 1.0
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
        
        res = tracker.solve(x0, ploy.reshape(-1), trjdt, time_factor=time_factor)
        
        x = res['x'].full().flatten()
        Tt = 1*(x[tracker._Herizon*13+0]+x[tracker._Herizon*13+1]+x[tracker._Herizon*13+2]+x[tracker._Herizon*13+3])
        
        u = ThrustRates()
        u.thrust = Tt/4/quad._T_max
        u.wx = x[10]
        u.wy = x[11]
        u.wz = x[12]
        ctrl_pub.publish(u)

        # data save
        state_saver.log(time.time(), x0, [u.thrust, u.wx, u.wy, u.wz], [0,0,0])

rospy.Subscriber("q_sim/odom", Odometry, odom_cb, queue_size=1, tcp_nodelay=True)

def update_1hz(e):
    pub_path_visualization(traj)

rospy.Timer(rospy.Duration(1), update_1hz)

rospy.spin()
rospy.loginfo("ROS: Goodby")
import matplotlib.pyplot as plt
ax = plt.gca()
plt.plot(traj._pos[:,0], traj._pos[:,1])
plt.plot(r_x, r_y)
plt.show()
