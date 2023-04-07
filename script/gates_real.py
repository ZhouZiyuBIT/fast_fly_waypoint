#! /usr/bin/python3.8

import rospy
import numpy as np

from fast_fly_waypoint.msg import TrackTraj
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

import os, sys
BASEPATH = os.path.abspath(__file__).split('script/', 1)[0]+'script/fast_fly/'
sys.path += [BASEPATH]

from gates.gates import Gates

import time

rospy.init_node("gates_sim")
rospy.loginfo("ROS: Hello")
gates_pub = rospy.Publisher("~gates", TrackTraj, tcp_nodelay=True, queue_size=1)
gates_marker_pub = rospy.Publisher("/plan/gates_marker", Marker, queue_size=1)

gates = Gates(BASEPATH+"gates/gates_real.yaml")
dyn_gate_id = 5

def timer_cb(event):
    gates_traj = TrackTraj()
    gates_marker = Marker()
    for i in range(gates._N):
        pos = Point()
        pos.x = gates._pos[i][0]
        pos.y = gates._pos[i][1]
        pos.z = gates._pos[i][2]
        gates_traj.position.append(pos)

        pos = Point()
        pos.y = gates._pos[i][0]
        pos.x = gates._pos[i][1]
        pos.z = -gates._pos[i][2]
        gates_marker.header.frame_id = "world"
        gates_marker.action=Marker.ADD
        gates_marker.type = Marker.SPHERE_LIST
        gates_marker.pose.position.x = 0
        gates_marker.pose.position.y = 0
        gates_marker.pose.position.z = 0
        gates_marker.pose.orientation.w = 1
        gates_marker.pose.orientation.x = 0
        gates_marker.pose.orientation.y = 0
        gates_marker.pose.orientation.z = 0
        gates_marker.scale = Vector3(0.25,0.25,0.25)
        gates_marker.points.append(pos)
        gates_marker.colors.append(ColorRGBA(1,0,0,1))
    gates_pub.publish(gates_traj)
    gates_marker_pub.publish(gates_marker)

def dyn_gate_cb(msg: Point):
    p_now = np.array([msg.x, msg.y, msg.z])
    p = np.array(gates._pos[dyn_gate_id-1])
    if np.linalg.norm(p_now - p) > 0.1:
        gates._pos[dyn_gate_id-1][0] = p_now[0]
        gates._pos[dyn_gate_id-1][1] = p_now[1]
        gates._pos[dyn_gate_id-1][2] = p_now[2]
    pass
    
rospy.Timer(rospy.Duration(0.01), timer_cb)

rospy.Subscriber("/dynamic_gate", Point, dyn_gate_cb, queue_size=1, tcp_nodelay=True)

rospy.spin()
rospy.loginfo("ROS: Byby")