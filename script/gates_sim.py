#! /usr/bin/python3.8

import rospy
import numpy as np

from fast_fly_waypoint.msg import TrackTraj
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

import os, sys
BASEPATH = os.path.abspath(__file__).split('script', 1)[0]+'script/fast_fly/'
sys.path += [BASEPATH]

from gates.gates import Gates

import time

rospy.init_node("gates_sim")
rospy.loginfo("ROS: Hello")
gates_pub = rospy.Publisher("~gates", TrackTraj, tcp_nodelay=True, queue_size=1)
gates_marker_pub = rospy.Publisher("/plan/gates_marker", Marker, queue_size=1)

# gates = Gates(BASEPATH+"gates/gates_n5.yaml")
gates = Gates()
# gates.add_gate([ 0, 1, -1])
# gates.add_gate([-1, 0, -1])
# gates.add_gate([ 0,-1, -1])
# gates.add_gate([ 1.2, 0, -1])
gates.add_gate([0,0,-0.4])
gates.add_gate([-3.9,-1.6,-1.55])
gates.add_gate([-7.3,0,-0.4])
gates.add_gate([-9.5,1.36,-1.22])
gates.add_gate([-9.2,-1.52,-1.32])
gates.add_gate([-6.8,0.45,-1.33])
gates.add_gate([-3.94,1.35,-1.2])

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



rospy.Timer(rospy.Duration(0.1), timer_cb)

rospy.spin()
rospy.loginfo("ROS: Byby")