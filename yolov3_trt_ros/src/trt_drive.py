#!/usr/bin/env python2

import rospy, serial, time
from xycar_msgs.msg import xycar_motor
from yolov3_trt.msg import BoundingBoxes, BoundingBox

motor_msg = xycar_motor()
trt_msg = BoundingBoxes()
obj_id = -1

def callback(data) :
    global obj_id
    for bbox in data.bounding_boxes:
        print(bbox.id)
        obj_id = bbox.id
	
def drive_left():
	print("drive_left")
	global distance, motor_msg
	motor_msg.speed = 30
	motor_msg.angle = -10
	pub.publish(motor_msg)

def drive_right():
	print("drive_right")
	global distance, motor_msg
	motor_msg.speed = 30
	motor_msg.angle = 10
	pub.publish(motor_msg)

def drive_stop():
    print("drive_stop")
    global distance, motor_msg
    motor_msg.speed = 0
    motor_msg.angle = 0
    pub.publish(motor_msg)

def find_traffic_light():
    print("find_traffic_light")
    global distance, motor_msg
    motor_msg.speed = 0
    motor_msg.angle = 0
    pub.publish(motor_msg)

def find_cross_walk():
    print("find_cross_walk")
    global distance, motor_msg
    motor_msg.speed = 0
    motor_msg.angle = 0
    pub.publish(motor_msg)
    
def find_u_turn():
    print("find_u_turn")
    global distance, motor_msg
    motor_msg.speed = 0
    motor_msg.angle = 0
    pub.publish(motor_msg)


rospy.init_node('trt_driver')
rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, callback, queue_size=1)
pub = rospy.Publisher('xycar_motor',xycar_motor,queue_size=1)

rate = rospy.Rate(10)

while not rospy.is_shutdown():
    if obj_id == 0:
        drive_left()
    elif obj_id == 1:
        drive_right()
    elif obj_id == 2:
        drive_stop()
    elif obj_id == 3:
        find_cross_walk()
    elif obj_id == 4:
        find_u_turn()
    elif obj_id == 5:
        find_traffic_light()
    
    #reset obj_id
    obj_id = -1

    rate.sleep()