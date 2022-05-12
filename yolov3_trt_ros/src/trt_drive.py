#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, cv2, math
import numpy as np
from xycar_msgs.msg import xycar_motor
from yolov3_trt.msg import BoundingBoxes, BoundingBox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

### PID
class PID():
    def __init__(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def pid_control(self, cte):
        self.d_error = cte-self.p_error
        self.p_error = cte
        self.i_error += cte
        self.angle =  self.kp*self.p_error + self.ki*self.i_error + self.kd*self.d_error

        return self.angle

### Stanley Method
## https://velog.io/@legendre13/Stanley-Method
## https://github.com/zhm-real/MotionPlanning
def stanley():
    return 
# def get_steer_angle(curr_position, l_slope, r_slope):
#     # Lane tracking algorithm here

#     k = 1.0

#     if -0.2 < curr_position < 0.2 :  # 좀 더 천천히 조향해도 괜찮은 상황
#         k = 1.0
    
#     elif curr_position > 0.4 or curr_position < -0.4 :  # 신속하게 가운데로 들어와야 함
#         k = 4.0
        
#     else:   # 그 중간의 경우 계수는 linear 변화
#         k = 20.0 * abs(curr_position) - 3.0

#     steer_angle = k * math.atan(curr_position)* 180 / math.pi

#     return steer_angle




class Detect:
    def __init__(self):
        rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.bbox_callback, queue_size=1)
        self.bboxes = []

        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        self.bridge = CvBridge()
        self.image = None

        ### 영상처리 파라미터들
        self.kernel_size = 5
        self.canny_thres1 = 140 # TODO 바꿀 것
        self.canny_thres2 = 70 # TODO 바꿀 것
        self.roi_x = 380    # TODO 바꿀 것
        self.roi_h = 60 # TODO 바꿀 것
        self.hough_thres = 30 # TODO 바꿀 것
        self.hough_minLineLen = 30 # TODO 바꿀 것
        self.hough_maxLineGap = 10 # TODO 바꿀 것

        ### 차선 구별 상수값
        self.min_slope = 0 # TODO 바꿀 것
        self.max_slope = 10 # TODO 바꿀 것
        self.line_thick = 20    # TODO 임의값
        self.min_mid_to_b = 50   # TODO 임의값 / 프레임 중앙에서 이 수치 이상 벗어나야 차선으로 인식함
        self.lane_half = 220    # TODO 임의값 / 450(도로 폭 대략)의 약 절반값
        self.prev_mid = 320
        self.min_pos_gap = 20   # TODO 임의값 / 이전 mid와 지금 mid가 얼마나 차이가 나나

        ### 객체 인식 변수들
        self.obj_id = -1 # -1이면 아무것도 검출 안됨
        #XXX 임시용 maxareabbox 및 클래스 색성 저장 변수
        #self.obj_bbox=[]
        #self.color=[(0, 255, 0),(0,128,0),(255,0,0),(128,0,0),(0,255,255),(0,128,128)]
        self.min_box_area = 100 # TODO 임의값임
        self.min_probability = 0.3  # TODO 임의값임
        self.can_trust = False  # 탐지한 박스들을 믿을 수 있는가. False이면 박스 아무것도 없다 보고 차선 탐지만 하면 됨

        ### 신호 탐지 변수들
        self.light_color = "red"    # "red", "green"
        self.light_color_distinguish = 230

        # XXX createTrackbar, 범위 수정 필요
        cv2.namedWindow("trackbar")
        cv2.createTrackbar('min_pos_gap','trackbar', 20, 100, lambda x : x)
        cv2.createTrackbar('min_box_area','trackbar', 30, 200, lambda x : x)    
        cv2.createTrackbar('min_probability','trackbar', 0, 10, lambda x : x)
        cv2.createTrackbar('light_color_distinguish','trackbar', 200, 255, lambda x : x)
        cv2.createTrackbar('min_mid_to_b','trackbar', 40, 60, lambda x : x)   
        cv2.createTrackbar('line_thick','trackbar', 10, 30, lambda x : x)


    #XXX 필요한 파라미터 추가
    def set_param(self, min_pos_gap,min_box_area,min_probability,
                    light_color_distinguish,min_mid_to_b,line_thick):
        self.min_pos_gap = min_pos_gap
        self.min_box_area = min_box_area
        self.min_probability = min_probability / float(10)
        self.light_color_distinguish = light_color_distinguish
        self.min_mid_to_b = min_mid_to_b
        self.line_thick = line_thick

    def bbox_callback(self, msg):
        for box in msg.bounding_boxes:
            self.bboxes.append(box)

    def img_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def select_object(self):
        if len(self.bboxes) == 0:
            self.obj_id = -1
            #XXX temp
            #self.obj_bbox=[]

        max_area = 0
        self.can_trust = False
        max_area_obj = self.bboxes[0]
        for box in self.bboxes:
            area = (box.xmax - box.xmin) * (box.ymax - box.ymin)
            if area < self.min_box_area or box.probability < self.min_probability:
                continue

            if max_area < area:
                max_area = area
                max_area_obj = box

            self.can_trust = True
        self.obj_id = max_area_obj.id
        #XXX temp
        #self.obj_bbox = max_area_obj[1:5]
        #cv2.rectangle(self.image, (self.obj_bbox[0],self.obj_bbox[1]),(self.obj_bbox[2],self.obj_bbox[3]),self.color[self.obj_id], 2)

    def detect_lane(self):
        while not self.image.size == (640 * 480 * 3):
            continue

        roi = self.process_image()
        
        lines = cv2.HoughLinesP(roi, 1, math.pi / 180, self.hough_thres, self.hough_minLineLen, self.hough_maxLineGap)
        if lines in None:
            return -1

        left_tilt, right_tilt= self.filter_lines(lines) # 왼쪽으로 기울어짐(직선의 우측 차선) / 오른쪽으로 기울어짐(직선의 좌측 차선)
        
        if self.obj_id == 0:    #좌회전
            left_pos = self.select_left_lane(left_tilt)
            lane_mid_pos = left_pos + self.lane_half
        elif self.obj_id == 1:  # 우회전
            # TODO 만약에 오른쪽에 차선이 없어서 640으로 right_pos가 들어갔는데 왼쪽 차선은 잡혔다면??
            right_pos = self.select_right_lane(right_tilt)
            lane_mid_pos = right_pos - self.lane_half
        else:    # 직진 차선일 때
            right_pos = self.select_right_lane(right_tilt)  # 왼쪽 차선
            left_pos = self.select_left_lane(left_tilt) # 오른쪽 차선
            lane_mid_pos = (right_pos + left_pos) / 2

        #XXX 화면 실행
        cv2.imshow('view', self.image)

        if abs(self.prev_mid - lane_mid_pos) > self.min_pos_gap:
            lane_mid_pos = self.prev_mid    # 이동평균 적용된 값이 prev_mid에 있음(만약에 한참동안이나 prev가 업데이트 안되었다면??? --> 이동평균필터)
            return self.prev_mid
        else:
            return lane_mid_pos


    def process_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)
        canny = cv2.Canny(np.unit8(blur), self.canny_thres1, self.canny_thres2)
        roi = canny[self.roi_x : self.roi_x + self.roi_x, 0 : 640]
        return roi


    def filter_lines(self, lines):
        left_tilt = []
        right_tilt = []
        left_tilt_filtered = []
        right_tilt_filtered = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue    # 가로선

            slope = (y2 - y1) / (x2 - x1)
            if self.min_slope < abs(slope) < self.max_slope:
                if slope < 0:
                    b = self.find_vertical_mid(slope, x1, y1, x2, y2)
                    left_tilt.append([slope, b])
                else:
                    b = self.find_vertical_mid(slope, x1, y1, x2, y2)
                    right_tilt.append([slope, b]) # 오른쪽으로 기울어짐

        left_tilt.sort(key=lambda x:x[1])
        right_tilt.sort(key=lambda x:x[1])

        cur_line = left_tilt[0]
        for i in range(1, len(left_tilt)):
            if abs(cur_line[1] - left_tilt[i][1]) < self.line_thick:
                slope_avg = (cur_line[0] + left_tilt[i][0]) / 2
                b_avg = (cur_line[1] + left_tilt[i][1]) / 2
                cur_line = [slope_avg, b_avg]
            else:
                left_tilt_filtered.append(cur_line)
                cur_line = left_tilt[i]
        left_tilt_filtered.append(cur_line) # 마지막 라인 추가하는 부분
        
        cur_line = right_tilt[0]
        for i in range(1, len(right_tilt)):
            if abs(cur_line[1] - right_tilt[i][1]) < self.line_thick:
                slope_avg = (cur_line[0] + right_tilt[i][0]) / 2
                b_avg = (cur_line[1] + right_tilt[i][1]) / 2
                cur_line = [slope_avg, b_avg]
            else:
                right_tilt_filtered.append(cur_line)
                cur_line = right_tilt[i]
        right_tilt_filtered.append(cur_line)
        
        return left_tilt_filtered, right_tilt_filtered


    def find_vertical_mid(self, slope, x1, y1, x2, y2):
        x_avg = (x1 + x2) / 2
        y_avg = (y1 + y2) / 2
        b_temp = -slope * x_avg + y_avg # y=ax+b에서 a가 slope, b가 b_temp임. 현재.
        return ((self.roi_h / 2 ) - b_temp) / slope  # roi의 가운데의 x좌표


    def select_right_lane(self, right_tilt):
        b_sum = 0
        cnt = 0
        for line in right_tilt:
            if line[1] > 320 + self.min_mid_to_b:
                b_sum += line[1]
                cnt += 1
        if cnt == 0:
            return 640
        else:
            return (b_sum / cnt)


    def select_left_lane(self, left_tilt):
        b_sum = 0
        cnt = 0
        for line in left_tilt:
            if line[1] < 320 - self.min_mid_to_b:
                b_sum += line[1]
                cnt += 1
        if cnt == 0:
            return 0
        else:
            return b_sum / cnt

    def traffic_light_color(self):
        img = self.image
        red_green_mask = cv2.inRange(img, (117, 110, 74), (179, 240, 255)) # 빨간색과 초록색 구별하는 필터
        result_red_green = cv2.bitwise_and(img, img, mask=red_green_mask)
        
        breakcheck = False
        for i in range(result_red_green.shape[0]):
            for j in range(result_red_green.shape[1]):
                if result_red_green[i][j][1] > self.light_color_distinguish: # 기본은 230
                    self.light_color = "red"
                    breakcheck = True
                    break
            if(breakcheck == True):
                break
        if(breakcheck == False):
            self.light_color = "green"


class Drive:
    def __init__(self):
        self.motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        self.motor_msg = xycar_motor()
        self.default_speed = 15
        self.drive_mode = "None"
    
    def drive_normal(self, target_angle):
        self.drive_mode = "Straight"
        if self.motor_msg.speed < self.default_speed:
            cur_speed = self.motor_msg.speed
            for _ in range(5):
                if cur_speed >= self.default_speed:
                    break
                cur_speed += 1
                self.motor_msg.speed = cur_speed
                self.motor_msg.angle = target_angle
                self.motor_pub.publish(self.motor_msg)
                rospy.sleep(0.1)

            # 밑에대로 가면 중간에 회전각 변경 안되고 계속 그대로 유지하며 감.
            # 따라서 일정 시간만 속도 올리고 drive_normal을 나가서 다시 main의 while문으로 가야 함

            # while cur_speed >= self.default_speed:
            #     cur_speed += 1
            #     self.motor_msg.speed = cur_speed
            #     self.motor_msg.angle = target_angle
            #     self.motor_pub.publish(self.motor_msg)
            #     rospy.sleep(0.1)
        else:
            self.motor_msg.speed = self.default_speed
            self.motor_msg.angle = target_angle
            self.motor_pub.publish(self.motor_msg)


    def drive_rotate(self, target_angle): # TODO 속도 줄였다가 회전할 건지. 아니면 normal과 합쳐도 됨
        self.drive_mode = "Rotate"
        self.motor_msg.speed = self.default_speed
        self.motor_msg.angle = target_angle
        self.motor_pub.publish(self.motor_msg)


    def drive_stop(self):
        self.drive_mode = "Stop"
        remain_speed = self.motor_msg.speed
        while remain_speed > 0:
            remain_speed -= 1
            self.motor_msg.speed = remain_speed
            self.motor_msg.angle = 0
            self.motor_pub.publish(self.motor_msg)
            rospy.sleep(0.1)


class MovingAverageFilter:
    def __init__(self, n):
        self.n = n  # 큐 사이즈
        self.queue = []

    def add_data(self, x):
        if len(self.queue) >= self.n:
            self.queue = self.queue[1:] + [x]
        else:
            self.queue.append(x)

    def get_data(self):
        return sum(self.data)/len(self.data)

# TODO
# 좌우회전 진입했으면 기울기 다른 경우는 무시
# 이전 중앙값 저장해뒀으면 너무 달라지면 무시 -> 연속 몇 번 이상치이면 무시하도록 // DONE
# 표지판 좌우회전 잡힌 거로 기울기 하나만 뽑도록 함 // DONE
#    -> 좌회전이면 왼쪽 차선 따라가도록
# PID이든 Stanley든 제어 아직 없음
# imshow로 화면 출력
# 몇 번 이상 누적되면 탐지하는 기능 아직 없음
# 몇 번 이상 벗어나면 prev_mid 갱신하는 기능 없음 // DONE
# drive 쪽에서 정지는 상관 없는데 normal 쪽은 한 번 pub하고 돌아오도록 수정해야 함 // DONE
# 라인 필터링 기능 넣자. slope가 (오른쪽 기욺 -> 왼쪽 기욺) 형태로 있다던가. 한쪽에만 +, 한쪽에만 - 있어야 하는데 섞여있다던가 제외시키도록


def pixel_to_angle(pixel):
    # input_max = 640, input_min = 0
    # output_max = 50, output_min = -50

    angle = (pixel - 0) * (50 - (-50)) / (640 - 0) + (-50)
    return angle

def angle_to_pixel(angle):
    angle = (angle - (-50)) * (640 - (0)) / (50 - (-50)) + (0)
    return angle

def main():
    rospy.init_node('trt_drive', anonymous=False)

    detect = Detect()
    drive = Drive()
    angle_maf = MovingAverageFilter(5)
    prev_mid_maf = MovingAverageFilter(3)
    rate = rospy.Rate(10)

    obj_name = "None"

    rospy.sleep(3) # 노드 연결될 때까지 잠시 대기
    print("\n===== START =====\n")

    while not rospy.is_shutdown():
        detect.select_object()  # 가장 믿을만한 객체 선택

        lane_mid_pos = detect.detect_lane()
        if lane_mid_pos == -1:
            lane_mid_pos = detect.prev_mid  # lane_mid_pos가 -1이면 예외처리

        error_pixel = 320 - lane_mid_pos

        error_angle = pixel_to_angle(error_pixel)   # pixel(cam frame) -> angle(servo)
        fixed_angle = 제어기(error_angle)  # TODO 만들기!

        angle_maf.add_data(fixed_angle)
        target_angle = angle_maf.get_data()

        # XXX getTrackbarPos
        min_pos_gap = cv2.getTrackbarPos('min_pos_gap','trackbar')
        min_box_area = cv2.getTrackbarPos('min_box_area','trackbar')
        min_probability = cv2.getTrackbarPos('min_probability','trackbar')
        light_color_distinguish = cv2.getTrackbarPos('light_color_distinguish','trackbar')
        min_mid_to_b = cv2.getTrackbarPos('min_mid_to_b','trackbar')
        line_thick = cv2.getTrackbarPos('line_thick','trackbar')
        
        detect.set_param(min_pos_gap,min_box_area,min_probability,
                        light_color_distinguish,min_mid_to_b,line_thick)

        ### 탐지된 표지판에 따라 주행 모드 결정
        if detect.can_trust:        # True이면 표지판 잡은 것
            if detect.obj_id == 0:
                ### 좌회전
                obj_name = "Left"
                drive.drive_rotate(target_angle)
            elif detect.obj_id == 1:
                ###우회전
                obj_name = "Right"
                drive.drive_rotate(target_angle)
            elif detect.obj_id == 2:
                ### stop
                obj_name = "Stop"
                drive.drive_stop(target_angle)
            elif detect.obj_id == 3:
                ### crosswalk
                obj_name = "Crosswalk"
                drive.drive_stop(target_angle)
            elif detect.obj_id == 4:
                ### u-turn
                pass
            elif detect.obj_id == 5:
                ### traffic light
                obj_name = "Trafficlight"
                detect.traffic_light_color()
                ### 불 색깔 구별
                if detect.light_color == "red":
                    drive.drive_stop()
                else:
                    drive.drive_normal(target_angle)
        else:
            obj_name = "None"
            drive.drive_normal(target_angle)

        ####print status###
        print("Detected Object      {}".format(obj_name))
        if obj_name == "Trafficlight":
            print("                     ({})".format(detect.light_color))

        print("---------------------------------------------")
        print("Prev Mid             {}".format(detect.prev_mid))
        print("Curr Mid             {}".format(lane_mid_pos))
        print("CTE                  {}".format(error_pixel))
        print("Error Angle          {}".format(error_angle))
        print("After PID            {}".format(fixed_angle))
        print("After MAF            {}".format(target_angle))

        print("---------------------------------------------")
        print("Drive Mode           {}".format(drive.drive_mode))
        print("Speed                {}".format(drive.motor_msg.speed))
        print("Angle                {}".format(drive.motor_msg.angle))

        print("\n")

        prev_mid_maf.add_data(angle_to_pixel(target_angle))
        detect.prev_mid = prev_mid_maf.get_data()

        rate.sleep()

if __name__ == "__main__":
    main()