#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, cv2, math
import numpy as np
from xycar_msgs.msg import xycar_motor
from yolov3_trt.msg import BoundingBoxes, BoundingBox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Detect:
    def __init__(self):
        rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.bbox_callback, queue_size=1)
        self.bboxes = []

        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        self.bridge = CvBridge()
        self.image = None

        ### 영상처리 파라미터들
        self.kernel_size = 5
        self.Width = 640
        self.Height = 480
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
        self.light_color = "red"    # "red", "green", "orange"
        
        # XXX createTrackbar
        cv2.namedWindow("trackbar")
        cv2.createTrackbar('min_pos_gap','trackbar', 20, 100, lambda x : x)
        cv2.createTrackbar('min_box_area','trackbar', 30, 200, lambda x : x)
        cv2.createTrackbar('min_probability','trackbar', 0, 10, lambda x : x)
        
    #XXX 필요한 파라미터 추가
    def set_param(self, min_pos_gap,min_box_area,min_probability):
        self.min_pos_gap = min_pos_gap
        self.min_box_area = min_box_area
        self.min_probability = min_probability / float(10)

    def bbox_callback(self, msg):
        for box in msg.bounding_boxes:
            self.bboxes = box

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
                pass

            if max_area < area:
                max_area = area
                max_area_obj = box

            self.can_trust = True
        self.obj_id = max_area_obj.id
        #XXX temp
        #self.obj_bbox = max_area_obj[1:5]
        #cv2.rectangle(self.image, (self.obj_bbox[0],self.obj_bbox[1]),(self.obj_bbox[2],self.obj_bbox[3]),self.color[self.obj_id], 2)
# TODO

    """
    라인 필터링 기능 넣자.
    slope가 (오른쪽 기욺 -> 왼쪽 기욺) 형태로 있다던가
    한쪽에만 +, 한쪽에만 - 있어야 하는데 섞여있다던가
    제외시키도록
    """ 


    def detect_lane(self):
        while not self.image.size == (640 * 480 * 3):
            continue

        roi = self.process_image()
        
        lines = cv2.HoughLinesP(roi, 1, math.pi / 180, self.hough_thres, self.hough_minLineLen, self.hough_maxLineGap)
        if lines in None:
            # TODO 차선 탐색 안됨 -> 교차로에서 차선 없을 때 이걸로 들어가니까 가던대로 계속 가도록 하자
            pass
        
        left_tilt, right_tilt = self.seperate_lines(lines)  # 왼쪽으로 기울어짐(직선의 우측 차선) / 오른쪽으로 기울어짐(직선의 좌측 차선)
        
        ##### 여기 아래 막코딩... 수정 대대적으로 필요.
        line_pos = []
        # 좌우회전 표지판이 있을 때 -> 한쪽 차선만 씀
        if self.obj_id == 0:
            print(len(left_tilt))   # 지금은 left_tilt 안에 2개만 있다고 가정하고 진행
            for line in left_tilt:
                m, b = self.get_line_params(left_tilt)  # 두께 처리되어 하나의 선으로 들어감을 가정
                line_pos.append(self.get_line_pos(m, b))
        elif self.obj_id == 1:
            print(len(right_tilt))   # 지금은 right_tilt 안에 2개만 있다고 가정하고 진행
            for line in right_tilt:
                m, b = self.get_line_params(right_tilt)  # 두께 처리되어 하나의 선으로 들어감을 가정
                line_pos.append(self.get_line_pos(m, b))
        # 직진 차선일 때(정지일 경우는  일단 무시하세요. 지금은 차선 위치만 판단하는 거니까.)
        else:
            for line in left_tilt:  # 이상적일 경우 하나만 나옴
                m, b = self.get_line_params(left_tilt)  # 두께 처리되어 하나의 선으로 들어감을 가정
                line_pos.append(self.get_line_pos(m, b))
            right_pos = sum(line_pos) / len(line_pos)
            line_pos = []
            for line in right_tilt:  # 이상적일 경우 하나만 나옴
                m, b = self.get_line_params(right_tilt)  # 두께 처리되어 하나의 선으로 들어감을 가정
                line_pos.append(self.get_line_pos(m, b))
            left_pos = sum(line_pos) / len(line_pos)
            line_pos = [left_pos, right_pos]

        lane_mid = sum(line_pos) / len(line_pos)

        if abs(lane_mid - self.prev_mid) < self.min_pos_gap:
            pass    # 아직 구현 못함

        # TODO 만약에 한참동안이나 prev가 업데이트 안되었다면???
        
        #XXX 화면 실행
        cv2.imshow('view', self.image)
        
        return lane_mid

    def process_image(self):
        gray = cv2.cvtColor()
        blur = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)
        canny = cv2.Canny(np.unit8(blur), self.canny_thres1, self.canny_thres2)
        roi = canny[self.roi_x : self.roi_x + self.roi_x, 0 : 640]
        return roi

    def seperate_lines(self, lines):
        left_tilt = []
        right_tilt = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue    # 가로선

            slope = (y2 - y1) / (x2 - x1)
            if self.min_slope < abs(slope) < self.max_slope:
                if slope < 0:
                    left_tilt.append(line)
                else:
                    right_tilt.append(line) # 오른쪽으로 기울어짐
            # TODO 원래 코드에서 middle_thresh 부분 없앴음 크게 이상하면 추가하자

        # 선 두께 때문에 두 개의 선으로 된 부분 하나로 합치는 부분 여기 있었으면...
        # left_tilt.sort(key=lambda x:x[1])   # y1기준으로 정렬
        # right_tilt.sort(key=lambda x:x[1])
        # for i in range(len(left_tilt) - 1):
        #     y_avg = (left_tilt[i][1] + left_tilt[i][3]) / 2
        #     y_next_avg = (left_tilt[i+1][1] + left_tilt[i+1][3]) / 2
        #     if abs(y_avg - y_next_avg) < self.line_thick:

        return left_tilt, right_tilt

    def get_line_params(self, lines):
        x_sum = 0
        y_sum = 0.0
        m_sum = 0.0

        size = len(lines)
        # if size == 0:
        #     return 0, 0

        for line in lines:
            x1, y1, x2, y2 = line[0]

            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += float(y2 - y1) / float(x2 - x1)

        x_avg = x_sum / (size * 2)
        y_avg = y_sum / (size * 2)
        m = m_sum / size
        b = y_avg - m * x_avg

        return m, b

    def get_line_pos(self, m, b):
        ### m, b를 받아 픽셀 내에서 차선의 위치(y)값 구함
        if m == 0 and b == 0:
            # TODO 예외처리 어떻게 할지?
            pass
        else:
            y = self.roi_h / 2
            pos = (y - b) / m

            b+=self.roi_x
            x1 = (self.Height - b) / float(m)
            x2 = (self.Height/2 - b) / float(m)
            #XXX : line visualization
            cv2.line(self.image, (int(x1), self.Height), (int(x2), int(self.Height/2)), (255, 0, 0), 1)

        return pos

    def traffic_light_color(self):
        # self.light_color에 색 str로 부여
        # TODO 형석 오라버니 여기에 채워주세요
        pass

    def mb_to_xy(self, m, b):

        return [x1, y1, x2, y2]

    def xy_to_mb(self, coor):
        x1, y1, x2, y2 = coor
        m = float(y2 - y1) / float(x2 - x1)

        x_avg = (x2 + x1) / 2
        y_avg = (y2 + y1) / 2
        b = y_avg - m * x_avg
        return m, b
    
        
class Drive:
    def __init__(self):
        self.motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        self.motor_msg = xycar_motor()
        self.default_speed = 15
        self.drive_mode = "None"
    
    def pixel_to_angle(self, target_pixel):
        # input_max = 640, input_min = 0
        # output_max = 50, output_min = -50

        target_angle = (target_pixel - 0) * (50 - (-50)) / (640 - 0) + (-50)
        return target_angle

    def drive_normal(self, target_pixel):
        self.drive_mode = "Straight"
        if self.motor_msg.speed < self.default_speed:
            cur_speed = self.motor_msg.speed
            while cur_speed >= self.default_speed:
                cur_speed += 1   # TODO 속도 증가 폭 조정하기
                self.motor_msg.speed = cur_speed
                self.motor_msg.angle = self.pixel_to_angle(target_pixel)
                self.motor_pub.publish(self.motor_msg)
                rospy.Rate.sleep(10) # TODO 시간 따라 줄일 것인지? rospy.Rate 설정해서?
        else:
            self.motor_msg.speed = self.default_speed
            self.motor_msg.angle = self.pixel_to_angle(target_pixel)
            self.motor_pub.publish(self.motor_msg)

    
    def drive_rotate(self, target_pixel):
        # 속도 줄였다가 회전할 건지. 아니면 normal과 합쳐도 됨
        self.drive_mode = "Rotate" # TODO 왼쪽 오른쪽 나누기
        self.motor_msg.speed = self.default_speed
        self.motor_msg.angle = self.pixel_to_angle(target_pixel)
        self.motor_pub.publish(self.motor_msg)

    def drive_stop(self):
        self.drive_mode = "Stop"
        remain_speed = self.motor_msg.speed
        while remain_speed > 0:
            remain_speed -= 1   # TODO 속도 감소 폭 조정하기
            self.motor_msg.speed = remain_speed
            self.motor_msg.angle = 0
            self.motor_pub.publish(self.motor_msg)
            rospy.Rate.sleep(10) # TODO 시간 따라 줄일 것인지? rospy.Rate 설정해서?


        

# TODO
# 좌우회전 진입했으면 기울기 다른 경우는 무시
# 이전 중앙값 저장해뒀으면 너무 달라지면 무시 -> 연속 몇 번 이상치이면 무시하도록
# 표지판 좌우회전 잡힌 거로 기울기 하나만 뽑도록 함(구현함. 섬세한 수정 남음)
#    -> 좌회전이면 왼쪽 차선 따라가도록 stanley처럼
# PID이든 Stanley든 제어 아직 없음
# imshow로 화면 출력

def main():
    rospy.init_node('trt_drive', anonymous=False)

    detect = Detect()
    drive = Drive()
    rate = rospy.Rate(10)

    

    while not rospy.is_shutdown():
        detect.select_object()  # 가장 믿을만한 객체 선택
        pos = detect.detect_lane()
        error_pixel = 320 - pos
        target_pixel = 제어기(error_pixel)
        
        # XXX getTrackbarPos
        min_pos_gap = cv2.getTrackbarPos('min_pos_gap','trackbar')
        min_box_area = cv2.getTrackbarPos('min_box_area','trackbar')
        min_probability = cv2.getTrackbarPos('min_probability','trackbar')
        detect.set_param(min_pos_gap,min_box_area,min_probability)

        ### 탐지된 표지판에 따라 주행 모드 결정
        if detect.can_trust:        # True이면 표지판 잡은 것
            if detect.obj_id == 0:
                ### 좌회전
                drive.drive_rotate(target_pixel)
            elif detect.obj_id == 1:
                ###우회전
                drive.drive_rotate(target_pixel)
            elif detect.obj_id == 2:
                ### stop
                drive.drive_stop(target_pixel)
            elif detect.obj_id == 3:
                ### crosswalk
                drive.drive_stop(target_pixel)
            elif detect.obj_id == 4:
                ### u-turn
                pass
            elif detect.obj_id == 5:
                ### traffic light
                ### 불 색깔 구별
                if detect.light_color == "red":
                    drive.drive_stop()
                elif detect.light_color == "orange":
                    pass
                else:
                    drive.drive_normal(target_pixel)
        else:
            drive.drive_normal(target_pixel)

        ####print status###

        print(drive.drive_mode)
        print(drive.motor_msg.speed)
        print(drive.motor_msg.angle)
    
        rospy.sleep()

if __name__ == "__main__":
    main()