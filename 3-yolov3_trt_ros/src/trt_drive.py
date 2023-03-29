#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, cv2, math
import numpy as np
from cv_bridge import CvBridge

from xycar_msgs.msg import xycar_motor
from yolov3_trt_ros.msg import BoundingBoxes, BoundingBox
from sensor_msgs.msg import Image


### PID (적용하지는 않음)
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
class stanley():
    def __init__(self):
        self.k = 1
        self.v = 2

    def control(self, lane_mid_pos):
        cte = 320 - lane_mid_pos
        cte = np.arctan2(self.k * cte, self.v)
        return cte

class Detect:
    """
    객체를 탐지하고, 차선을 인식하는 클래스
    """

    def __init__(self):
        rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.bbox_callback, queue_size=1)
        self.bboxes = []

        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        self.bridge = CvBridge()
        self.image = None

        ### 영상처리 파라미터들
        self.kernel_size = 5
        self.canny_thres1 = 140
        self.canny_thres2 = 70
        self.roi_y = 330
        self.roi_h = 60
        self.hough_thres = 30
        self.hough_minLineLen = 15
        self.hough_maxLineGap = 10

        ### 차선 구별 상수값
        self.min_slope = 0.2
        self.max_slope = 40
        self.line_thick = 0
        self.min_mid_to_b = 50   # 프레임 중앙에서 이 수치 이상 벗어나야 차선으로 인식함
        self.lane_half = 250    # 450(도로 폭 대략)의 약 절반값
        self.prev_mid = 350
        self.min_pos_gap = 100   # 이전 mid와 지금 mid가 얼마나 차이가 나나

        ### 객체 인식 변수들
        self.obj_id = -1 # -1이면 아무것도 검출 안됨
        #self.obj_bbox=[]   # 선택된 bbox에 대한 정보를 저장
        self.min_box_area = 2500
        self.min_probability = 0.4
        self.can_trust = False  # 탐지한 박스들을 믿을 수 있는가. False이면 박스 아무것도 없다 보고 차선 탐지만 하면 됨

        ### 신호 탐지 변수들
        self.light_color = "green"    # "red", "green" 둘 중 하나
        self.light_color_distinguish = 230  # 특정 픽셀의 값이 red인지 green인지 판단할 임곗값

        ### 트랙바 선언
        cv2.namedWindow("trackbar")
        cv2.createTrackbar('min_pos_gap','trackbar', 20, 100, lambda x : x)
        cv2.createTrackbar('min_box_area','trackbar', 30, 200, lambda x : x)    
        cv2.createTrackbar('min_probability','trackbar', 0, 10, lambda x : x)
        cv2.createTrackbar('light_color_distinguish','trackbar', 200, 255, lambda x : x)
        cv2.createTrackbar('min_mid_to_b','trackbar', 40, 60, lambda x : x)   
        cv2.createTrackbar('line_thick','trackbar', 10, 30, lambda x : x)


    def set_param(self, min_pos_gap, min_box_area, min_probability, light_color_distinguish, min_mid_to_b, line_thick):
        """
        트랙바를 사용해 해당 변수들의 값을 변경할 때 사용
        """

        self.min_pos_gap = min_pos_gap
        self.min_box_area = min_box_area
        self.min_probability = min_probability / float(10)
        self.light_color_distinguish = light_color_distinguish
        self.min_mid_to_b = min_mid_to_b
        self.line_thick = line_thick


    def bbox_callback(self, msg):
        self.bboxes = []    # 누적을 막기 위해 callback마다 초기화
        for box in msg.bounding_boxes:
            self.bboxes.append(box)


    def img_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")


    def select_object(self):
        """
        탐지된 객체 중 가장 믿을만한 객체(=주행 모드를 결정할 표지)를 선택함
        선택의 기준
            (1) 가장 넓은 bbox 크기, 최소 bbox 크기 이상일 것
            (2) 최소 probability 이상일 것
        """

        self.can_trust = False
        if len(self.bboxes) == 0:
            self.obj_id = -1    # 아무것도 탐지되지 않았을 때
            self.obj_bbox=[]    # bbox 목록을 다 비움
            self.obj_prob=0.0
            return

        ids = [i.id for i in self.bboxes]
        max_area = 0
            # 객체가 탐지되었어도 믿을만한 객체가 아닐 수 있음. 조건을 충족시킨 객체가 있다면 True, 아니면 False
        max_area_obj = []
            # 가장 넓은 bbox 넓이를 가지는 객체. 초깃값은 idx = 0으로 임의 할당
        for box in self.bboxes: # 탐지된 모든 bbox를 순회
            area = (box.xmax - box.xmin) * (box.ymax - box.ymin)    # bbox의 크기
            if area < self.min_box_area or box.probability < self.min_probability:
                continue

            if box.xmax<0 or box.xmin<0 or box.ymax<0 or box.ymin<0:
                continue

            if box.id ==5 and area/3 <self.min_box_area:
                continue

            if max_area < area: # 가장 넓은 bbox를 선택해나감
                max_area = area
                max_area_obj = box

            self.can_trust = True   # continue가 아니라면 믿을만한 객체가 하나는 선택되었다는 뜻이므로 True로 변경
            print("max area = ", max_area)
        if self.can_trust == False:
            self.obj_id = -1
            self.obj_temp_id = -1
            self.obj_detect_cnt = 0
            return

        if self.obj_temp_id == max_area_obj.id:
            self.obj_detect_cnt += 1
            if self.obj_detect_cnt > 3:
                self.obj_id = self.obj_temp_id
                self.obj_detect_cnt = 0
                self.obj_prob = max_area_obj.probability
        else:
            self.obj_temp_id = max_area_obj.id
            self.obj_detect_cnt += 1
        
        self.obj_bbox=[int(max_area_obj.xmin*640/416),int(max_area_obj.ymin*480/416),int(max_area_obj.xmax*640/416),int(max_area_obj.ymax*480/416)]

        ### 최종 선택된 객체를 화면에 그림
        cv2.rectangle(self.image, (self.obj_bbox[0],self.obj_bbox[1]),(self.obj_bbox[2],self.obj_bbox[3]),self.color[self.obj_id], 2)
        cv2.putText(self.image, "{} : {:.2f}".format(self.obj_id,self.obj_prob),(self.obj_bbox[0] - 10, self.obj_bbox[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,1)


    def detect_lane(self):
        """
        차선을 탐지해 차선의 중간 지점을 찾음. 최종적으로는 이 중간 지점으로 이동하고자 함
        """

        while not self.image.size == (640 * 480 * 3):
            continue

        roi = self.process_image()
            # 이미지를 사전 처리하고 캐니 에지 검출기로 직선을 추출한 뒤 ROI를 자름
        cv2.imshow("roi", roi)
        
        lines = cv2.HoughLinesP(roi, 1, math.pi / 180, self.hough_thres, self.hough_minLineLen, self.hough_maxLineGap)
            # 허프라인 변환을 사용해 ROI 내에서 직선을 검출
        
        if lines is None:
            return -1

        left_tilt, right_tilt = self.filter_lines(lines)
            # 각각 왼쪽으로 기울어진 직선(직선구간으로 치자면 우측 차선) / 오른쪽으로 기울어진 직선(직선구간으로 치자면 좌측 차선)

        ### 각 직선을 그림
        self.drawlane(left_tilt, right_tilt)
        # print("left:{}".format(left_tilt))
        # print("right:{}".format(right_tilt))

        ### 탐지한 객체에 맞추어 '차선 중앙 위치'를 결정함
        if self.obj_id == 0:    #좌회전
            left_pos = self.select_left_lane(left_tilt)
            lane_half_size = self.prev_mid - left_pos
            lane_mid_pos = left_pos + lane_half_size
            # cv2.circle(self.image, (int(lane_mid_pos),int((self.roi_x + self.roi_h/2))), 5, (0,0,255),3)

        elif self.obj_id == 1:  # 우회전
            right_pos = self.select_right_lane(right_tilt)
            lane_half_size = -(self.prev_mid - right_pos)
            lane_mid_pos = right_pos + lane_half_size
            # cv2.circle(self.image, int((lane_mid_pos),int((self.roi_x + self.roi_h/2))), 5, (0,0,255),3)
        else:    # 직진 차선일 때
            right_pos = self.select_right_lane(right_tilt)  # 왼쪽 차선
            left_pos = self.select_left_lane(left_tilt) # 오른쪽 차선
            lane_mid_pos = (right_pos + left_pos) / 2
            print("right {} left {} mid {}".format(right_pos, left_pos, lane_mid_pos))

        cv2.imshow('view', self.image)

        ### 최종적으로 조향각을 좌우할 '차선 중앙 위치'를 결정함
        # if abs(self.prev_mid - lane_mid_pos) > self.min_pos_gap:
        #     lane_mid_pos = self.prev_mid    
        #         # 이동평균필터가 적용되었던 '이전 프레임에서 차선의 중앙 위치'와 현재 계산된 값이 차이가 크다면
        #         # 현재 계산된 값이 이상치나 노이즈 등으로 인해 잘못된 값이 되었다고 판단
        #         # 이전 값을 그대로 다시 사용함
        #     return self.prev_mid
        # else:
        #     return lane_mid_pos
        return lane_mid_pos


    def process_image(self):
        """
        grayscale 변환 -> 가우시안 블러 -> 캐니 에지 검출기 -> ROI 자르기
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)
        canny = cv2.Canny(blur, self.canny_thres1, self.canny_thres2)
        roi = canny[self.roi_y : self.roi_y + self.roi_h, 0 : 640]
        return roi


    # def DrawLane(self, lines):
    #     """
    #     직선들을 화면에 그림. ROI의 높이가 작기 때문에 조금 더 확장해 그림
    #     """
    #     for line in lines:
    #         m, b = line
    #         b += self.roi_y # 현재 b는 roi 내부의 값이므로, 화면 전체를 따졌을 땐.roi_y(roi 시작 픽셀)을 더해줘야 함
    #         x1 = (480 - b) / float(m)   # y=480일 때 x좌표. 직선의 아래점
    #         x2 = (240 - b) / float(m)   # y=240일 때 x좌표. 직선의 윗점

    #         cv2.line(self.image, (int(x1), 480), (int(x2), 240), (255, 0, 0), 1)
    
    
    def drawlane(self, left_tilt, right_tilt):
        # slope를 평균, b를 평균내서 대표 lane만 표시
        if len(left_tilt) == 0 or len(right_tilt) == 0:
            return

        y = 300 # 임의값, 선 그리고 싶은 길이 지정
        left_avg = np.array(left_tilt).mean(0).tolist() # 리스트에서 각 index별로 평균내서 list로 변환, slope와 b 각각의 평균 계산
        right_avg = np.array(right_tilt).mean(0).tolist() 
        # print("left_avg", left_avg)
        # print("right_avg", right_avg)
        
        l_slope , lx1 = left_avg[0], int(left_avg[1]) # lx1은 pixel 단위로 할 것이므로 int
        r_slope , rx1 = right_avg[0], int(right_avg[1])
        
        l_b = (self.roi_x + self.roi_h/2) - lx1 * l_slope # lx1은 roi의 중앙을 기준으로 했으므로 roi_x + roi_h/2
        lx2 = (y - l_b) / l_slope
        
        r_b = (self.roi_x + self.roi_h/2) - rx1 * r_slope
        rx2 = (y - r_b) / r_slope

        cv2.circle(self.image, (lx1,(self.roi_x + self.roi_h/2)), 2, (255,0,0),3)
        cv2.circle(self.image, (rx1,(self.roi_x + self.roi_h/2)), 2, (0,255,0),3)
        cv2.line(self.image, (lx1,(self.roi_x + self.roi_h/2)), (int(lx2), y), (255,0,0), 2)
        cv2.line(self.image, (rx1,(self.roi_x + self.roi_h/2)), (int(rx2), y), (0,255,0), 2)
        

    def filter_lines(self, lines):
        """
        검출된 직선들 중에서 왼쪽으로 기울어진 직선과 오른쪽으로 기울어진 직선을 분리
        """
        left_tilt = []  # 단순히 왼쪽으로 기울어진 직선
        right_tilt = [] # 단순히 오른쪽으로 기울어진 직선
        left_tilt_filtered = []      # 인접하거나 한 차선인 직선을 합친, 왼쪽으로 기울어진 직선
        right_tilt_filtered = []     # 인접하거나 한 차선인 직선을 합친, 오른쪽으로 기울어진 직선

        ### 검출된 모든 직선을 순회하며 기울기와 직선이 지나는 한 점을 계산
        for line in lines:
            x1, y1, x2, y2 = line[0]
                # 1번 점과 2번 점을 구분하는 기준은 x 기준으로 더 작은 값
            if x2 - x1 == 0:
                continue    # 가로선은 제외

            x1_f, y1_f, x2_f, y2_f = float(x1), float(y1), float(x2),float(y2)
            slope = round((y2_f - y1_f) / (x2_f - x1_f),2)   # 직선의 기울기
            
            if self.min_slope < abs(slope) < self.max_slope:    # 최대/최소 기울기 이내의 직선만 택함
                if slope > 0:   # 기울기가 양수라면 왼쪽으로 기울어짐
                    b = self.find_vertical_mid(slope, x1, y1, x2, y2)   # 직선이 ROI 중앙과 만나는 x 좌표
                    left_tilt.append([slope, b])
                else:
                    b = self.find_vertical_mid(slope, x1, y1, x2, y2)   # 직선이 ROI 중앙과 만나는 x 좌표
                    right_tilt.append([slope, b])

        ### x좌표가 낮은 순서대로(왼쪽에 있는 직선부터) 정렬
        left_tilt.sort(key=lambda x:x[1])
        right_tilt.sort(key=lambda x:x[1])

        ### 왼쪽으로 기울어진 직선들을 필터링: 가까이 위치한 직선들끼리 합침
        if len(left_tilt) != 0:
            cur_line = left_tilt[0] # 현재 계산중인 직선
            for i in range(1, len(left_tilt)):
                if abs(cur_line[1] - left_tilt[i][1]) < self.line_thick:
                        # 차선의 두께보다 두 직선 사이의 거리(x 값의 차)가 작으면 합침
                    slope_avg = (cur_line[0] + left_tilt[i][0]) / 2
                    b_avg = (cur_line[1] + left_tilt[i][1]) / 2
                    cur_line = [slope_avg, b_avg]
                        # 직선을 합칠 때는 두 직선의 기울기와 각각이 지나는 점을 평균냄
                else:
                    # 바로 이전 직선과 거리가 멀리 떨어져 있다면 각자 다른 직선이라 여겨 바로 추가
                    left_tilt_filtered.append(cur_line)
                    cur_line = left_tilt[i]
            left_tilt_filtered.append(cur_line) # 마지막 라인 추가하는 부분
        
        ### 오른쪽으로 기울어진 직선들을 필터링: 가까이 위치한 직선들끼리 합침
        if len(right_tilt) != 0:
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
        """
        직선을 ROI를 가로지르도록 그었을 때 ROI의 세로 중간 지점을 지나는 x좌표를 구함
        """
        x_avg = (x1 + x2) / 2
        y_avg = (y1 + y2) / 2
        b_temp = -slope * x_avg + y_avg # 직선이 지나는 한 점. y=mx+k 식을 사용
        return ((self.roi_h / 2 ) - b_temp) / slope  # roi의 가운데의 x좌표


    def select_right_lane(self, right_tilt):
        """
        오른쪽으로 기울어진 직선들 중 대표 직선 하나를 만듦
        """
        b_sum = 0
        cnt = 0
        for line in right_tilt:
            if line[1] < 320 + self.min_mid_to_b:
                # 화면 중앙(+ span)보다 왼쪽에 있는 직선들 중에서만 대표 직선을 찾음
                b_sum += line[1]
                cnt += 1
        if cnt == 0:
            return 0    # 화면의 왼쪽에 직선이 없다면 직선의 위치는 0
        else:
            return (b_sum / cnt)  # 대표 직선의 x좌표 위치


    def select_left_lane(self, left_tilt):
        """
        왼쪽으로 기울어진 직선들 중 대표 직선 하나를 만듦
        """
        b_sum = 0
        cnt = 0
        for line in left_tilt:
            if line[1] > 320 - self.min_mid_to_b:
                # 화면 중앙(+ span)보다 오른쪽에 있는 직선들 중에서만 대표 직선을 찾음
                b_sum += line[1]    # roi의 중앙에 해당하는 x점을 다 더함(나중에 평균함)
                cnt += 1
        if cnt == 0:
            return 640    # 화면의 오른쪽에 직선이 없다면 직선의 위치는 0
        else:
            return b_sum / cnt  # 대표 직선의 x좌표 위치


    def traffic_light_color(self):
        """
        신호등을 탐지했을 때 신호등의 색깔을 구별함. 노란색은 빨간색과 같다고 가정, 빨강과 초록만 구별함
        """
        img = self.image
        red_green_mask = cv2.inRange(img, (117, 110, 74), (179, 240, 255)) # 빨간색과 초록색 구별하는 필터
        result_red_green = cv2.bitwise_and(img, img, mask=red_green_mask)
        
        crop_img_up = result_red_green[self.obj_bbox[1]:self.obj_bbox[1]+int((self.obj_bbox[3]-self.obj_bbox[1])/3),\
            self.obj_bbox[0]:self.obj_bbox[2]]   #ymin:ymin+(ymax-ymin/2),xmin:xmax]
        cv2.imshow("crop", crop_img_up)

        breakcheck = False
        for i in range(crop_img_up.shape[0]):
            for j in range(crop_img_up.shape[1]):
                # 각 픽셀을 돌면서 해당 픽셀이 설정된 값을 넘으면 빨강이라 간주
                if crop_img_up[i][j][1] > self.light_color_distinguish:
                    self.light_color = "red"
                    breakcheck = True
                    break
            if(breakcheck == True):
                break
        if(breakcheck == False):
            self.light_color = "green"


class Drive:
    """
    주행 모드(직진, 회전, 정지 등)에 따라 모터로 명령을 내리는 클래스
    """

    def __init__(self):
        self.motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        self.motor_msg = xycar_motor()
        self.default_speed = 4 # 주행 기본 속도
        self.drive_mode = "None"    # 주행 모드 (Straight, Rotate, Stop)
    
    def drive_normal(self, target_angle):
        """
        기본 주행 모드로, 주로 직선 구간에서 사용
        """
        self.drive_mode = "Straight"
        if self.motor_msg.speed < self.default_speed:
            # 기본 속도보다 작을 때는 0.5초간 서서히 속도를 증가시킴
            # 정지했다가 줄발할 때 급히 가속하게 되면 모터에 무리가 갈 수 있음
            cur_speed = self.motor_msg.speed
            for _ in range(5):
                if cur_speed >= self.default_speed:
                    break
                cur_speed += 1
                self.motor_msg.speed = cur_speed
                self.motor_msg.angle = target_angle
                self.motor_pub.publish(self.motor_msg)
                rospy.sleep(0.1)
        else:
            self.motor_msg.speed = self.default_speed
            self.motor_msg.angle = target_angle
            self.motor_pub.publish(self.motor_msg)

    def drive_rotate(self, target_angle, dir):
        """
        회전 시의 모드. drive_normal과 다를 것은 없음
        """
        self.drive_mode = "Rotate"
        self.motor_msg.speed = self.default_speed
        if dir == 0 and target_angle > 0:
            self.drive_mode = "left"
            target_angle *= -1
        
        if dir== 1 and target_angle < 0:
            self.drive_mode = "right"
            target_angle *= -1
        self.motor_msg.angle = target_angle
        self.motor_pub.publish(self.motor_msg)


    def drive_stop(self, detect):
        """
        정지 모드
        """
        self.drive_mode = "Stop"

        # remain_speed = self.motor_msg.speed
        # while remain_speed > 0:
        #     # 급정거를 할 시 모터에 무리가 갈 수 있으므로
        #     # 속도가 0이 될 때까지 0.1초에 1씩 감소시키며 서서히 속도를 줄이도록 함
        #     remain_speed -= 1
        #     self.motor_msg.speed = remain_speed
        #     self.motor_msg.angle = 0
        #     self.motor_pub.publish(self.motor_msg)
        #     rospy.sleep(0.1)

        self.motor_msg.speed = 0
        self.motor_pub.publish(self.motor_msg)

        rospy.sleep(5)

        time = 4
        while time > 0:
            time -= 0.5
            lane_mid_pos = detect.detect_lane()

            # 차선 중앙 ( lane_mid_pos , roi 높이 + roi 크기/2, 빨간색)
            cv2.circle(detect.image, (int(lane_mid_pos), int(detect.roi_x + detect.roi_h//2)),4,(0,0,255),2)    
            # 화면 중앙 ( 화면 중앙, roi 높이 + roi 크기/2, 청록?색)
            cv2.circle(detect.image, (320, int(detect.roi_x + detect.roi_h//2)),4,(200,200,0),2)
        
            error_pixel = 320 - lane_mid_pos

            error_angle = pixel_to_angle(error_pixel)   # pixel(cam frame) -> angle(servo)

            self.drive_rotate(error_angle, dir=3)
            # self.motor_pub.publish(self.motor_msg)
            rospy.sleep(0.5)



class MovingAverageFilter:
    """
    '현재 조향해야 할 각'과 '차선의 이전 프레임에서의 중앙값'을 저장할 이동평균 필터 클래스
    """

    def __init__(self, n):
        self.n = n  # 큐 사이즈
        self.queue = []

    def add_data(self, x):
        if len(self.queue) >= self.n:
            self.queue = self.queue[1:] + [x]
        else:
            self.queue.append(x)

    def get_data(self):
        return sum(self.queue)/len(self.queue)

def pi2angle(cte):
    return 50 / 180 * cte

def pixel_to_angle(pixel):
    """
    -320 ~ 320의 픽셀로 들어오는 값을 -50 ~ +50의 서보모터 값으로 바꾸는 함수
    mapping 관계를 이용함
    """
    # input_max = 320, input_min = -320
    # output_max = 50, output_min = -50

    angle = (pixel - (-320)) * (50 - (-50)) / (320 - (-320)) + (-50)
    return angle * 2    # 값이 너무 미세하게 변해 2배 처리

def angle_to_pixel(angle):
    """
    -50 ~ +50의 서보모터 값을 0 ~ 640의 픽셀로 바꾸는 함수
    mapping 관계를 이용함
    """

    angle = (angle - (-50)) * (640 - (0)) / (50 - (-50)) + (0)
    return angle


def main():
    rospy.init_node('trt_drive', anonymous=False)
    rate = rospy.Rate(10)

    detect = Detect()   # 객체를 탐지하는 클래스
    drive = Drive()     # 모터의 조향각과 속도를 제어하는 클래스

    angle_maf = MovingAverageFilter(5)  # 조향각에 사용할 이동평균필터
    prev_mid_maf = MovingAverageFilter(3)   # 이전 차선 중간값(제어목표)를 저장할 이동평균필터

    obj_name = "None"   # 탐지한 객체의 클래스

    rospy.sleep(1) # 노드 연결될 때까지 잠시 대기
    print("\n===== START =====\n")

    while not rospy.is_shutdown():
        detect.select_object()  # 가장 믿을만한 객체 선택

        lane_mid_pos = detect.detect_lane() # 차선을 탐지해 차선의 중간 지점(조향해야 하는 기준)을 계산
        if lane_mid_pos == -1:
            # lane_mid_pos = detect.prev_mid  # lane_mid_pos가 -1이면 예외처리
            lane_mid_pos = 320
            print("lane mid -1")

        # 차선 중앙 ( lane_mid_pos , roi 높이 + roi 크기/2, 빨간색)
        cv2.circle(detect.image, (int(lane_mid_pos), int(detect.roi_x + detect.roi_h//2)),4,(0,0,255),2)    
        # 화면 중앙 ( 화면 중앙, roi 높이 + roi 크기/2, 청록?색)
        cv2.circle(detect.image, (320, int(detect.roi_x + detect.roi_h//2)),4,(200,200,0),2)

        ### Stanley 제어를 사용한 버전
        # st = stanley()
        # cte = st.control(lane_mid_pos)
        # fixed_angle = pi2angle(cte)
        ### 제어기를 사용하지 않은 버전
        error_pixel = lane_mid_pos - 320
        error_angle = pixel_to_angle(error_pixel)   # pixel(cam frame) -> angle(servo)

        ### 이동평균필터로 최종 조향각 결정
        angle_maf.add_data(error_angle)
        target_angle = angle_maf.get_data()

        ### 트랙바 값 변경 시 변수를 조정
        min_pos_gap = cv2.getTrackbarPos('min_pos_gap','trackbar')
        min_box_area = cv2.getTrackbarPos('min_box_area','trackbar')
        min_probability = cv2.getTrackbarPos('min_probability','trackbar')
        light_color_distinguish = cv2.getTrackbarPos('light_color_distinguish','trackbar')
        min_mid_to_b = cv2.getTrackbarPos('min_mid_to_b','trackbar')
        line_thick = cv2.getTrackbarPos('line_thick','trackbar')
        detect.set_param(min_pos_gap, min_box_area, min_probability, light_color_distinguish, min_mid_to_b, line_thick)

        ### 탐지된 표지판에 따라 주행 모드 결정
        if detect.can_trust:        # True이면 표지판 잡은 것
            if detect.obj_id == 0:
                ### 좌회전
                obj_name = "Left"
                drive.drive_rotate(target_angle, dir=0)
            elif detect.obj_id == 1:
                ###우회전
                obj_name = "Right"
                drive.drive_rotate(target_angle, dir=1)
            elif detect.obj_id == 2:
                ### stop
                obj_name = "Stop"
                drive.drive_stop(detect)
            elif detect.obj_id == 3:
                ### crosswalk
                obj_name = "Crosswalk"
                drive.drive_stop(detect)
            elif detect.obj_id == 4:
                ### u-turn
                pass
            elif detect.obj_id == 5:
                ### traffic light
                obj_name = "Trafficlight"
                detect.traffic_light_color()
                ### 불 색깔 구별
                if detect.light_color == "red":
                    drive.drive_stop(detect)
                else:
                    drive.drive_normal(target_angle)
        else:
            obj_name = "None"
            drive.drive_normal(target_angle)

        #### 현 상황을 출력
        print("Detected Object      {}".format(detect.obj_id))
        if obj_name == "Trafficlight":
            print("                     ({})".format(detect.light_color))

        print("---------------------------------------------")
        print("Prev Mid             {}".format(detect.prev_mid))
        print("Curr Mid             {}".format(lane_mid_pos))
        print("CTE                  {}".format(error_pixel))
        # print("Error Angle          {}".format(error_angle))
        # print("After PID            {}".format(fixed_angle))
        print("After MAF            {}".format(target_angle))

        print("---------------------------------------------")
        print("Drive Mode           {}".format(drive.drive_mode))
        print("Speed                {}".format(drive.motor_msg.speed))
        print("Angle                {}".format(drive.motor_msg.angle))

        print("\n")

        ### 방금 조향했던 내용을 바탕으로 '이전값'을 저장. 이동평균필터를 사용
        prev_mid_maf.add_data(lane_mid_pos)
        detect.prev_mid = prev_mid_maf.get_data()

        detect.prev_id = detect.obj_id

        cv2.imshow('view', detect.image)

        if cv2.waitKey(1) == 27:
            break
        
        rate.sleep()

if __name__ == "__main__":
    main()
