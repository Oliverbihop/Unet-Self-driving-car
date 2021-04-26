# -*- coding: utf-8 -*-
import base64
import numpy as np
import socketio
import eventlet
# import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO
# ------------- Add library ------------#
# import matplotlib as plt
# import math
import time
import threading
from car_detect import detect_car

# --------------------------------------#
error_arr = np.zeros(5)
dt = time.time()
global temp_angle
is_object=False
# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = 0  # Góc lái hiện tại của xe
        speed_callback = 0  # Vận tốc hiện tại của xe
        image = 0  # Ảnh gốc

        steering_angle = float(data["steering_angle"])
        speed_callback = float(data["speed"])
        # Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        """
        - Chương trình đưa cho bạn 3 giá trị đầu vào:
            * steering_angle: góc lái hiện tại của xe
            * speed: Tốc độ hiện tại của xe
            * image: hình ảnh trả về từ xe

        - Bạn phải dựa vào 3 giá trị đầu vào này để tính toán và gửi lại góc lái và tốc độ xe cho phần mềm mô phỏng:
            * Lệnh điều khiển: send_control(sendBack_angle, sendBack_Speed)
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """
        sendBack_angle = 0
        sendBack_Speed = 30
        # ------------------------------------------  Work space  ----------------------------------------------#
        lane_image = np.copy(image)
        is_object=detect_car(image)
        bird = bird_view(lane_image)
        #bird=filter_colors(bird)
        canny_image = Can(bird)
        cropping_image = ROI(canny_image)
        cv2.imshow("CANNY", canny_image) # vung ROI da duoc xu ly
        cv2.waitKey(1)
        try:
            lines = cv2.HoughLinesP(cropping_image, 3, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=30)#40 20
            averaged_lines,is_Lane = average_slope_intercept(bird, lines)
            line_image, sendBack_angle = display_lines(bird, averaged_lines)

            sendBack_angle = - sendBack_angle
            sendBack_angle=Scale_Angle(sendBack_angle)
            temp_angle=sendBack_angle
            while is_object:
                print("car")
                car_t=time.time()
                sendBack_angle=-25
                sendBack_Speed=150
                time.sleep(0.2)
                #is_object=False
                

            if is_Lane == False :
                sendBack_angle=temp_angle+20
                
                print("matlane/////////////////////////////////////////////////////////")

            combo_image = cv2.addWeighted(bird, 0.8, line_image, 1, 1)
            cv2.imshow("image_line", combo_image)
            cv2.waitKey(1)
            # ------------------------------------------------------------------------------------------------------#
        except Exception as e:
            print(e)
        if speed_callback < 16:
            sendBack_Speed = 140
        elif speed_callback > 23:
            sendBack_Speed = -6

       # print('toc do nap len {} : {}'.format(sendBack_angle, sendBack_Speed))
       # print('van toc tra ve {} : {}'.format(steering_angle, speed_callback))
        send_control(sendBack_angle, sendBack_Speed)

    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)

def Scale_Angle(x):
    return 5/12*x
    #return (1/3)*x
    #return 0.3*x
def Can(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 7)
    #blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blur, 10, 200)
    return canny


def bird_view(image):

    width, height = 300, 320
    pts1 = np.float32([[0, 100], [300, 100], [0, 200], [300, 200]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    birdview = cv2.warpPerspective(image, matrix, (height, width))
    '''
    pts1 = np.float32([[0, 100], [300, 100], [0, 200], [300, 200]])
    pts2 = np.float32([[0, 0], [200, 0], [200 - 140, 300], [170, 300]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    birdview = cv2.warpPerspective(image, matrix, (200, 350))
    '''
    return birdview

def ROI(image):
    height = image.shape[0]
    shape = np.array([[10, height], [450, height], [450, 100], [50, 100]])
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[1]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
def filter_colors(image):
        hsv =cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([179, 150, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_image = cv2.bitwise_and(image, image, mask=white_mask)
        return white_image
def PID(error, p=0.45, i=0.05, d=0.02):
    global dt
    global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error * p
    delta_t = time.time() - dt
    dt = time.time()
    D = (error - error_arr[1]) / delta_t * d
    I = np.sum(error_arr) * delta_t * i
    angle = P + I + D
    if abs(angle) > 35:
        angle = np.sign(angle) * 60
    return -int(angle)


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10, 8)
        xa = int((x1 + x2) / 2 - 110)  # red 110 160
        ya = int((y1 + y2) / 2)
        cv2.circle(line_image, (xa - 20, ya), 5, (0, 255, 0), -1)
        x_d = xa - 150  # green 150
        cv2.circle(line_image, (x_d, ya), 5, (0, 0, 255), -1)
        angle_PID = PID(x_d)
        if angle_PID > 20:
            angle_PID = 15
        elif angle_PID < -20:
            angle_PID = -15
    return line_image, angle_PID


def average_slope_intercept(image, lines):
    is_Lane=True
    right_fit = []
    img_x_center = image.shape[1] / 2
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1==x2:
            continue
        else:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]  # he so a
            intercept = parameters[1]  # he so b
        if slope > 0:
            right_fit.append((slope, intercept))
    # sap xep right_fit theo chieu tang dan cua intercept
    leng = len(right_fit)
    right_fit = np.array(sorted(right_fit, key=lambda a_entry: a_entry[0]))
    right_fit = right_fit[::-1]
    right_fit_focus = right_fit
    if leng > 2:
        right_fit_focus = right_fit[:1]
    elif leng ==0:
        is_Lane=False
    right_fit_average = np.average(right_fit_focus, axis=0)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([right_line]),is_Lane


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


if __name__ == '__main__':
    # -----------------------------------  Setup  ------------------------------------------#

    # --------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

