import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import cv2
import argparse
from PIL import Image
from flask import Flask
from io import BytesIO
#------------- Add library ------------#
from sematicSeg import Run, model

#--------------------------------------#
#Global variable
MAX_SPEED = 30
MAX_ANGLE = 25
# Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED
MIN_SPEED = 10

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)

global model

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    global model
    if data:
        steering_angle = 0  #Góc lái hiện tại của xe
        speed_callback = 0           #Vận tốc hiện tại của xe
        image = 0           #Ảnh gốc

        steering_angle = float(data["steering_angle"])
        speed_callback = float(data["speed"])
        #Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        
        sendBack_angle = 0
        sendBack_Speed = 0
        try:
            #------------------------------------------  Work space  ----------------------------------------------#
            print(image.shape)
            #image=bird_view(image)
            
            #out.write(image)
            
            sendBack_Speed,sendBack_angle=Run(speed_callback,steering_angle,image, model)

            #cv2.waitKey(1)
            
            #------------------------------------------------------------------------------------------------------#
            print('{} : {}'.format(sendBack_angle, sendBack_Speed))
            send_control(sendBack_angle, sendBack_Speed)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def region_of_interest (image):
    height=image.shape[0]
    width= image.shape[1]
    channel_count=image.shape[2]
    polygons = np.array([
        [(0,height),(width,height),(width,int(height/3)),(0,int(height/3))]
        ])
    mask=np.zeros_like(image)
    mask_color = (255,)*channel_count
    cv2.fillPoly(mask,polygons,mask_color)
    masked_image= cv2.bitwise_and(image,mask)
    return masked_image

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)


if __name__ == '__main__':
    
    #-----------------------------------  Setup  ------------------------------------------#
    global model
    argparser = argparse.ArgumentParser(
        description='Run Road Segmentation Model on a video')
    argparser.add_argument(
        'model',
        type=str,
        help='path to model file')
    args = argparser.parse_args()

    model = model()
    model.load_weights(args.model)

    #--------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    
