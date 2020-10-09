import os
import numpy as np
import cv2
import time
import os
import tensorflow as tf
import h5py
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from MidPoint import findMiddlePoint,detect_Intersect,bird_view
from car_controll import PID,Scale_Angle,PID_speed
import argparse
IMAGE_SIZE = 256

def mask_parse(mask):
	mask = np.squeeze(mask)
	mask = [mask, mask, mask]
	mask = np.transpose(mask, (1, 2, 0))
	return mask

def read_image(x):
	#x = cv2.imread(path, cv2.IMREAD_COLOR)
	x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
	x = x/255.0
	return x

def read_mask(path):
	x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
	x = np.expand_dims(x, axis=-1)
	x = x/255.0
	return x

def bird_view( source_img, isBridge=False):
	pts1=np.float32([[0,85],[300,85],[0, 200], [300, 200]  ])
	pts2=np.float32([[0,0],[200,0],[200-130,300],[150,300]])
	matrix=cv2.getPerspectiveTransform(pts1,pts2)
	bird_view= cv2.warpPerspective(source_img,matrix,(240,350))
	return bird_view


def model():
	inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
	
	encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
	skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
	encoder_output = encoder.get_layer("block_13_expand_relu").output
	
	f = [16, 32, 48, 64]
	x = encoder_output
	for i in range(1, len(skip_connection_names)+1, 1):
		x_skip = encoder.get_layer(skip_connection_names[-i]).output
		x = UpSampling2D((2, 2))(x)
		x = Concatenate()([x, x_skip])
		
		x = Conv2D(f[-i], (3, 3), padding="same")(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)
		
		x = Conv2D(f[-i], (3, 3), padding="same")(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)
		
	x = Conv2D(1, (1, 1), padding="same")(x)
	x = Activation("sigmoid")(x)
	
	models = Model(inputs, x)
	return models



def segmentation(img, models):
	t=time.time()
	x = read_image(img)
	y_pred = models.predict(np.expand_dims(x, axis=0))[0] > 0.8
	y_pred.dtype=np.uint8
	binary=np.zeros_like(y_pred)
	binary[(y_pred==1)]=255
	kernel = np.ones((1,1),np.uint8)
	dilation = cv2.dilate(binary,kernel,iterations = 1)
	intersection=detect_Intersect(dilation)
		#print("image",dilation.shape)
	#cv2.imshow("a",dilation)
	MidPoint=findMiddlePoint(dilation)
	Point=(MidPoint,160)
	
	#x=cv2.circle(x,Point,2,(255,0,0),3)
	#print("middle_pos",MidP_269oint)
	try:  
		angle=PID(MidPoint-130)#131
		angle=Scale_Angle(angle)
		angle=format(angle,'.9f')
	except:
		pass
	
	#print("angle:{}".format(angle))
	#cv2.imshow("visualize",x)
	end =time.time()-t
	print("fps",1/end)
	return angle,intersection


def Run(callback_speed,calback_angle,img, models):
	global speed
	angle,intersection=segmentation(img, models)
	
	speed=0
	if (callback_speed>26):
	 	speed=-4
	elif callback_speed<=16:
		speed=130
	else: 
		speed=30
	if calback_angle>=9.5 or calback_angle<= -9.5:
		return -3,angle
		
	if intersection:
		angle=0
	
	fn_speed=speed
	 
	return fn_speed,angle