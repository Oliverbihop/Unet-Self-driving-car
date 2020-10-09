import numpy as np
import time

error_arr = np.zeros(5)
error_sp = np.zeros(5)

t=time.time()
dif = time.time()
speed_max=30


def PID(error, p= 0.43, i =0, d = 0.01): #0.43,0,0.02
	    global t
	    error_arr[1:] = error_arr[0:-1]
	    error_arr[0] = error
	    P = error*p
	    delta_t = time.time() -t
	    t = time.time()
	    D = (error-error_arr[1])/delta_t*d
	    I = np.sum(error_arr)*delta_t*i
	    angle = P + I + D
	    if abs(angle)>30:
	    	angle = np.sign(angle)*40
	    return int(angle)


def Scale_Angle(x):
	#return 5/12*x
	return 23/60*x
	#return (1/3)*x
	#return 0.3*x


def PID_speed(error):
	global dif
	p = 0.55
	i = 0.2
	d = 0.01
	error_sp[1:] = error_sp[0:-1]
	error_sp[0] = error
	P = error*p
	delta_t = time.time() - dif
	dif = time.time()
	D = (error-error_sp[1])/delta_t*d
	I = np.sum(error_sp)*delta_t*i
	speed = P + I + D
	if speed > speed_max: 
		speed = speed_max
	if speed < 0: 
		speed = 1
	return speed    