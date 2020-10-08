import numpy as np
import cv2
def bird_view( source_img):
    width,height=290,310
    pts1=np.float32([[0,100],[300,100],[0, 200], [300, 200]  ])
    pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    bird_view= cv2.warpPerspective(source_img,matrix,(height,width))
    return bird_view

def findMiddlePoint(image):
    interested_row1 = image[int((image.shape[0] / 2)+21), :].reshape((-1,))
    white_pixels1 = np.argwhere(interested_row1 > 0)
    interested_row2 = image[int((image.shape[0] / 2)+19), :].reshape((-1,))#19,20 map2
    white_pixels2 = np.argwhere(interested_row2 > 0)
    interested_row3 = image[int((image.shape[0] / 2)+45), :].reshape((-1,))#19,20 map2
    white_pixels3 = np.argwhere(interested_row3 > 0)
    interested_row4 = image[int((image.shape[0] / 2)+90), :].reshape((-1,))#19,20 map2
    white_pixels4 = np.argwhere(interested_row4 > 0)
  
    if white_pixels1.size != 0 and white_pixels2.size != 0 :
            middle_pos = (np.mean(white_pixels2)+np.mean(white_pixels1))/2
           
    elif  white_pixels3.size != 0 and white_pixels1.size == 0 and white_pixels2.size == 0:
            print("lay diem gan")
            middle_pos = np.mean(white_pixels3)
    else:
         print("diem cuc gan")
         middle_pos = np.mean(white_pixels4)
    middle_pos=middle_pos.astype(int)
       
    return middle_pos
def detect_Intersect(image):
    isIntersect=False
    _,contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    best = -1
    maxsize = -1
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > maxsize :
            maxsize = cv2.contourArea(cnt)
            best = count
        count = count + 1
    
    best_cnt=contours[best]
    size_best=cv2.contourArea(best_cnt)
    if (size_best/1000)>26.3:
        print("intersect_here")
        isIntersect=True
    else:
        isIntersect=False
    return isIntersect


