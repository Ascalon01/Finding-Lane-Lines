"""
Created on Wed Jun 19 14:48:25 2019

@author: Ascalon
"""
import numpy as np
import cv2
import collections
import glob
import os.path

flagvideo=1   
previous_frames = collections.deque([], 5)
rho=2
theta=np.pi/180
threshold=50
min_line_len=15
max_line_gap=10

def hsv(img):
    
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow1 = np.array([ 0,120, 120], dtype=np.uint8)
    yellow2 = np.array([ 100,255, 255,], dtype=np.uint8)
    yellow = cv2.inRange(img, yellow1, yellow2)
    white1 = np.array([0, 0, 200], dtype=np.uint8)
    white2 = np.array([255, 30, 255], dtype=np.uint8)
    white = cv2.inRange(img1, white1, white2)
    out=cv2.bitwise_and(img, img, mask=(yellow | white))
    return out
    
def canny(img, low_threshold, high_threshold):
    
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
   
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)     
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def find_lanes(img, lines, color=[0, 255, 0], thickness=8):
    difference = 0.1
    leftSlopeAvg = 0
    pointsLeft = []
    l = 1

    rightSlopeAvg = 0
    pointsRight = []
    r = 1
    
    yy=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            dx=x2-x1
            if dx!=0:
                slope = (y2-y1)/dx
                if slope < 0:
                    rightSlopeAvg = rightSlopeAvg + (slope - rightSlopeAvg) / r
                    if np.absolute(rightSlopeAvg - slope) < difference:
                        pointsRight.append((x1, y1))
                        pointsRight.append((x2, y2))
                        yy.append(y1)
                        yy.append(y2)
                    r += 1
                else:
                    leftSlopeAvg = leftSlopeAvg + (slope - leftSlopeAvg) / l
                    if np.absolute(leftSlopeAvg - slope) < difference:
                        pointsLeft.append((x1, y1))
                        pointsLeft.append((x2, y2))
                        yy.append(y1)
                        yy.append(y2)
                    l += 1
         

    if len(pointsRight) > 0 and len(pointsLeft) > 0:
        # right lane
        [vx, vy, x, y] = cv2.fitLine(np.array(pointsRight, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
        rightSlope = vy / vx
        rightIntercept = y - (rightSlope * x)

        # left lane
        [vx, vy, x, y] = cv2.fitLine(np.array(pointsLeft, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
        leftSlope = vy / vx
        leftIntercept = y - (leftSlope * x)
    
        previous_frames.append((rightIntercept, rightSlope, leftIntercept, leftSlope))
    
    try:
        if len(previous_frames) > 0:
            avg = np.sum(previous_frames, -3) / len(previous_frames)
            rightIntercept = avg[0]
            rightSlope = avg[1]
            leftIntercept = avg[2]
            leftSlope = avg[3]
        
        startY = max(yy)
        endY = int(img.shape[0]/1.6) 
        
        rightX1 = (startY - rightIntercept) / rightSlope
        rightX2 = (endY - rightIntercept) / rightSlope   
        leftX1 = (startY - leftIntercept) / leftSlope
        leftX2 = (endY - leftIntercept) / leftSlope

        cv2.line(img, (rightX1, startY), (rightX2, endY), color, thickness)
        cv2.line(img, (leftX1, startY), (leftX2, endY), color, thickness)
        
    except Exception  as e:
        
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    find_lanes(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=1, β=1, γ=0):
 
    return cv2.addWeighted(initial_img, α, img, β, γ)
    

if flagvideo:
    
    files=glob.glob(os.path.join('test_videos/','*.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    for i in files:
        tmpname=str.split(i,'\\')
        tmpname=str.split(tmpname[1],'.')
        cap = cv2.VideoCapture(i)
        width = int(cap.get(3) )
        height = int(cap.get(4))
        midx=width/2
        maxt=height/2+30
        vertices = np.array([[(30,height),(midx-30, maxt), (midx+30, maxt), (width-30,height)]], dtype=np.int32)
        video = cv2.VideoWriter('output_videos/Annotated_'+tmpname[0]+'.mp4', fourcc, cap.get(5), (width,height))
        while True:
            ret,image=cap.read()
            if ret:
                hsvimg=hsv(image)
                blur=gaussian_blur(hsvimg,3)
                edge=canny(blur,60,80)
                roiimg=region_of_interest(edge,vertices)
                himg=hough_lines(roiimg, rho, theta, threshold, min_line_len, max_line_gap)
                out = weighted_img(himg, image, 1, 0.8, 0) 
                cv2.imshow('Output',out)
                video.write(out) 
                k=cv2.waitKey(1)
                if k==ord('q'):
                    break
            else:
                break
        video.release()
        cap.release()
    cv2.destroyAllWindows()
    
else:
    
    files=glob.glob(os.path.join('test_images/','*.jpg'))
    for i in files:
        tmpname=str.split(i,'\\')
        tmpname=str.split(tmpname[1],'.')
        img=cv2.imread(i)
        width = img.shape[1]
        height = img.shape[0]
        midx=width/2
        maxt=height/2+30
        vertices = np.array([[(30,height),(midx-30, maxt), (midx+30, maxt), (width-30,height)]], dtype=np.int32)
        hsvimg=hsv(img)
        blur=gaussian_blur(hsvimg,3)
        edge=canny(blur,80,160)
        roiimg=region_of_interest(edge,vertices)
        himg=hough_lines(roiimg, rho, theta, threshold, min_line_len, max_line_gap)
        out = weighted_img(himg, img, 1, 0.8, 0) 
        cv2.imwrite('output_images/Annotated_'+tmpname[0]+'.jpg',out)
        cv2.destroyAllWindows()


