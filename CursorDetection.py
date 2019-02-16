# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:17:46 2018

@author: Himanshu Garg
UBName : hgarg
UB Person : 50292195
"""
import cv2
import numpy as np
#import time

#p = time.time()

def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

kernel_size = 3
itr = 10
maxVal = 0
maxLoc = []
temp = []
res = []
ims = ["pos_","t1_","t2_","t3_","neg_"]
for name in ims:
    for i in range(15):
        if name == "t1_":
            template = cv2.imread("input\\t1.png",0)
        elif name == "t2_":
            template = cv2.imread("input\\t2.png",0)
        elif name == "t3_":
            template = cv2.imread("input\\t3.png",0)
        elif name == "neg_":
            template = cv2.imread("input\\t3.png",0)
        elif name == "pos_":
            template = cv2.imread("input\\t3.png",0)
        w, h = template.shape[::-1]
        path = name + str(i+1) + ".jpg"
        wpath = name + str(i+1) + "_detect_result.png"
        im = cv2.imread('input\\'+path,1)
        if im is None:
            continue
        threshold = 0.645
        # threshold for bonus images can be taken as 0.6 for better result
        copyim = im.copy()
        templ1 = template.copy()
        templ2 = template.copy()
        ddepth = cv2.CV_8U
        blur1 = cv2.GaussianBlur(copyim,(3,3),0)
        blur1 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        lpOrgImg = cv2.Laplacian(blur1, ddepth, kernel_size)
        lpOrgImg = cv2.convertScaleAbs(lpOrgImg)
        
        for i in range(itr):
            lpTempImg = cv2.Laplacian(templ1, ddepth, kernel_size)
            lpTempImg = cv2.convertScaleAbs(lpTempImg)
            result = cv2.matchTemplate(lpOrgImg,lpTempImg,cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val > maxVal:
                    maxVal = max_val
                    maxLoc = max_loc
                    temp = templ1.copy()
                    w, h = templ1.shape[::-1]
                    res = result
            upscale = 125  
            width = int(templ1.shape[1] * upscale / 100)
            height = int(templ1.shape[0] * upscale / 100)
            dim = (width, height)
            templ1 = cv2.resize(templ1, dim,interpolation = cv2.INTER_AREA)
            
        loc = np.where( res >= threshold)
        
        if len(loc[0]) == 0 and len(loc[1]) == 0 :
            for i in range(2):
                lpTempImg = cv2.Laplacian(templ2, ddepth, kernel_size)
                lpTempImg = cv2.convertScaleAbs(lpTempImg)
                result = cv2.matchTemplate(lpOrgImg,lpTempImg,cv2.TM_CCORR_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val > maxVal:
                        maxVal = max_val
                        maxLoc = max_loc
                        temp = templ2.copy()
                        w, h = templ2.shape[::-1]
                        res = result
                downscale = 96        
                width = int(templ2.shape[1] * downscale / 100)
                height = int(templ2.shape[0] * downscale / 100)
                dim = (width, height)
                templ2 = cv2.resize(templ2, dim,interpolation = cv2.INTER_AREA)
        
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(im, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        maxVal = 0
        maxLoc = []
        temp = []
        #cv2.imwrite(wpath,im)
        display(path,im)
