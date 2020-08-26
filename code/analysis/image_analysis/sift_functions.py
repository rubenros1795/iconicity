import cv2 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from skimage import color
import pandas as pd
import glob
from numpy.linalg import norm
from shapely import geometry

def sifter(img1,img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h = img1.shape[0]
        w = img1.shape[1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst_target = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst_target)],True,255,3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),10))
        
    #plt.imshow(img3, 'gray'),plt.show()
    return [tuple(x[0]) for x in src_pts], [tuple(x[0]) for x in dst_target]

def surface_target(img,dst):
    poly = geometry.Polygon(dst)
    area = poly.area
    s = round(area) / (img.shape[0] * img.shape[1])
    if s > 1:
        s = 1
    return s

def surface_source(img,dst):
    min_y = min(dst, key = lambda t: t[1])[1]
    max_y = max(dst, key = lambda t: t[1])[1]
    min_x = min(dst, key = lambda t: t[0])[0]
    max_x = max(dst, key = lambda t: t[0])[0]

    width = max_x - min_x
    height = max_y - min_y
    s = round(width * height) / (img.shape[0] * img.shape[1])
    if s > 1:
        s = 1
    return [min_x,max_y,max_x,min_y],s