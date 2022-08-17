import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-mat', type = str)
parse = parser.parse_args()

# Read Image
img = cv2.imread('../data/distorted.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
zeros_arr = np.zeros_like(gray)
zeros_arr[np.where(gray>4)] = 1.0

# Get Corner Points
dst = cv2.cornerHarris(zeros_arr,2,3,0.15)
#result is dilated formarking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
new_image = np.copy(img)
# Threshholding the corners
new_image[dst>0.05*dst.max()]=[0,0,255]
new_dst = dst*(dst>0.01*dst.max())
np.where(dst>0.01*dst.max())
temp = np.copy(img)
temp[np.where(dst>0.01*dst.max())] = [0,0,255]
# Final three points
pts1 = np.float32([[598,61],
            [660,660],
            [61,598]])

pts2 = np.float32([[700,0],
            [700,700],
            [0,700]])
#1. Manual
if parse.mat=='manual':
    arr1 = pts2.T
    arr2 = pts1.T
    oness = np.ones((1,3))
    arr1 = np.concatenate((arr1,oness), axis = 0)
    arr2 = np.concatenate((arr2,oness), axis = 0)

    mat = np.matmul(arr1, np.linalg.inv(arr2))
    udst = cv2.warpAffine(img, mat[:2], (img.shape[0], img.shape[1]))
    cv2.imwrite('../results/undistorted_manual.jpg',udst)
    cv2.imshow('freame', udst)
    cv2.waitKey(0)

#2. API
elif parse.mat=='api':
    mat = cv2.getAffineTransform(pts1, pts2)
    udst = cv2.warpAffine(img, mat, (img.shape[0], img.shape[1]))
    cv2.imwrite('../results/undistorted_api.jpg',udst)
    cv2.imshow('freame', udst)
    cv2.waitKey(0)
    print(mat)

#3. For image in (c): Black and White ChessBoard
# Read Image
img = cv2.imread('../data/figc.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
zeros_arr = np.zeros_like(gray)
zeros_arr[np.where(gray<235)] = 1.0
# Get Corner Points
dst = cv2.cornerHarris(zeros_arr,2,3,0.05)
#result is dilated formarking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
new_image = np.copy(img)
# Threshholding the corners
new_image[dst>0.05*dst.max()]=[0,0,255]
new_dst = dst*(dst>0.01*dst.max())
np.where(dst>0.01*dst.max())
temp = np.copy(img)
temp[np.where(dst>0.01*dst.max())] = [0,0,255]

cv2.imwrite('../results/fig_c_corners.jpg', new_image)

# Final three points
pts1 = np.float32([[218,651],
       [219,133],
       [576, 33],
       #576,763]
        ])

pts2 = np.float32([[0,800],
       [0,0],
       [800,0],
       #[800,800]
        ])
#1. Manual
if parse.mat=='manual':
    arr1 = pts2.T
    arr2 = pts1.T
    oness = np.ones((1,3))
    arr1 = np.concatenate((arr1,oness), axis = 0)
    arr2 = np.concatenate((arr2,oness), axis = 0)

    mat = np.matmul(arr1, np.linalg.inv(arr2))
    udst = cv2.warpAffine(img, mat[:2], (img.shape[0], img.shape[1]))
    cv2.imwrite('../results/undistorted_manual.jpg',udst)
    cv2.imshow('freame', udst)
    cv2.waitKey(0)

#2. API
elif parse.mat=='api':
    mat = cv2.getAffineTransform(pts1, pts2)
    udst = cv2.warpAffine(img, mat, (img.shape[0], img.shape[1]))
    cv2.imwrite('../results/undistorted_api_4.jpg',udst)
    cv2.imshow('freame', udst)
    cv2.waitKey(0)
    

