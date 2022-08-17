import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#1. Images are uploaded in data directory
parser = argparse.ArgumentParser()
parser.add_argument('v', type = str)
parse = parser.parse_args()
image_path = os.listdir(parse.v)
images = []
#2. Read all the images into a numpy array:
for path in image_path:
    images.append(cv2.imread(parse.v+'/'+ path))

#3. Display the first image in window:
i = 0
img = images[i]
cv2.imshow("display_images",img)
#4. Keep showing the images:
print("Press 'n' for next image, 'p' for previous image and 'esc' for exit")
while True:
    k = cv2.waitKey(0)
    if k==27:
        break
    elif k == ord('n'):
        i+=1
        i = i%len(images)
        img = images[i]
        cv2.imshow("display_images",img)
    elif k == ord('p'):
        i-=1
        i=i%len(images)
        img = images[i]
        cv2.imshow("display_images",img)
    
