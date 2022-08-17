import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('v', type = str)
parse = parser.parse_args()
#1. Read the input
img = cv2.imread(parse.v)
img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#2. Normalize the pixel value
img_norm = img1/255;
#3. Saving the images
plt.figure()
plt.imshow(img1)
plt.title('Original Image')
plt.savefig('../results/original_image.png')
plt.figure()
plt.imshow(img_norm)
plt.title('Normalized Image')
plt.savefig('../results/normalized_image.png')
#plt.show()
# 4. For displaying the image please uncomment the below three lines
#cv2.imshow("Original Image",img)
#cv2.imshow("Normalized Image", img_norm)
#cv2.waitKey(0)
cv2.imwrite("../results/original_image_cv.png", img)
cv2.imwrite("../results/normalized_image_cv.png", (255*img_norm).astype('uint8'))
 
