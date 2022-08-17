import numpy as np
import cv2


def error_rate(image_1, image_2):
    # Converting the images into Grayscale for ease
    i1_gray = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
    i2_gray = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)

    i_difference = i1_gray - i2_gray
    flatten = np.reshape(i_difference, (i_difference.size,1))
    error_val = np.dot(flatten, flatten)/(i_difference.shape[0]*i_difference.shape[1])
    return error_val


if __name__=="__main__":
    path = "manual_saved"
    I1 = cv2.imread("manual_saved/manual_stitched_norm0.jpg")
    I2 = cv2.imread("manual_saved/manual_stitched_norm1.jpg")
    print("Error rate between the normalized and non-normalized case is : ", error_rate(I1, I2))

