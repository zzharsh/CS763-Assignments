"""
Instructions to run the code smoothly
1) Make sure the images are added to the directory path
2) Command for standard testing: python manual.py ../data/campus -normalize (0 or 1)
3) For testing on custom dataset, Command python manual.py ../data/custom -normalize (0 or 1)
4) For selecting the key-points manually, left click on the point where you wish to select (select atleast 4 points)
"""

# Importing modules
import argparse
import cv2
import os
import numpy as np
from crop import cropify


# Function for selecting points in image 1
def get_points_1(event, x, y, flags, param):
    global points_1
    global count_1
    if event == cv2.EVENT_LBUTTONDOWN:
        points_1.append((x, y))
        cv2.circle(I1, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        count_1 += 1
        cv2.putText(I1, str(count_1), (x-10, y+10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.8, color=(0, 0, 255))
        cv2.imshow("Image 1", I1)


# Function for selecting points in image 2
def get_points_2(event, x, y, flags, param):
    global points_2
    global count_2
    if event == cv2.EVENT_LBUTTONDOWN:
        points_2.append((x, y))
        cv2.circle(I2, (x, y), radius=3, color=(255, 0, 0), thickness=-1)
        count_2 += 1
        cv2.putText(I2, str(count_2), (x - 10, y + 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.8, color=(255, 0, 0))
        cv2.imshow("Image 2", I2)


# Function for getting the normalization matrix for keypoints
def normalize_matrix(x):
    mean_x, mean_y, _ = np.mean(x, axis=1)
    std_x, std_y, _ = np.std(x, axis=1)

    # ----- matrix D = |1/sx      0       -mx/sx |----- #
    # -----------------|0         1/sy    -my/sy |----- #
    # -----------------|0         0            1 |----- #

    D = np.array([[(1/std_x),  0,         (-mean_x/std_x)],
                  [0,          (1/std_y), (-mean_y/std_y)],
                  [0,          0,                      1]])

    return D


def corrected_homography(homo_matrix, dst_img, src_img):
    h, w = dst_img.shape[0], dst_img.shape[1]

    # ------ Want to see where the corners are translated ----- #
    corner_matrix = np.array([[0, w-1, w-1, 0],
                              [0, 0, h-1, h-1],
                              [1, 1, 1, 1]])

    corners_now = np.dot(homo_matrix, corner_matrix)

    # ----- Convert into standardized form ------ #
    [x, y, z] = corners_now
    x = np.divide(x, z)
    y = np.divide(y, z)

    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    new_width = max_x
    new_height = max_y

    correction = [0, 0]
    if min_x < 0:
        new_width -= min_x
        correction[0] = abs(min_x)

    if min_y < 0:
        new_height -= min_y
        correction[1] = abs(min_y)

    if new_width < src_img.shape[1] + correction[0]:
        new_width = src_img.shape[1] + correction[0]
    if new_height < src_img.shape[0] + correction[1]:
        new_height = src_img.shape[0] + correction[1]

    # Finding the coordinates of the corners of the image if they all were within the frame.
    x = np.add(x, correction[0])
    y = np.add(y, correction[1])

    old_initial_points = np.float32([[0, 0],
                                   [w - 1, 0],
                                   [w - 1, h - 1],
                                   [0, h - 1]])
    new_final_points = np.float32(np.array([x, y]).transpose())

    # Updating the homography matrix. Done so that now the secondary image completely
    # lies inside the frame
    homography_matrix = cv2.getPerspectiveTransform(old_initial_points, new_final_points)

    return [new_height, new_width], correction, homography_matrix


def error_rate(image_1, image_2):
    # Converting the images into Grayscale for ease
    i1_gray = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
    i2_gray = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)

    i_difference = i1_gray - i2_gray
    flatten = np.reshape(i_difference, (i_difference.size,1))
    error_val = flatten.T @ flatten/(i_difference.shape[0]*i_difference.shape[1])
    return error_val

# Argument parser for taking command line arguments
parser = argparse.ArgumentParser()

# Argument for taking in the path to the image directory
parser.add_argument(" ", nargs="+", type=str, help="position arguments here refer to the string containing the path of "
                                                   "the folder containing the two images")

# Argument for keypoint normalization
parser.add_argument("-normalize", "--normalize", type=int, required=True, help="0 for no normalization, 1 for "
                                                                               "normalization")
args = vars(parser.parse_args())

# Collecting the arguments
path = args[" "][0]
normalize_bool = args["normalize"]

if normalize_bool not in [0, 1]:
    print("Normalization boolean is not set to either of 0 or 1")
    exit()


# List of images in the image directory
image_name_list = os.listdir(path)

# Reading the images
I1 = cv2.imread(os.path.join(path, image_name_list[0]))
I2 = cv2.imread(os.path.join(path, image_name_list[1]))

# creating unannotated copies for stitching
I1_unannotated = np.copy(I1)
I2_unannotated = np.copy(I2)

# # Initializing the array for collecting the key-points in the image
points_1 = []
points_2 = []

# # Counter for updating the selected key-point index in the displayed image
count_1 = 0
count_2 = 0

# Collecting the keypoint indices and displaying the selected key-point along with the index
while 1:
    cv2.imshow("Image 1", I1)
    cv2.moveWindow("Image 1", 20, 20)
    cv2.setMouseCallback("Image 1", get_points_1)
    cv2.imshow("Image 2", I2)
    cv2.moveWindow("Image 2", 20 + I1.shape[1], 20)
    cv2.setMouseCallback("Image 2", get_points_2)
    k = cv2.waitKey(0)
    if k == ord('q'):
        # print("Points in Image 1 : ", points_1, "\nPoints in Image 2 : ", points_2)
        break
cv2.destroyAllWindows()


# Normalization
if normalize_bool == 1:
    # Converting points 1 and points 2 into homogenous coordinates
    points_1 = [list(np.append(i, [1], axis=0)) for i in points_1]
    points_2 = [list(np.append(i, [1], axis=0)) for i in points_2]

    kp1 = np.array(points_1).T
    kp2 = np.array(points_2).T

    D1 = normalize_matrix(kp1)  # Getting the transformation matrix for src points
    D2 = normalize_matrix(kp2)  # Getting the transformation matrix for dst points

    kp1 = np.matmul(D1, kp1)
    kp2 = np.matmul(D2, kp2)

    Hn, _ = cv2.findHomography(kp1.T, kp2.T)

    # ------ H = inv(D2).(Hn.D1) ------ #

    H = np.matmul(np.linalg.inv(D2), np.matmul(Hn, D1))
else:
    H, mask = cv2.findHomography(np.array(list(points_1)), np.array(list(points_2)))

dimensions, correction, H_corrected = corrected_homography(H, I1_unannotated, I2_unannotated)
# Cropping the image
final_image, stitched_image = cropify(I2_unannotated, I1_unannotated, dimensions, H_corrected, correction)

# Displaying the stitched image
while True:
    cv2.imshow("Cropped Stitched Image", final_image)
    cv2.moveWindow("Cropped Stitched Image", 20, 20)
    cv2.imshow("Stitched Image", stitched_image)
    cv2.moveWindow("Stitched Image", 20 + final_image.shape[0], 20)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
cv2.destroyAllWindows()


# Computing the error rate between the normalized and non normalized case
# Compute this by setting the normalize parameter as 1
if normalize_bool == 1:
    H_normalized = H
    H_actual,_ = cv2.findHomography(np.array(list(points_1)), np.array(list(points_2)))
    dimensions, correction, H_corrected_norm = corrected_homography(H_normalized, I1_unannotated, I2_unannotated)
    I1_norm_transformed = cv2.warpPerspective(I1_unannotated, H_corrected_norm, (dimensions[1], dimensions[0]))
    _, _, H_corrected_actual = corrected_homography(H_actual, I1_unannotated, I2_unannotated)
    I1_actual_transformed = cv2.warpPerspective(I1_unannotated, H_corrected_actual, (dimensions[1], dimensions[0]))




    # dimensions, correction, H_corrected_norm = corrected_homography(H_normalized, I1_unannotated, I2_unannotated)
    #
    # final_image_norm, stitched_image_norm = cropify(I2_unannotated, I1_unannotated, dimensions, H_corrected, correction)
    #
    # dimensions, correction, H_corrected_actual = corrected_homography(H_actual, I1_unannotated, I2_unannotated)
    # # Cropping the image
    # final_image_non_norm, stitched_image_non_norm = cropify(I2_unannotated, I1_unannotated, dimensions, H_corrected_actual, correction)

    error_stitched = error_rate(I1_actual_transformed, I1_norm_transformed)
    print("Percentage error between the normalized and non normalized case is", error_stitched[0][0]*100)
    # 0.0047


# Reporting Error Metric for difference in normalized and non-normalized case
# Loading the saved stitched images with I1 as source





