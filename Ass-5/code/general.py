"""
Instructions for running the code smoothly
1) Command for standard testing: python general.py ../data/campus -normalize (0 or 1) -mode (auto-ransac or custom-ransac)
                                -idx (Index of reference)
   Here,
    'auto-ransac' will use the inbuilt RaNsac implementation within OpenCV.
    'custom-ransac' will use the ransac function from ransac.py
2) Note: If 'custom-ransac' throws error in findHomography function, please re-run the code, this is because there are
insuffiecient number of inliers to compute the homography matrix. This is due to the randomness of ransac for the particular image
"""


import numpy as np
import cv2
import os
from ransac import ransac
import argparse


# Keypoint detector
def dog_detector(source, target, n=50):
    sift = cv2.xfeatures2d.SIFT_create(n)
    source_keypoint = sift.detect(source, None)
    target_keypoint = sift.detect(target, None)
    return source_keypoint, target_keypoint


# Keypoint descriptor
def sift_descriptor(source, source_keypoint, target, target_keypoint):
    sift = cv2.xfeatures2d.SIFT_create()
    source_descriptor = sift.compute(source, source_keypoint)[1]
    target_descriptor = sift.compute(target, target_keypoint)[1]
    return source_descriptor, target_descriptor


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


# Obtaining the homography matrix for two images
def get_homography(I1, I2, normalize_bool, mode_bool):
    # Total number of key-points to be detected in the image
    n_total = 800

    # Number of best matches to be selected from those keypoints
    n_best = 50
    kp1, kp2 = dog_detector(I1, I2, n_total)
    des1, des2 = sift_descriptor(I1, kp1, I2, kp2)

    # Using the brute force matcher
    brute_force_obj = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    total_matches = brute_force_obj.match(des1, des2)

    # Filtering out the best key-points based on the L2- norm distance
    best_matches = sorted(total_matches, key=lambda x: x.distance)[0:n_best]

    # Indices corresponding to the best key-points in source and target
    query_indices = [x.queryIdx for x in best_matches]
    target_indices = [x.trainIdx for x in best_matches]

    # Key-point coordinate corresponding to the indices
    kp1_best = [kp1[i] for i in query_indices]
    kp2_best = [kp2[i] for i in target_indices]

    # Generating the set to be used for homography
    points1 = [x.pt for x in kp1_best]
    points2 = [x.pt for x in kp2_best]
    arr_pt1 = np.array(points1)
    arr_pt2 = np.array(points2)
    arr_pt1 = arr_pt1.astype('int32')
    arr_pt2 = arr_pt2.astype('int32')

    # Using the custom ransac
    if mode_bool == "custom-ransac":
        # implementing custom ransac based automatic image mosaicing
        pt1_homo = [list(np.append(i, [1], axis=0)) for i in arr_pt1]
        pt2_homo = [list(np.append(i, [1], axis=0)) for i in arr_pt2]

        kp1_ransac_homo, kp2_ransac_homo = ransac(np.array(pt1_homo), np.array(pt2_homo), threshold=5.0)
        kp1_ransac = [list(i)[0:len(i) - 1] for i in kp1_ransac_homo]
        kp2_ransac = [list(i)[0:len(i) - 1] for i in kp2_ransac_homo]

        # Normalized key-points
        if normalize_bool == 1:
            points_1 = kp1_ransac
            points_2 = kp2_ransac

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
            H, mask = cv2.findHomography(np.array(kp1_ransac), np.array(kp2_ransac))
        return H
    # Using OpenCV's integrated ransac
    elif mode_bool == "auto-ransac":
        # implementing OpenCV's ransac based automatic image mosaicing
        if normalize_bool == 1:

            points_1 = arr_pt1
            points_2 = arr_pt2

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
            H, mask = cv2.findHomography(arr_pt1, arr_pt2, method=cv2.RANSAC)
        return H


# Function for correcting the homography matrix to translate the warped image within the frame
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


# Function for stitching the image
def stitch(image_src, image_dst, normalize_bool, mode):
    H = get_homography(image_dst, image_src, normalize_bool=normalize_bool,
                       mode_bool=mode)  # images[4] (destination image)
    dimensions, correction, H_corrected = corrected_homography(H, image_dst, image_src)
    warped = cv2.warpPerspective(image_dst, H_corrected, (dimensions[1], dimensions[0]))
    im_mask = 255 * np.ones_like(image_dst)
    warped_mask = cv2.warpPerspective(im_mask, H_corrected, (dimensions[1], dimensions[0]))

    # Additional code for removing the black tatti
    h, w = image_dst.shape[0], image_dst.shape[1]

    # ------ Want to see where the corners are translated ----- #
    corner_matrix = np.array([[0, w - 1, w - 1, 0],
                              [0, 0, h - 1, h - 1],
                              [1, 1, 1, 1]])

    corners_now = np.dot(H_corrected, corner_matrix)

    # ----- Convert into standardized form ------ #
    [x, y, z] = corners_now
    x = np.divide(x, z)
    y = np.divide(y, z)

    final_image = np.zeros_like(warped)
    final_image[correction[1]:correction[1] + image_src.shape[0],
    correction[0]: correction[0] + image_src.shape[1]] = image_src
    # final_image[min_y:max_y,min_x:max_x] = warped[min_y:max_y,min_x:max_x]
    stitched_image = np.where(warped_mask, warped, final_image)
    # stitched_image = final_image
    return stitched_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(" ", nargs="+", type=str, help="position arguments here refer to the string containing the path"
                                                       "of"
                                                       "the folder containing the two images")
    parser.add_argument("-mode", "--mode", type=str, required=True, help="argument for selecting auto-ransac and"
                                                                         "custom-ransac algorithm")
    parser.add_argument("-idx", "--idx", type=int, required=True, help="argument for selecting the reference image "
                                                                       "from the list of images")
    parser.add_argument("-normalize", "--normalize", type=int, required=True, help="0 for no normalization, 1 for"
                                                                                   "normalization")

    args = vars(parser.parse_args())

    # Collecting all arguments
    path = args[" "][0]
    mode = args["mode"]
    idx = args["idx"]-1
    normalize_bool = args["normalize"]

    if normalize_bool not in [0, 1]:
        print("Normalization boolean is not set to either of 0 or 1")
        exit()

    if mode not in ["auto-ransac", "custom-ransac"]:
        print("Mode should either be auto-ransac or custom-ransac")
        exit()

    filenames = sorted(os.listdir(path), key=lambda x: os.path.getctime(os.path.join(path,x)))

    if len(filenames) <= idx < 0:
        print("Index of reference is not valid")
        exit()

    images = []
    for file in filenames:
        image = cv2.imread(os.path.join(path, file))
        images.append(image)

    reference = images[idx]
    stitched_image = reference.copy()
    try:
        for i in range(len(images)):
            if i != idx:
                stitched_image = stitch(stitched_image, images[i], normalize_bool=normalize_bool, mode=mode)

        final_stitched_image = stitch(stitched_image, reference, normalize_bool=normalize_bool, mode=mode)
        while True:
            output_image = cv2.resize(final_stitched_image,(final_stitched_image.shape[1] // 2, final_stitched_image.shape[0] // 2))
            cv2.imshow("Final Stitched Image", output_image)
            k = cv2.waitKey(0)
            if k == ord('q'):
                break

        # Saving the cropped image
        output_directory = "../results"
        filename = "general_norm_" + str(normalize_bool) + "_" + mode + ".jpg"
        cv2.imwrite(os.path.join(output_directory, filename), output_image)
    except:
        print("ERROR : Please rerun the code.")

