import time
import cv2
import os
from features_modified import fast_detector, dog_detector, sift_descriptor, brief_descriptor, brute_force

cases = {
    1: ["dog", "sift", "scale"],
    2: ["dog", "sift", "rot"],
    3: ["dog", "sift", "view"],
    4: ["dog", "sift", "light"],
}

function_call_dict = {
    "fast": fast_detector,
    "dog": dog_detector,
    "sift": sift_descriptor,
    "brief": brief_descriptor
}

time_list = []
for case in range(1, 5):
    source_img_name = cases[case][2] + "_S.ppm"
    target_img_name = cases[case][2] + "_T.ppm"
    N = 20

    S = cv2.imread(os.path.join(r"../data", source_img_name))
    T = cv2.imread(os.path.join(r"../data", target_img_name))
    kp1, kp2 = function_call_dict[cases[case][0]](S, T, 20)

    des1, des2 = function_call_dict[cases[case][1]](S, kp1, T, kp2)

    output = brute_force(S, kp1, des1, T, kp2, des2, N)
    output_name = "incorrect_" + str(case) + ".png"
    output_path = os.path.join(r"incorrect_images", output_name)
    cv2.imwrite(output_path, output)

