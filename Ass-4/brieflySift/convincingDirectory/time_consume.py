import time
import cv2
import os
from features_modified import fast_detector, dog_detector, sift_descriptor, brief_descriptor, brute_force

cases = {
    1 : ["fast","brief","scale"],
    2 : ["fast","brief","rot"],
    3 : ["fast","brief","view"],
    4 : ["fast","brief","light"],
    5: ["fast", "sift", "scale"],
    6: ["fast", "sift", "rot"],
    7: ["fast", "sift", "view"],
    8: ["fast", "sift", "light"],
    9: ["dog", "brief", "scale"],
    10: ["dog", "brief", "rot"],
    11: ["dog", "brief", "view"],
    12: ["dog", "brief", "light"],
    13: ["dog", "sift", "scale"],
    14: ["dog", "sift", "rot"],
    15: ["dog", "sift", "view"],
    16: ["dog", "sift", "light"],
}

function_call_dict = {
    "fast": fast_detector,
    "dog": dog_detector,
    "sift": sift_descriptor,
    "brief": brief_descriptor
}

time_list = []
for case in range(1, 17):
    source_img_name = cases[case][2] + "_S.ppm"
    target_img_name = cases[case][2] + "_T.ppm"
    N = 50

    S = cv2.imread(os.path.join(r"../data", source_img_name))
    T = cv2.imread(os.path.join(r"../data", target_img_name))

    t1 = time.time()
    for _ in range(0,10):
        kp1, kp2 = function_call_dict[cases[case][0]](S, T, 300)

        des1, des2 = function_call_dict[cases[case][1]](S, kp1, T, kp2)

        output = brute_force(S, kp1, des1, T, kp2, des2, N)
    t2 = time.time()
    print("time for case  : ", str(case)," ", (t2-t1)*100, " ms")

    output_name = "_".join(cases[case]) + ".png"
    output_path = os.path.join(r"time_images", output_name)
    cv2.imwrite(output_path, output)
    time_list.append((t2-t1)*100)
