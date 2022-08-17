import cv2
import numpy as np
import os
import argparse

'''
Instructions: 
- load the database specified in the question
- kindly add the pictures of your team members in database_image named as low.jpg, middle.jpg and high.jpg
- In replicate/capturedImages, kindly add the masked images of your team members. The naming of the pictures should be
  as follows: 
  a) The reference masked image : maskLow.jpg
  b) Middle Intruder images : maskedMiddle?.jpg
'''


def fast_detector(source, target, n=50):
    orb = cv2.ORB_create(n)
    source_keypoint = orb.detect(source, None)
    target_keypoint = orb.detect(target, None)
    return source_keypoint, target_keypoint


def dog_detector(source, target, n=50):
    sift = cv2.xfeatures2d.SIFT_create(n)
    source_keypoint = sift.detect(source, None)
    target_keypoint = sift.detect(target, None)
    return source_keypoint, target_keypoint


def sift_descriptor(source, source_keypoint, target, target_keypoint, n=50):
    sift = cv2.xfeatures2d.SIFT_create(n)
    source_descriptor = sift.compute(source, source_keypoint)[1]
    target_descriptor = sift.compute(target, target_keypoint)[1]
    return source_descriptor, target_descriptor


def brief_descriptor(source, source_keypoint, target, target_keypoint, n=50):
    orb = cv2.ORB_create(n)
    source_descriptor = orb.compute(source, source_keypoint)[1]
    target_descriptor = orb.compute(target, target_keypoint)[1]
    return source_descriptor, target_descriptor


def brute_force_distance(source_descriptor, target_descriptor):
    brute_force_obj = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    num_matches = brute_force_obj.match(source_descriptor, target_descriptor)
    num_matches = sorted(num_matches, key=lambda x: x.distance)
    kp_distances = list(map(lambda x: x.distance, num_matches[0:50]))
    average_distance = np.mean(kp_distances / (np.max(kp_distances) + 1e-10))
    return average_distance


def retrieval_scores(masked_image, detector, descriptor):
    function_call_dict = {

        "fast": fast_detector,
        "dog": dog_detector,
        "sift": sift_descriptor,
        "brief": brief_descriptor
    }
    database_files = os.listdir("../replicate/database_image/")
    image_kp_distances = {}
    for i in database_files:
        s = cv2.imread(os.path.join(r"..\replicate\database_image", i))
        t = masked_image
        kp1, kp2 = function_call_dict[detector](s, t)
        des1, des2 = function_call_dict[descriptor](s, kp1, t, kp2)
        distance = brute_force_distance(des1, des2)
        image_kp_distances[os.path.join("../replicate/database_image", i)] = 1 - distance  # Adding the similairty index
    return image_kp_distances


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    kp_list = ["fast", "dog"]
    des_list = ["sift", "brief"]
    trans_list = ["view", "scale", "rot", "light"]

    parser.add_argument("-i", "--i", help='''Should be path of "maskLow.jpg"''', required=True)
    parser.add_argument("-j", "--j", help='''Should be path of "mask_Middle?.jpg"''', required=True)

    args = parser.parse_args()

    masked_low_path = args.i
    masked_middle_path = args.j

    # Loading maskLow and maskMiddle images
    masked_low = cv2.imread(masked_low_path)
    masked_middle = cv2.imread(masked_middle_path)

    scores = retrieval_scores(masked_low, "dog", "sift")

    reference = scores[os.path.join("../replicate/database_image", "low.jpg")]
    print("Reference similarity index : ", format(reference, ".3f"))

    scores_with_middle = retrieval_scores(masked_middle, "dog", "sift")

    potential_intruders = {}
    for i in scores_with_middle.keys():
        if scores_with_middle[i] > reference:
            potential_intruders[i] = scores_with_middle[i]
    print("Number of Potential Intruders : ", len(potential_intruders.keys()))

    if os.path.join("../replicate/database_image", "middle.jpg") in list(potential_intruders.keys()):
        print("Intruder in database")
    else:
        print("Intruder not in database")

    # reading lines in retScores.txt
    retScores_file = open('../replicate/output/retScores.txt', 'r+')
    lines = retScores_file.readlines()
    if len(lines) - 1 > 0:
        scores_dictionary = {}
        for i in range(1, len(lines)):
            val = lines[i].split(":")[1]
            scores_dictionary[lines[i].split(":")[0]] = float(val.split("\n")[0])
        if scores_with_middle[os.path.join("../replicate/database_image", "middle.jpg")] > np.max(
                list(scores_dictionary.values())):
            cv2.imwrite("../replicate/output/maskMiddleBest.jpg", masked_middle)

    string = str(args.j).split("Images/")[1] + " : " + str(
        format(scores_with_middle[os.path.join("../replicate/database_image", "middle.jpg")], ".3f")) + "\n"
    retScores_file.write(string)
    retScores_file.close()
