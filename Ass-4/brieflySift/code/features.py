import cv2
import os
import argparse

'''
Instructions: 
 - Kindly, use Python version: 3.7.9
 - Before running this script, please install the required dependencies from requirements.txt (in convincingDirectory)
   in your virtual environment
 - Command for installing from requirements.txt : pip install -r requirements.txt
 - The code for finding out average time for various cases is in the convincingDirectory 
 - The incorrect.png in question 4 of 1.2.1 is inside convincingDirectory/incorrect_images.
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


def sift_descriptor(source, source_keypoint, target, target_keypoint):
    sift = cv2.xfeatures2d.SIFT_create()
    source_descriptor = sift.compute(source, source_keypoint)[1]
    target_descriptor = sift.compute(target, target_keypoint)[1]
    return source_descriptor, target_descriptor


def brief_descriptor(source, source_keypoint, target, target_keypoint):
    orb = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    source_descriptor = orb.compute(source, source_keypoint)[1]
    target_descriptor = orb.compute(target, target_keypoint)[1]
    return source_descriptor, target_descriptor


def brute_force(source, source_keypoint, source_descriptor, target, target_keypoint, target_descriptor, nm):
    brute_force_obj = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    num_matches = brute_force_obj.match(source_descriptor, target_descriptor)
    if nm < len(num_matches):
        num_matches = sorted(num_matches, key=lambda x: x.distance)[0:nm]
        print("Matches entered nm : ", nm, " Matches detected in brute force : ", len(num_matches))
    else:
        print("Number of matches detected : ", len(num_matches))
        print("nm entered : ", nm)
        print("Hence we will show all possible ", len(num_matches), " matches")
    output_image = cv2.drawMatches(source, source_keypoint, target, target_keypoint, num_matches, None, flags=2)
    return output_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    kp_list = ["fast", "dog"]
    des_list = ["sift", "brief"]
    trans_list = ["view", "scale", "rot", "light"]

    function_call_dict = {
        "fast": fast_detector,
        "dog": dog_detector,
        "sift": sift_descriptor,
        "brief": brief_descriptor
    }

    n_points = [300, 1000]

    parser.add_argument("-kp", "--kp", help='''Should be either of "fast" or "dog" ''', required=True)
    parser.add_argument("-des", "--des", help='''Should be either of "sift" or "brief" ''', required=True)
    parser.add_argument("-trans", "--trans", help='''Should be one of "view","scale","rot" or "light" ''',
                        required=True)
    parser.add_argument("-nm", "--nm", help='''Number of points to be matched ''', default=50, type=int)

    args = parser.parse_args()
    CLI_flag = (args.kp in kp_list) and (args.des in des_list) and (args.trans in trans_list) and (args.nm > 0)
    if not CLI_flag:
        print(CLI_flag)
        raise Exception("Invalid Command arguments : Command line arguments do not match the description. Enter again")
    if args.nm > 300:
        raise Exception(
            "Number of matches cannot be more than 300 since the keypoints are 300 and 1000 in two iterations")

    source_img_name = args.trans + "_S.ppm"
    target_img_name = args.trans + "_T.ppm"
    N = args.nm

    S = cv2.imread(os.path.join(r"../data", source_img_name))
    T = cv2.imread(os.path.join(r"../data", target_img_name))

    for np in n_points:
        print("n_points = ", np)
        kp1, kp2 = function_call_dict[args.kp](S, T, np)
        des1, des2 = function_call_dict[args.des](S, kp1, T, kp2)

        output = brute_force(S, kp1, des1, T, kp2, des2, N)

        cv2.imshow("Output (Press q to quit)", output)
        k = cv2.waitKey(0)

        while True:
            if k == ord('q'):
                break
        cv2.destroyAllWindows()

        output_name = args.kp + "_" + args.des + "_" + args.trans + "_" + str(np) + ".png"
        output_path = os.path.join(r"../results", output_name)
        cv2.imwrite(output_path, output)
