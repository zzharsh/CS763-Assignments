import cv2
import os
import argparse

'''
Instructions: 
- To check for the command line arguments required, use command : python real.py -h
- Default case command : python real.py -trans light
- Outputs are stored in the results/created folder. Kindly refer to it for verification of results.
'''


def surf_detector(source, target, hT=5000):
    surf = cv2.xfeatures2d.SURF_create(hT, extended=True, upright=False)
    source_keypoint, source_descriptor = surf.detectAndCompute(source, None)
    target_keypoint, target_descriptor = surf.detectAndCompute(target, None)
    return source_keypoint, source_descriptor, target_keypoint, target_descriptor


def brute_force(source, source_keypoint, source_descriptor, target, target_keypoint, target_descriptor, nf):
    brute_force_obj = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    num_matches = brute_force_obj.match(source_descriptor, target_descriptor)
    num_matches = sorted(num_matches, key=lambda x: x.distance)[0:nf]
    output_image = cv2.drawMatches(source, source_keypoint, target, target_keypoint, num_matches, None, flags=2)
    return output_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    trans_list = ["view", "scale", "rot", "light"]

    parser.add_argument("-hT", "--hT", help='''Changing the hessian threshold. If not entered, it will take 5000 by 
    default. This will detect around 1000 points''', default = 5000)
    parser.add_argument("-trans", "--trans", help='''Should be one of "view","scale","rot" or "light" ''',
                        required=True)
    parser.add_argument("-nm", "--nm", help='''approximate number of points to be matched, default is 50. 
                        Should be less than 1000 ''', default=50, type=int)

    args = parser.parse_args()
    CLI_flag = (args.trans in trans_list) and (0 < args.nm < 1000)
    if not CLI_flag:
        print(CLI_flag)
        raise Exception("Invalid Command arguments : Command line arguments do not match the description. Enter again")

    source_img_name = args.trans + "_S.ppm"
    target_img_name = args.trans + "_T.ppm"
    N = args.nm

    S = cv2.imread(os.path.join(r"../data/created", source_img_name))
    T = cv2.imread(os.path.join(r"../data/created", target_img_name))

    kp1, des1, kp2, des2 = surf_detector(S, T)
    output = brute_force(S, kp1, des1, T, kp2, des2, N)

    output_name = "surf_surf" + "_" + args.trans + "_" + str(args.nm) + ".png"
    output_path = os.path.join(r"../results/created", output_name)
    cv2.imwrite(output_path, output)