# Code for ransac
import numpy as np
import cv2
import math


def compute_N(w, p):
    N = math.log10(1 - p)/math.log10(1 - w**4)
    return int(N)


def ransac(srcPoints, dstPoints, threshold=1.0):
    p = 0.99
    w_init = 0.5
    N_init = compute_N(w_init, p)
    print("Starting RANSAC with initial N : ", N_init)
    N_updated = N_init
    prev_inpercent = 100
    best_inliers = []
    second_best_inliers = []
    count_1 = 0
    count_2 = 0
    for i in range(N_init):

        print("Iteration number : ", i)

        # ------ Select 4 Random points ----- #
        random_indices = np.random.choice(np.arange(0, len(srcPoints)), size=4, replace=False)
        kp1 = srcPoints[random_indices]
        kp2 = dstPoints[random_indices]

        kp1_left = srcPoints[np.delete(np.arange(0, len(srcPoints)), random_indices)]
        kp2_left = dstPoints[np.delete(np.arange(0, len(srcPoints)), random_indices)]

        # ------ Compute Homography matrix ----- #
        h, _ = cv2.findHomography(kp1, kp2)

        # ------ Compute Inliers ------ #
        #print([np.linalg.norm(pd.T - np.matmul(h, ps.T)/np.matmul(h, ps.T)[2]) for ps, pd in zip(kp1_left, kp2_left)])
        inliers_indices = [True if (np.linalg.norm(pd.T - np.matmul(h, ps.T)/np.matmul(h, ps.T)[2]) < threshold) else False for ps, pd in zip(kp1_left, kp2_left)]
        #print(inliers_indices)
        inliers_src = kp1_left[inliers_indices]
        inliers_dst = kp2_left[inliers_indices]

        # ----- Percentage of Inliers ----- #
        in_percent = 100*inliers_src.shape[0]/srcPoints.shape[0]
        out_percent = 100 - in_percent
        print("Outliers (%) : ", out_percent)

        if in_percent > w_init*100:
            best_inliers = inliers_src, inliers_dst
            w_init = in_percent/100
            N_updated = compute_N(w_init, p)
            print("Best inlier percentage upto now : ", in_percent)
            print("Updated N : ", N_updated)
            count_1 += 1

        if in_percent < prev_inpercent:
            prev_inpercent = in_percent
            second_best_inliers = inliers_src, inliers_dst
            count_2 += 1

        if i > N_updated:
            break
    # ----- Save the best model corresponding to a particular random selection ----- #
    if best_inliers:
        print("Total point correspondances fed : ", len(srcPoints))
        print("Inliers detected : ", len(best_inliers[0]))
        print("Number of times T is used : ", count_1*100/N_init)
        return best_inliers
    else:
        print("Total point correspondances fed : ", len(srcPoints))
        print("Inliers detected (second best since inliers are not more than 50% for a given threshold) : ", len(second_best_inliers[0]))
        print("Number of times T is used : ", count_2 * 100 / N_init)
        return second_best_inliers







