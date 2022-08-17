"""
1) Command: python stitcher.py path-to-directory containing two images
2) Uncomment line 39 to resize the image, to make it fit within the window.
"""

# Code for implementation of image mosaicing using OpenCV
import cv2
import argparse
import os


def opencv_stitching(path_to_images):
    filenames = os.listdir(path_to_images)
    images = []
    for i in filenames:
        images.append(cv2.imread(os.path.join(path_to_images, i)))

    stitcher = cv2.createStitcher(False)
    (status, result) = stitcher.stitch(images)
    if status == 0:
        output = result
    else:
        return print("Stitching failed")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(" ", nargs="+", type=str, help="position arguments here refer to the string containing the path"
                                                       "of"
                                                       "the folder containing the two images")

    args = vars(parser.parse_args())
    path = args[" "][0]

    result = opencv_stitching(path)
    while True:
        # result = cv2.resize(result, (result.shape[1]//2, result.shape[0]//2))
        cv2.imshow("Cropped Stitched Image using Stitcher in OpenCV ", result)
        cv2.moveWindow("Cropped Stitched Image using Stitcher in OpenCV", 20, 20)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
