import numpy as np
import cv2


def cropify(image_src, image_dst, dimensions, H_corrected, correction):
    warped = cv2.warpPerspective(image_dst, H_corrected, (dimensions[1], dimensions[0]))
    im_mask = 255 * np.ones_like(image_dst)
    warped_mask = cv2.warpPerspective(im_mask, H_corrected, (dimensions[1], dimensions[0]))

    # Additional code for cropping the images
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

    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    right_min_x = int(round(min(x[1:3])))
    left_max_x = int(round(max([x[0], x[3]])))
    top_max_y = int(round(max(y[0:2])))
    bottom_min_y = int(round(min([y[2], y[3]])))

    final_image = np.zeros_like(warped)
    final_image[min_y:max_y, min_x:max_x] = warped[min_y:max_y, min_x:max_x]
    final_image[correction[1]:correction[1] + image_src.shape[0],
    correction[0]: correction[0] + image_src.shape[1]] = image_src

    stitched_image = final_image

    final_image = final_image[max(top_max_y, correction[1]):min(bottom_min_y, correction[1] + image_src.shape[0]),
                      min(correction[0], left_max_x):max(right_min_x, correction[0] + image_src.shape[1])]

    return final_image, stitched_image


def crop(img, img_white):
    img_white = np.mean(img_white, axis=2)
    x1 = np.argmin(img_white[0])
    x2 = np.argmin(img_white[-1])
    return img[:, :min(x1, x2)]


def mast_crop(I1, I2, H):
    # Warping the source image to get the transformation
    Img = cv2.warpPerspective(I1, H, (I2.shape[1] + I1.shape[1], I2.shape[0] + I1.shape[0]))

    # Stitching the warped image along with the original image
    Img[0:I1.shape[0], 0:I1.shape[1]] = I1
    Img[0:I2.shape[0], 0:I2.shape[1]] = I2
    #
    # Creating a mask for cropping the image out
    I1_white = 255 * np.ones_like(I1)
    I2_white = 255 * np.ones_like(I2)

    # Warping the masks
    Img_white = cv2.warpPerspective(I1_white, H, (I1.shape[1] + I2.shape[1], I1.shape[0] + I2.shape[0]))

    # Stitching the masks
    Img_white[0:I1.shape[0], 0:I1.shape[1]] = I1_white
    Img_white[0:I2.shape[0], 0:I2.shape[1]] = I2_white

    img_white = np.mean(Img_white, axis=2)
    x1 = np.argmin(img_white[0])
    x2 = np.argmin(img_white[-1])
    y1 = np.argmin(img_white[:, 0])
    y2 = np.argmin(img_white[:, -1])

    return Img[:max(y1,y2), :max(x1, x2)], Img

