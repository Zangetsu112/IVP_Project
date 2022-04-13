import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def extract_splice(gray_image):
    """
        Extracts segments from images so they can be spliced
        or copy pasted into other images
    """
    _, thresh = cv2.threshold(gray_image, 160, 210, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE)[-2], key=cv2.contourArea, reverse=True)
    for contour in cnt:
        if cv2.contourArea(contour) < 2000:
            cnt = contour
            break
    mask = np.zeros(gray_image.shape, np.uint8)
    mask = cv2.drawContours(mask, [cnt], -1, (255), -1, cv2.LINE_AA)
    splice = cv2.bitwise_and(gray_image, mask)
    return splice


def random_rotation_and_scaling(splice, width, height):
    """ Rotates the splice to any random angle and scales it (max 1.5)"""
    factor = random.random()
    scale = 1.5 if (factor > 0.5) else 1 + factor
    rotation = random.randrange(0, 360)
    rotation_matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2), rotation, scale)
    splice = cv2.warpAffine(splice, rotation_matrix, (width, height))
    return splice


def check_overlap(mask, image):
    """
        Helper function to other forgery functions to check for overlap
        between image and mask
    """
    nonzero_count = np.count_nonzero(mask)
    _, thresh = cv2.threshold(image, image.mean(), 250, cv2.THRESH_BINARY)
    overlap = cv2.bitwise_and(mask, thresh)
    if (np.count_nonzero(overlap) < nonzero_count // 3):
        return (False, None)
    return (True, overlap)


def new_image(mask, image, splice):
    """ Applies the chosen splice to the image """
    mask = cv2.bitwise_not(mask)
    new_image = cv2.bitwise_and(image, mask)
    new_image = new_image + splice
    return new_image


def add_splice(image, all_splice):
    """ Assumes image: gray_image same for splices"""
    for iter in range(len(all_splice)):
        splice = random.choice(all_splice)
        splice = random_rotation_and_scaling(splice, \
                        image.shape[1], image.shape[0])
        _, mask = cv2.threshold(splice, 130, 250, cv2.THRESH_BINARY)
        splice = cv2.bitwise_and(splice, mask)
        overlap_result = check_overlap(mask, image)
        if not (overlap_result[0]):
            continue
        plt.imshow(overlap_result[1], cmap='gray')
        plt.show()
        return new_image(overlap_result[1], image, splice)
    return np.zeros(image.shape)


def copy_paste(image, splice):
    """ Assumes image: gray_image same for splices"""
    for iter in range(3):
        splice = random_rotation_and_scaling(splice, \
                                    image.shape[1], image.shape[0])
        dx = random.randrange(-200, 200)
        dy = random.randrange(-200, 200)
        shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        splice = cv2.warpAffine(splice, shift_matrix, \
                                (splice.shape[1], splice.shape[0]))
        _, mask = cv2.threshold(splice, 130, 255, cv2.THRESH_BINARY)
        overlap_result = check_overlap(mask, image)
        if not (overlap_result[0]):
            continue
        return new_image(overlap_result[1], image, splice)
    return np.zeros(image.shape)
