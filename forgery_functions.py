import cv2
import numpy as np
import random


def extract_splice(gray_image, splice_size=2000):
    """
        Extracts segments from images so they can be spliced
        or copy pasted into other images
    """
    _, thresh = cv2.threshold(gray_image, 160, 210, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE)[-2], key=cv2.contourArea, reverse=True)
    chosen_contour = None
    for contour in cnt:
        if cv2.contourArea(contour) in range(1000, splice_size):
            # Test to make sure splice isn't too big
            # Spliced section should't be noise at the edge of X-Ray
            # Test to make sure splice isn't an outline
            moment = cv2.moments(contour)
            cX, cY = int(moment["m10"] / moment["m00"]), \
                        int(moment["m01"] / moment["m00"])
            if cY > 100 and thresh[cX, cY]:    
                chosen_contour = contour
                break
    if type(chosen_contour) == type(None):
        return (False, None)
    mask = cv2.drawContours(np.zeros(gray_image.shape, np.uint8), \
                            [chosen_contour], -1, (255), -1, cv2.LINE_AA)
    splice = cv2.bitwise_and(gray_image, mask)
    return (True, splice)


def random_rotation_and_scaling(splice, width, height):
    """ Rotates the splice to any random angle and scales it (max 1.5)"""
    factor = random.random()
    scale = 0.5 if (factor < 0.5) else factor
    rotation = random.randrange(0, 360)
    rotation_matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2), rotation, scale)
    splice = cv2.warpAffine(splice, rotation_matrix, (width, height))
    scale = 1 / scale 
    scale += 0.5 if (factor > 0.5) else factor
    rotation_matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2), 0, scale)
    splice = cv2.warpAffine(splice, rotation_matrix, (width, height))
    return splice


def check_overlap(mask, image):
    """
        Helper function to other forgery functions to check for overlap
        between image and mask
    """
    nonzero_count = np.count_nonzero(mask)
    _, thresh = cv2.threshold(image, 70, 250, cv2.THRESH_BINARY)
    overlap = cv2.bitwise_and(mask, thresh)
    if (np.count_nonzero(overlap) < nonzero_count // 1.5):
        return (False, None)
    return (True, overlap)


def new_image(mask, image, splice):
    """ Applies the chosen splice to the image """
    mask = cv2.bitwise_not(mask)
    new_image = cv2.bitwise_and(image, mask) 
    new_image = new_image + splice
    return new_image


def add_splice(image, splice, rng):
    """ Assumes image: gray_image same for splices"""
    for iter in range(rng//10):
        splice = random_rotation_and_scaling(splice, \
                                    image.shape[1], image.shape[0])
        dx = random.randrange(-rng, rng)
        dy = random.randrange(-rng, rng)
        shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        splice = cv2.warpAffine(splice, shift_matrix, \
                                (splice.shape[1], splice.shape[0]))
        if np.count_nonzero(splice) < 1000:
            continue
        _, mask = cv2.threshold(splice, 120, 255, cv2.THRESH_BINARY)
        _, splice = cv2.threshold(splice, 120, 255, cv2.THRESH_TOZERO)
        overlap_result = check_overlap(mask, image)
        if not (overlap_result[0]):
            continue
        new_img = new_image(overlap_result[1], image, splice)
        difference = image - new_img
        if np.count_nonzero(difference) < 300: continue
        return (True, new_img)
    return (False, None)
