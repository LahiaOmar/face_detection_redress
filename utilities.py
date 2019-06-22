import numpy as np
import cv2 as cv


def rect_to_bb(rect):
    """ Transforme rectangle to bounding box

    Parameters :
    rect : dlib rect

    Returns :
    tuple (x, y, w, h) : coordinates of bounding box
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype='int'):
    """ Convert a dlib full_object_detection to numpy array

    Paramaeters:
    shape : array of dlib full_object_detection
    dtype : type of array 

    Returns :
    coords : numpy array od coordinates landmark point
    """

    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
