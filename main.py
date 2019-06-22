import cv2 as cv
import numpy as np
import dlib
import math
from utilities import rect_to_bb, shape_to_np


def get_rectangle_face(frame):
    """ get ractangle of faces position with landmark point

    Parameters :
    frame : image / frame

    returns : 
    tuple (x_min, x_max, y_min, y_max) : coordinates of face
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rects = detector(gray)
    x_min, y_min, x_max, y_max = (999999, 999999, -1, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for i in range(len(shape)):
            x_min = min(x_min, shape[i][0])
            x_max = max(x_max, shape[i][0])

            y_max = max(y_max, shape[i][1])
            y_min = min(y_min, shape[i][1])
    return (x_min, x_max, y_min, y_max)


global path_predictor,  detector, predictor
# path of landmark file
path_predictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_predictor)
cap = cv.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rects = detector(gray)
    print(rects.__class__)
    angle = -1
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(rect)
        x_left = 0
        y_left = 0
        x_right = 0
        y_right = 0

        # calculate middle value of landmark eyes position
        for i in range(36, 42):
            x_left += shape[i][0]
            y_left += shape[i][1]
        for i in range(42, 48):
            x_right += shape[i][0]
            y_right += shape[i][1]

        if y_right <= y_left:

            cv.line(frame, (int(x_left / 6), int(y_left / 6)),
                    (int(x_right / 6), int(y_right / 6)), (0, 0, 255))
            # horizontal line
            cv.line(frame, (int(x_left / 6), int(y_left / 6)),
                    (int(x_right / 6), int(y_left / 6)), (0, 255, 0))
            # vectors
            vect_1 = (x_right - x_left, y_right - y_left)
            vect_2 = (x_right - x_left, y_left - y_left)

            vect_1_len = math.sqrt(vect_1[0]*vect_1[0] + vect_1[1]*vect_1[1])
            vect_2_len = math.sqrt(vect_2[0]*vect_2[0] + vect_2[1]*vect_2[1])

            vect_prod = vect_1[0] * vect_2[0] + vect_1[1] * vect_2[1]

            cons_angle = vect_prod / (vect_1_len * vect_2_len)
            arc_cos = math.acos(cons_angle)
            angle = 0
            if int(arc_cos*10) > 0:
                angle = -arc_cos*100 + 8
            print("angle : ", arc_cos)

        else:
            cv.line(frame, (int(x_left / 6), int(y_left / 6)),
                    (int(x_right / 6), int(y_right / 6)), (255, 0, 0))

            # horozontal line
            cv.line(frame, (int(x_left/6), int(y_right / 6)),
                    (int(x_right / 6), int(y_right / 6)), (0, 0, 0))

            vect_1 = (x_right - x_left, y_right - y_left)
            vect_2 = (x_right - x_left, y_right - y_right)

            vect_1_len = math.sqrt(vect_1[0]*vect_1[0] + vect_1[1]*vect_1[1])
            vect_2_len = math.sqrt(vect_2[0]*vect_2[0] + vect_2[1]*vect_2[1])

            vect_prod = vect_1[0] * vect_2[0] + vect_1[1] * vect_2[1]

            cons_angle = vect_prod / (vect_1_len * vect_2_len)
            arc_cos = math.acos(cons_angle)
            angle = 0
            if int(arc_cos*10) > 0:
                angle = arc_cos*100 - 8
            print("angle : ", arc_cos)

    M = cv.getRotationMatrix2D(
        (frame.shape[0] / 2, frame.shape[1] / 2), angle, 1.0)
    rt = cv.warpAffine(frame, M, (frame.shape[0], frame.shape[1]))
    x_min, x_max, y_min, y_max = get_rectangle_face(rt)
    rt = cv.cvtColor(rt, cv.COLOR_BGR2GRAY)
    crop = rt[y_min:y_max, x_min:x_max]
    if crop.shape[0] != 0 and crop.shape[1] != 0:
        print("shape crop ", crop.shape)
        cv.imshow("crop", crop)
    cv.imshow("frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
