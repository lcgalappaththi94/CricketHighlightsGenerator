import math

import cv2
import numpy as np

MAX_CORNERS = 100
USE_DYNAMIC_POINTS = False

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=MAX_CORNERS,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=10)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def get_good_features_to_track(frame, frame_w, frame_h):
    if USE_DYNAMIC_POINTS:
        mask = np.zeros_like(frame, np.uint8)
        mask = cv2.rectangle(mask, (0, 0), (frame_w, int(frame_h * 0.75)), 255, -1)  # get roi without scoreboard
        return cv2.goodFeaturesToTrack(frame, mask=mask, **feature_params)
    else:
        points = None
        for i in range(int(frame_w * 0.4), int(frame_w * 0.6), 50):
            for j in range(int(frame_h * 0.6), int(frame_h * 0.8), 50):
                if points is None:
                    points = np.array([[i, j]], dtype=np.float32)
                else:
                    points = np.append(points, np.array([[i, j]], dtype=np.float32), axis=0)
        return points.reshape(-1, 1, 2)


def is_movement_up(previous_x, previous_y, current_x, current_y):
    return current_y > previous_y and abs(current_x - previous_x) / abs(current_y - previous_y) < math.sqrt(3)


def calculate_optical_flow(previous_frame_gray, current_frame, frame_w, frame_h, previous_points):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    if previous_frame_gray is not None and previous_points is not None:
        # calculate optical flow
        new_points, status, err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, current_frame_gray, previous_points, None, **lk_params)

        good_old = previous_points[status == 1]
        if new_points is not None:
            # Select good points
            good_new = new_points[status == 1]

            # draw the tracks
            up_movement_count = 0
            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()  # [a,b]
                c, d = old.ravel()  # [c,d]

                cv2.arrowedLine(current_frame, (a, b), (c, d), (0, 0, 255), 2)
                if is_movement_up(c, d, a, b):
                    up_movement_count += 1

            if len(good_new) > 0:
                up_movement = up_movement_count / len(good_new) >= 0.5
            else:
                up_movement = False
        else:
            up_movement = False
            good_new = good_old

        return current_frame_gray.copy(), good_new.reshape(-1, 1, 2), up_movement, current_frame
    else:
        return current_frame_gray.copy(), get_good_features_to_track(current_frame_gray, frame_w, frame_h), False, current_frame
