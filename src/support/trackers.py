import cv2

tracker = None
CORRELATION_TRACKER = 'csrt'  # OPENCV_OBJECT_TRACKERS or dlib
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create(),
    "kcf": cv2.TrackerKCF_create(),
    "boosting": cv2.TrackerBoosting_create(),
    "mil": cv2.TrackerMIL_create(),
    "tld": cv2.TrackerTLD_create(),
    "medianflow": cv2.TrackerMedianFlow_create(),
    "mosse": cv2.TrackerMOSSE_create()
}


def init_tracker(frame):
    global tracker
    if CORRELATION_TRACKER == 'dlib':
        tracker = dlib.correlation_tracker()
        roi = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
        roi = dlib.rectangle(roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
        tracker.start_track(frame, roi)

    else:
        tracker = OPENCV_OBJECT_TRACKERS[CORRELATION_TRACKER]
        roi = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
        tracker.init(frame, roi)


def get_tracker_box(frame):
    if CORRELATION_TRACKER == 'dlib':
        tracker.update(frame)
        pos = tracker.get_position()
        return int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()), 'Found'

    else:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            return x, y, x + w, y + h, 'Found'
        else:
            print('Failed')
            return 0, 0, 0, 0, 'Not Found'
