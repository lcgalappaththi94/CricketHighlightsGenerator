from operator import itemgetter

import cv2
from imageai.Detection.Custom import CustomObjectDetection


def test_modal_manually(video_name):
    video_reader = cv2.VideoCapture(video_name)

    number_of_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print(number_of_frames)

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("detection_model-ex-002--loss-0000.363.h5")
    detector.setJsonPath("detection_config.json")
    detector.loadModel()

    for i in range(number_of_frames):
        success, frame = video_reader.read()
        if success:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            detected_image_array, detections = detector.detectObjectsFromImage(input_type="array", input_image=frame, output_type="array")
            best_detection = sorted(detections, key=itemgetter('percentage_probability'), reverse=True)[0]
            points = best_detection["box_points"]

            color = (0, 0, 255) if best_detection["name"] == 'not_scene_start' else (0, 255, 255)
            cv2.putText(frame, '{} [{}]'.format(best_detection["name"], best_detection["percentage_probability"]),
                        (points[0] + 50, points[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            cv2.rectangle(frame, (points[0], points[1]), (points[2], points[3]), color, 2)
            cv2.imshow('Frame', frame)

    video_reader.release()
    cv2.destroyAllWindows()


test_modal_manually('../../video/uploaded.mp4')
