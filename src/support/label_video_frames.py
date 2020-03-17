import os

import cv2
from pascal_voc_writer import Writer


def label_video_frames(video_name):
    video_reader = cv2.VideoCapture('../video/{}.mp4'.format(video_name))

    number_of_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    for i in range(number_of_frames):
        success, frame = video_reader.read()
        if success and i % 12 == 0:
            cv2.imshow("current_frame", frame)
            key = cv2.waitKey(0)

            image_file = '../train/{}-{}'.format(video_name, i)
            absolute_path = os.path.abspath('{}.jpg'.format(image_file))

            writer = Writer(absolute_path, frame_w, frame_h)

            not_start_scene = True
            if key == ord('s'):
                not_start_scene = False
                writer.addObject('scene_start', 0, 0, frame_w, frame_h)
            elif key == ord('q'):
                break

            if not_start_scene:
                writer.addObject('not_scene_start', 0, 0, frame_w, frame_h)

            cv2.imwrite('{}.jpg'.format(image_file), frame)
            writer.save('{}.xml'.format(image_file))


def find_missing_annotations():
    images = []
    annotations = []

    for root, dirs, files in os.walk("../../train/images"):
        for image in files:
            images.append(image.split('.')[0])

    for root, dirs, files in os.walk("../../train/annotations"):
        for annotation in files:
            annotations.append(annotation.split('.')[0])

    for img in images:
        if img not in annotations:
            print(img)
