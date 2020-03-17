import cv2
import numpy as np


def write_demo_video(frame_list, frame_w, frame_h, fps, demo_video_file_path):
    four_cc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(demo_video_file_path, four_cc, fps, (frame_w, frame_h))
    original_video_file_name = (demo_video_file_path.split('/')[-1]).replace('_demo_video', '')

    # write initial video frames
    get_initial_frames_for_summary_video(frame_w, frame_h, video_writer, "This is the demo video\ncreated for\nvideo file: {}".format(
        original_video_file_name))

    # write frames to video
    for frame in frame_list:
        video_writer.write(np.uint8(frame))

    video_writer.release()


def get_initial_frames_for_summary_video(width, height, video_writer, description):
    blank_image = np.zeros((height, width, 3), np.uint8)

    description_box_tl = (int(width * 0.1), int(height * 0.1))
    description_box_br = (int(width * 0.9), int(height * 0.5))
    blank_image = cv2.rectangle(blank_image, description_box_tl, description_box_br, (0, 0, 255), 4)

    font_size, dy, x0, y0 = get_font_size(description, description_box_tl, description_box_br)
    temp_black_prev = blank_image.copy()
    for i, line in enumerate(description.split('\n')):
        y = y0 + (i + 1) * dy
        accumulated_string = ""
        temp_black = temp_black_prev.copy()
        for char in line:
            accumulated_string = accumulated_string + char
            cv2.putText(temp_black, accumulated_string, (x0, y), cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 255), 2)
            video_writer.write(np.uint8(temp_black))
        temp_black_prev = temp_black

    for _ in range(0, 22):
        video_writer.write(np.uint8(temp_black_prev))


def get_font_size(violation_info, description_box_tl, description_box_br):
    margin_horizontal = 0.2
    margin_vertical = 0.2
    font_size_final = 0
    max_length_line = None

    description_box_width = int(description_box_br[0] - description_box_tl[0])
    description_box_height = int(description_box_br[1] - description_box_tl[1])
    dy = int(description_box_height * (1 - margin_horizontal) / len(violation_info.split('\n')))
    x0, y0 = int(description_box_tl[0] + description_box_width * margin_horizontal / 2), int(
        description_box_tl[1] + description_box_height * margin_vertical / 2)

    for line in violation_info.split("\n"):
        if max_length_line is None:
            max_length_line = line
        elif len(max_length_line) < len(line):
            max_length_line = line

    for font_size in np.arange(0.0, 10.0, 0.1):
        size = cv2.getTextSize(max_length_line, cv2.FONT_HERSHEY_DUPLEX, font_size, 2)
        text_width = size[0][0]
        if text_width < description_box_width * (1 - margin_horizontal):
            font_size_final = font_size
        else:
            break

    return font_size_final, dy, x0, y0
