import os
import uuid

import cv2


def get_crop_image(original, tlx, tly, brx, bry):
    return original[tly:bry, tlx:brx]


def get_image_on_image(original, crop, start_x, start_y):
    rows, cols, channels = crop.shape
    roi = original[start_y:start_y + rows, start_x:start_x + cols]
    img2gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(crop, crop, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    original[start_y:start_y + rows, start_x:start_x + cols] = dst
    cv2.rectangle(original, (start_x - 2, start_y - 2), (start_x + cols + 2, start_y + rows + 2), (0, 0, 255), 3)
    return original


def generate_unique_id():
    return str(uuid.uuid4())[:8]


def create_folder_structure(output_folder_path, session_id):
    if not os.path.exists('{}/{}'.format(output_folder_path, session_id)):
        os.makedirs('{}/{}'.format(output_folder_path, session_id))
        os.makedirs('{}/{}/demo'.format(output_folder_path, session_id))
        os.makedirs('{}/{}/highlight_scenes'.format(output_folder_path, session_id))
        os.makedirs('{}/{}/final_highlight_video'.format(output_folder_path, session_id))


def convert_str_time_code_to_seconds(time_code):  # time_code format should be 00:00:00
    time_code_split = time_code.split(':')
    return int(int(time_code_split[0]) * 3600 + int(time_code_split[1]) * 60 + int(time_code_split[2]))
