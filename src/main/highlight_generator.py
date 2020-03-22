import re
from operator import itemgetter

import boto3
import cv2
import pytesseract
from imageai.Detection.Custom import CustomObjectDetection

from src.main.demo_video_writer import init_demo_video_writer, write_to_demo_video, release_demo_video_writer
from src.main.optical_flow import calculate_optical_flow
from src.main.scene_detect import detect_camera_scenes_of_video
from src.main.utils import get_crop_image, generate_unique_id, create_folder_structure, convert_str_time_code_to_seconds, get_image_on_image, remove_black_bars
from src.main.video_utils import split_video_by_config, get_frame_rate, merge_all_clips

global_configs = None
rekognition_client = None
image_ai_detector = None
previous_score = None
previous_wickets = None


def init_video_processing(configs):
    global global_configs, rekognition_client, image_ai_detector
    print("Initiated video processing")
    global_configs = configs
    session = boto3.Session(profile_name='rekognition')
    rekognition_client = session.client('rekognition', region_name='us-east-1')

    image_ai_detector = CustomObjectDetection()
    image_ai_detector.setModelTypeAsYOLOv3()
    image_ai_detector.setModelPath("../model/detection_model-ex-002--loss-0000.363.h5")
    image_ai_detector.setJsonPath("../model/detection_config.json")
    image_ai_detector.loadModel()


def aws_rekognition_get_text_detections(frame):
    is_success, im_buf_arr = cv2.imencode(".jpg", frame)
    if is_success:
        byte_im = im_buf_arr.tobytes()
        return rekognition_client.detect_text(Image={'Bytes': byte_im})['TextDetections']
    else:
        return []


def recognize_text(cropped_roi, method):
    print('Recognize text by {}'.format(method))
    if method == 'aws':
        detected_text = []
        for text in aws_rekognition_get_text_detections(cropped_roi):
            if text['Type'] == 'LINE':
                detected_text.append(text['DetectedText'])

        return ' | '.join(detected_text)
    else:
        return pytesseract.image_to_string(cropped_roi)


def check_for_scoreboard_by_regex(text):
    if global_configs['SCOREBOARD_FORMAT'] == 'SF':
        return re.search(global_configs['SCORE_BOARD_REGEX_SF'], text)
    else:
        return re.search(global_configs['SCORE_BOARD_REGEX_WF'], text)


def get_roi_cropped(frame, frame_w, frame_h):
    roi_location = global_configs['ROI_LOCATION']
    if roi_location == 'bottom':
        distance = int(frame_h * global_configs['ROI_FRACTION'])
        cropped_roi = get_crop_image(frame, 0, frame_h - distance, frame_w, frame_h)
    elif roi_location == 'top':
        distance = int(frame_h * global_configs['ROI_FRACTION'])
        cropped_roi = get_crop_image(frame, 0, 0, frame_w, distance)
    elif roi_location == 'right':
        distance = int(frame_w * global_configs['ROI_FRACTION'])
        cropped_roi = get_crop_image(frame, frame_w - distance, 0, frame_w, frame_h)
    elif roi_location == 'left':
        distance = int(frame_w * global_configs['ROI_FRACTION'])
        cropped_roi = get_crop_image(frame, 0, 0, distance, frame_h)
    else:
        cropped_roi = frame

    return cropped_roi


def find_scoreboard_location(frame, frame_h, frame_w):
    cropped_roi = get_roi_cropped(frame, frame_w, frame_h)
    aws_result = aws_rekognition_get_text_detections(cropped_roi)

    roi_height, roi_width, channels = cropped_roi.shape

    for index, textResult in enumerate(aws_result):
        # Scoreboard should be preceded by country
        if check_for_scoreboard_by_regex(textResult['DetectedText']) and aws_result[index - 1]['DetectedText'].upper() in global_configs['COUNTRY_LIST']:
            detected_bbox = textResult['Geometry']['BoundingBox']
            width = int(detected_bbox['Width'] * roi_width)
            height = int(detected_bbox['Height'] * roi_height)
            left = int(detected_bbox['Left'] * roi_width)
            top = int(detected_bbox['Top'] * roi_height)

            return get_scoreboard_object_from_scoreboard_text(textResult['DetectedText']), (left, top, width, height)

    return None, None


def try_to_recognize_text(cropped_scoreboard):
    text_found = recognize_text(cropped_scoreboard, 'pytessaract')
    if text_found is not None and len(text_found) > 0 and check_for_scoreboard_by_regex(text_found):
        return text_found
    else:
        text_found = recognize_text(cropped_scoreboard, 'aws')
        if text_found is not None and len(text_found) > 0 and check_for_scoreboard_by_regex(text_found):
            return text_found
        else:
            return None


def crop_and_recognize_text_at_location(cropped_roi, location):
    tolerance = 12
    tlx = location[0] - tolerance
    tly = location[1] - tolerance
    brx = location[0] + location[2] + tolerance
    bry = location[1] + location[3] + tolerance

    scoreboard_crop = get_crop_image(cropped_roi.copy(), tlx, tly, brx, bry)
    scoreboard_crop = cv2.bilateralFilter(scoreboard_crop, 9, 80, 80)

    text_found = try_to_recognize_text(scoreboard_crop)
    if text_found:
        cv2.rectangle(cropped_roi, (tlx, tly), (brx, bry), (0, 0, 255), 2)
    return text_found


def read_scoreboard(frame, frame_h, frame_w, scoreboard_locations, previous_scoreboard_location):
    cropped_roi = get_roi_cropped(frame, frame_w, frame_h)

    if previous_scoreboard_location is not None:
        text_found = crop_and_recognize_text_at_location(cropped_roi, previous_scoreboard_location)
        if text_found:
            return get_scoreboard_object_from_scoreboard_text(text_found), previous_scoreboard_location

    for location in scoreboard_locations:
        if location != previous_scoreboard_location:
            text_found = crop_and_recognize_text_at_location(cropped_roi, location)
            if text_found:
                return get_scoreboard_object_from_scoreboard_text(text_found), location

    return None, None


def get_scoreboard_object_from_scoreboard_text(scoreboard_text):
    scoreboard_split = scoreboard_text.split(global_configs['SEPARATOR'])
    if global_configs['SCOREBOARD_FORMAT'] == 'SF':
        current_score = int(scoreboard_split[0])
        current_wickets = int(scoreboard_split[1])
    else:
        current_score = int(scoreboard_split[1])
        current_wickets = int(scoreboard_split[0])

    return {
        "score": current_score,
        "wickets": current_wickets
    }


def get_scoreboard_diff(scoreboard_text_object):
    global previous_score, previous_wickets

    current_score = scoreboard_text_object['score']
    current_wickets = scoreboard_text_object['wickets']

    if previous_score is not None and previous_wickets is not None:
        score_change = current_score - previous_score
        wicket_change = current_wickets - previous_wickets

        diff_object = {
            "score_change": score_change,
            "wicket_change": wicket_change
        }
    else:
        diff_object = {
            "score_change": current_score,
            "wicket_change": current_wickets
        }

    previous_wickets = current_wickets
    previous_score = current_score

    return diff_object


def get_text_from_diff_object(diff_object):
    score_change = diff_object['score_change']
    wicket_change = diff_object['wicket_change']
    if score_change == 4 or score_change == 6 or wicket_change == 1:
        return "It's a Wicket" if wicket_change == 1 else "It's a {}".format(score_change)
    else:
        return None


def is_cricket_scene_start_frame_by_model(frame):
    _, detections = image_ai_detector.detectObjectsFromImage(input_type="array", input_image=frame, output_type="array")
    no_of_detections = len(detections)
    if no_of_detections > 0:
        true_count = len([d for d in detections if d['name'] == 'scene_start'])
        if true_count / no_of_detections > 0.5:
            return True
        elif true_count / no_of_detections == 0.5:
            return sorted(detections, key=itemgetter('percentage_probability'), reverse=True)[0]['name'] == 'scene_start'
        else:
            return False
    else:
        return False


def get_end_frame_of_the_scene(scene_start_frame_number, scenes_list):
    index = scenes_list.index(scene_start_frame_number)
    return scenes_list[index + 1] - 1


def evaluate_camera_scene_type_by_first_frames(camera_scene_start_frames_decisions):
    return camera_scene_start_frames_decisions.count(True) / len(camera_scene_start_frames_decisions) > 0.5


def process_video(video_path, start_position=None, end_position=None):
    video_file_name = (video_path.split('/')[-1]).split('.')[0]

    if start_position is not None:
        start_position = convert_str_time_code_to_seconds(start_position)

    if end_position is not None:
        end_position = convert_str_time_code_to_seconds(end_position)

    video_reader = cv2.VideoCapture(video_path)
    number_of_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_fps = get_frame_rate(video_path)

    start_frame_number = int(start_position * video_fps) if start_position is not None else 0
    end_frame_number = int(end_position * video_fps) if end_position is not None else number_of_frames

    camera_scenes_list = detect_camera_scenes_of_video(video_path, start_frame_number, end_frame_number, video_fps)

    scoreboard_locations = []
    previous_scoreboard_location = None
    previous_diff_text = None

    frame_by_frame_diff = {}
    frame_by_frame_score = {}

    highlight_triggered_positions = []

    camera_scene_types = {}
    camera_scene_start_frames_decisions = []

    current_camera_scene_started_frame_number = 0
    current_camera_scene_end_frame_number = 0
    check_frame_type_until = 0
    current_camera_scene_started_frame = None

    previous_gray_frame = None
    previous_points = None

    # create unique id for the session
    unique_id_for_this_session = generate_unique_id()

    # create folder structure
    create_folder_structure(global_configs['OUTPUT_FOLDER_PATH'], unique_id_for_this_session)

    if global_configs['WRITE_DEMO_VIDEO']:
        demo_video_file_path = '{}/{}/demo/{}_demo_video.mp4'.format(global_configs['OUTPUT_FOLDER_PATH'], unique_id_for_this_session, video_file_name)
        init_demo_video_writer(frame_w, frame_h, int(video_fps * 2 / 3), demo_video_file_path)

    video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)  # set next frame number to read

    for i in range(start_frame_number, end_frame_number):
        success, frame = video_reader.read()
        frame_info = []
        if success:
            try:
                frame_info.append('Session Id: {}'.format(unique_id_for_this_session))
                frame_info.append('Frame Number: {}'.format(i))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # remove black bars
                frame = remove_black_bars(frame)

                # get decision about camera scenes
                if current_camera_scene_started_frame_number < i <= current_camera_scene_end_frame_number:

                    # check type of frames
                    if i < check_frame_type_until:
                        if check_frame_type_until - i > int((check_frame_type_until - current_camera_scene_started_frame_number) * 0.75):
                            frame_decision = is_cricket_scene_start_frame_by_model(frame)
                            frame_info.append('Scene start [Model]: {}'.format('YES' if frame_decision else 'NO'))
                        else:
                            previous_gray_frame, previous_points, frame_decision, frame = calculate_optical_flow(previous_gray_frame, frame, frame_w, frame_h,
                                                                                                                 previous_points)
                            frame_info.append('Scene start [Optical]: {}'.format('YES' if frame_decision else 'NO'))
                        camera_scene_start_frames_decisions.append(frame_decision)

                    elif i == check_frame_type_until:
                        # True if bawling start scene
                        camera_scene_types[current_camera_scene_started_frame_number] = evaluate_camera_scene_type_by_first_frames(
                            camera_scene_start_frames_decisions)
                        print('Get decision about camera scene type at {}'.format(i), camera_scene_start_frames_decisions,
                              camera_scene_types[current_camera_scene_started_frame_number])

                        # reset for next camera scene
                        camera_scene_start_frames_decisions = []
                        previous_gray_frame = None
                        previous_points = None

                else:
                    current_camera_scene_started_frame_number = i
                    current_camera_scene_end_frame_number = get_end_frame_of_the_scene(i, camera_scenes_list)
                    current_camera_scene_started_frame = cv2.resize(frame.copy(), (int(frame_w * 0.2), int(frame_h * 0.2)), interpolation=cv2.INTER_AREA)

                    camera_scene_length = current_camera_scene_end_frame_number - current_camera_scene_started_frame_number - 1
                    check_frame_type_until = i + (camera_scene_length if video_fps > camera_scene_length else video_fps)

                    print('New camera scene started at {}'.format(i), current_camera_scene_end_frame_number, check_frame_type_until)

                # read scoreboard
                scoreboard_data_object, found_location = read_scoreboard(frame, frame_h, frame_w, scoreboard_locations, previous_scoreboard_location)

                if scoreboard_data_object is None:
                    print('Finding new scoreboard location')
                    scoreboard_data_object, found_location = find_scoreboard_location(frame, frame_h, frame_w)
                    if scoreboard_data_object is not None and found_location not in scoreboard_locations:
                        scoreboard_locations.append(found_location)

                previous_scoreboard_location = found_location
                frame_by_frame_score[i] = scoreboard_data_object

                if scoreboard_data_object is None:
                    frame_info.append('Detected scoreboard: {}'.format('NA'))
                    frame_info.append('Highlight event: {}'.format('NA'))
                else:
                    diff_object = get_scoreboard_diff(scoreboard_data_object)
                    frame_by_frame_diff[i] = diff_object

                    frame_info.append('Detected scoreboard: {}'.format('{}-{}'.format(scoreboard_data_object['score'], scoreboard_data_object['wickets'])))
                    diff_text = get_text_from_diff_object(diff_object)

                    if diff_text is not None:  # if there is a highlight event
                        previous_diff_text = {
                            'text': diff_text,
                            'clear': int(i + video_fps)
                        }
                        frame_info.append('Highlight event: {}'.format(previous_diff_text['text']))
                        frame_info.append('Disappear after: {} frame(s)'.format(previous_diff_text['clear'] - i))

                        highlight_triggered_positions.append(i)

                    elif previous_diff_text is not None and previous_diff_text['clear'] >= i:
                        frame_info.append('Highlight event: {}'.format(previous_diff_text['text']))
                        frame_info.append('Disappear after: {} frame(s)'.format(previous_diff_text['clear'] - i))
                    else:
                        frame_info.append('Highlight event: {}'.format('No'))
                        previous_diff_text = None

                show_info_in_frame(frame, frame_h, frame_w, frame_info)

                # Showing current camera scene started frame as thumbnail
                get_image_on_image(frame, current_camera_scene_started_frame, 20, 20)

                if global_configs['DEMO_FULL_SCREEN']:
                    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                cv2.imshow("Frame", frame)

                if global_configs['WRITE_DEMO_VIDEO']:
                    write_to_demo_video(frame)

            except Exception as err:
                print(err)

    if global_configs['WRITE_DEMO_VIDEO']:
        release_demo_video_writer()

    video_reader.release()
    cv2.destroyAllWindows()

    print('camera scene types', camera_scene_types)
    print('highlight triggered positions', highlight_triggered_positions)

    # Generate highlight scene clips
    highlight_scenes_path = '{}/{}/highlight_scenes/{}'.format(global_configs['OUTPUT_FOLDER_PATH'], unique_id_for_this_session, video_file_name)
    written_highlight_clips = split_video_into_highlight_scenes(video_path, camera_scene_types, highlight_triggered_positions, video_fps,
                                                                highlight_scenes_path)

    # Merge all highlight scene clips to build final highlight video
    final_highlight_video_file_path = '{}/{}/final_highlight_video/{}_final_highlight.mp4'.format(global_configs['OUTPUT_FOLDER_PATH'],
                                                                                                  unique_id_for_this_session,
                                                                                                  video_file_name)
    merge_all_clips(written_highlight_clips, final_highlight_video_file_path)
    return unique_id_for_this_session


def show_info_in_frame(frame, frame_h, frame_w, info):
    color = (0, 0, 255)
    left_offset = 20
    info_window_start_point = (left_offset, int(frame_h * 0.3))
    text_x = left_offset + 20
    text_y = info_window_start_point[1] + 50
    for i, line in enumerate(info):
        cv2.putText(frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        text_y += 50

    info_window_width = int(frame_w * 0.5)
    info_window_end_point = (info_window_start_point[0] + info_window_width, text_y)
    cv2.rectangle(frame, info_window_start_point, info_window_end_point, color, 2)


def split_video_into_highlight_scenes(video_path, scene_type, highlight_triggered_positions, fps, output_file_path_prefix):
    config_list = []
    for position in highlight_triggered_positions:
        start_frame_number, end_frame_number = find_best_starting_ending_points(scene_type, position, fps)

        config_list.append({
            "start_time": round(start_frame_number / fps, 2),
            "length": round((end_frame_number - start_frame_number) / fps, 2),
            "rename_to": "{}_cut_{}_{}.mp4".format(output_file_path_prefix, start_frame_number, end_frame_number)
        })

    split_video_by_config(video_path, config_list)
    return [d['rename_to'] for d in config_list]


def find_best_starting_ending_points(scene_type, position, fps):
    starting_point = None
    ending_point = None
    for scene_start_frame_number, is_start_type in scene_type.items():
        if is_start_type:
            diff = position - scene_start_frame_number
            if scene_start_frame_number < position and diff > fps * 3:
                starting_point = scene_start_frame_number
            elif ending_point is None:
                ending_point = scene_start_frame_number
            else:
                break

    if starting_point is None:
        starting_point = 0
    if ending_point is None:
        ending_point = position

    return starting_point, ending_point - 1
