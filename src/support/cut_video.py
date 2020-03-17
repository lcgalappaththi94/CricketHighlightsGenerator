from src.main.video_utils import split_video_by_config

VIDEO_PATH = '../../video'
VIDEO_NAME = '02nd T20 _ West Indies Tour of Sri Lanka 2020'

config_list = [{
    "start_time": 964,
    "length": 1680,
    "rename_to": '{}/{}_cropped.mp4'.format(VIDEO_PATH, VIDEO_NAME)
}]

split_video_by_config('{}/{}.mp4'.format(VIDEO_PATH, VIDEO_NAME), config_list)
