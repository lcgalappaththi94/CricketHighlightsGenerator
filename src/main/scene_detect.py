from __future__ import print_function

from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager


def detect_camera_scenes_of_video(video_file, start, end, framerate):
    scenes_start_frames_list = []

    video_manager = VideoManager([video_file], framerate=framerate)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector(threshold=20))
    base_time_code = video_manager.get_base_timecode()

    try:
        start_time = base_time_code + start
        end_time = base_time_code + end
        print('start time: {}'.format(start_time), 'end time: {}'.format(end_time))
        # Set video_manager duration to read frames
        video_manager.set_duration(start_time=start_time, end_time=end_time)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()  # downscale factor is computed automatically based on the current video’s resolution

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_time_code)
        # Like FrameTimecodes, each scene in the scene_list can be sorted if the
        # list of scenes becomes unsorted.

        for i, scene in enumerate(scene_list):
            scenes_start_frames_list.append(scene[0].get_frames())

    finally:
        video_manager.release()

    return scenes_start_frames_list
