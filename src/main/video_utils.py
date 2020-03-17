import os.path
import shlex
import subprocess
import sys

from moviepy.editor import VideoFileClip, concatenate_videoclips


def split_video_by_config(filename, config, vcodec="h264", acodec="copy", extra=""):
    split_cmd = ["ffmpeg", "-i", filename, "-vcodec", vcodec, "-acodec", acodec, "-y"] + shlex.split(extra)
    try:
        file_ext = filename.split(".")[-1]
    except IndexError as e:
        raise IndexError("No . in filename. Error: " + str(e))
    for video_config in config:
        split_args = []
        try:
            split_start = video_config["start_time"]
            split_length = video_config.get("end_time", None)
            if not split_length:
                split_length = video_config["length"]
            file_base = video_config["rename_to"]
            if file_ext in file_base:
                file_base = ".".join(file_base.split(".")[:-1])

            split_args += ["-ss", str(split_start), "-t", str(split_length), file_base + "." + file_ext]
            print("########################################################")
            print("About to run: " + " ".join(split_cmd + split_args))
            print("########################################################")
            subprocess.check_output(split_cmd + split_args)
        except KeyError as e:
            print(e)
            raise SystemExit


def get_video_length(filename):
    output = subprocess.check_output(("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                      "default=noprint_wrappers=1:nokey=1", filename)).strip()
    video_length = int(float(output))
    print("Video length in seconds: " + str(video_length))
    return video_length


def get_frame_rate(filename):
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1
    out = subprocess.check_output(
        ["ffprobe", filename, "-v", "0", "-select_streams", "v", "-print_format", "flat", "-show_entries",
         "stream=r_frame_rate"])
    rate = str(out).split('"')[1].strip().split('/')
    if len(rate) == 1:
        return float(rate[0])
    if len(rate) == 2:
        return float(rate[0]) / float(rate[1])
    return -1


def extract_audio_from_video(video_file):
    audio_file = "{}.wav".format(video_file)
    if os.path.isfile(audio_file):
        print("Audio File {} Already Exist".format(audio_file))
    else:
        try:
            subprocess.check_output(
                ("ffmpeg", "-i", video_file, "-vn", "-f", "wav", "-ab", "192000", audio_file)).strip()
        except KeyError as e:
            print(e)
            raise SystemExit


def extract_audio_from_video_movie_py(video_file):
    audio_file = "{}.wav".format(video_file)
    if os.path.isfile(audio_file):
        print("Audio File {} Already Exist".format(audio_file))
    else:
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(audio_file)


def merge_all_clips(clips_list, output_file):
    if len(clips_list) > 0:
        clips = []
        for clip in clips_list:
            clips.append(VideoFileClip(clip))

        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_file)
