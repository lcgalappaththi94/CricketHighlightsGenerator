import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.main.video_utils import split_video_by_config


def merge_consecutive_time_intervals(df):
    temp = []
    i = 0
    j = 0
    n = len(df) - 2
    m = len(df) - 1
    while i <= n:
        j = i + 1
        while j <= m:
            if df['end'][i] == df['start'][j]:
                df.loc[i, 'end'] = df.loc[j, 'end']
                temp.append(j)
                j = j + 1
            else:
                i = j
                break

    df.drop(temp, axis=0, inplace=True)
    return df


def get_high_energy_intervals_with_threshold(energy, threshold):
    df = pd.DataFrame(columns=['energy', 'start', 'end'])

    row_index = 0
    for i in range(len(energy)):
        value = energy[i]
        if value >= threshold:
            i = np.where(energy == value)[0]
            df.loc[row_index, 'energy'] = value
            df.loc[row_index, 'start'] = i[0] * 5
            df.loc[row_index, 'end'] = (i[0] + 1) * 5
            row_index = row_index + 1

    return df


def get_highlight_points_from_extracted_audio(audio_path):
    x, sr = librosa.load(audio_path, sr=44100)
    print("length of the audio is {} minutes".format(int(librosa.get_duration(x, sr) / 60)))

    max_slice = 5
    window_length = max_slice * sr

    energy = np.array([sum(abs(x[i:i + window_length] ** 2)) for i in range(0, len(x), window_length)])

    plt.hist(energy)
    plt.show()

    threshold = int(input("Enter the threshold value based on plot: "))

    high_threshold_intervals_data_frame = get_high_energy_intervals_with_threshold(energy, threshold)
    final_high_energy_intervals = merge_consecutive_time_intervals(high_threshold_intervals_data_frame)

    return final_high_energy_intervals


def split_video_by_intervals(df, video_path, prefix):
    start = np.array(df['start'])
    end = np.array(df['end'])
    config_list = []

    for i in range(len(df)):
        if i != 0:
            start_lim = start[i] - 5
        else:
            start_lim = start[i]
        end_lim = end[i]

        config_list.append({
            "start_time": start_lim,
            "length": end_lim - start_lim,
            "rename_to": "../../output/{}_cut_{}.mp4".format(prefix, str(i + 1))
        })

    split_video_by_config(video_path, config_list)
    return [d['rename_to'] for d in config_list]
