from datetime import datetime

from src.main.highlight_generator import process_video, init_video_processing

SEPARATOR = '-'

configs = {
    'WRITE_DEMO_VIDEO': True,
    'DEMO_FULL_SCREEN': True,
    'OUTPUT_FOLDER_PATH': '../../output',
    'ROI_FRACTION': 0.25,
    'ROI_LOCATION': 'bottom',  # (top, bottom, left, right)
    'SEPARATOR': SEPARATOR,
    'SCORE_BOARD_REGEX_SF': '^[0-9]{1,3}' + SEPARATOR + '[0-9]$',  # (SF) scoreboard first
    'SCORE_BOARD_REGEX_WF': '^[0-9]' + SEPARATOR + '[0-9]{1,3}$',  # (WF) wickets first
    'SCOREBOARD_FORMAT': 'SF',  # (SF) scoreboard first | (WF) wickets first
    'COUNTRY_LIST': ['SL', 'WIN', 'AUS', 'IND', 'ENG', 'TAS', 'AFG', 'BAN']
}


def main():
    start_time = datetime.now()
    init_video_processing(configs)
    session_id = process_video("../../video/uploaded.mp4", start_position=None, end_position=None)  # 00:00:00
    end_time = datetime.now()
    print("finished {} in {} seconds".format(session_id, int((end_time - start_time).total_seconds())))


if __name__ == '__main__':
    main()
