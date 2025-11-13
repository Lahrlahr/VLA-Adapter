import json
import os.path

import numpy as np
from tqdm import tqdm
from pathlib import Path
import tyro
from process_util import find_second_level_dirs, find_first_level_dirs

load_json = 'sandwich_1113_data.json'
key_left = 'puppet/joint_position_left'
key_right = 'puppet/joint_position_right'
img_front = 'observations_rgb_images_camera_front_image'
img_left = "observations_rgb_images_camera_left_wrist_image"
img_right = "observations_rgb_images_camera_right_wrist_image"

def get_last_two(path):
    p = Path(path)
    if len(p.parts) < 2:
        return None  # 路径太短
    return "/".join(p.parts[-2:])

def load_and_filter_episodes(dir_path, min_length=50, pad=True):
    if not os.path.exists(os.path.join(dir_path, load_json)):
        return []
    with open(os.path.join(dir_path, load_json), 'r') as f:
        data = json.load(f)

    change = False
    episode_dict = {}
    episode_num = 0
    for episode in tqdm(data.values(), desc="Processing Episodes", total=len(data)):
        step_dict = {}
        step_num = 0
        for step in episode.values():
            step['prompt'] = 'Use the right arm to grab the breads and use the left arm to grab the meat and vegetables to make a sandwich.'
            step[img_front] = get_last_two(step[img_front])
            step[img_left] = get_last_two(step[img_left])
            step[img_right] = get_last_two(step[img_right])
            change = True

            if not (os.path.exists(os.path.join(dir_path, step[img_front])) and os.path.exists(
                    os.path.join(dir_path, step[img_left])) and os.path.exists(os.path.join(dir_path, step[img_right]))):
                change = True
                continue

            if step_num == 0:
                step_dict[f'step_{step_num}'] = step
                step_num += 1

                prev_left = np.round(step.get(key_left), 3)
                prev_right = np.round(step.get(key_right), 3)
                previous_14d = np.concatenate([prev_left, prev_right])
                continue

            curr_left = np.round(step.get(key_left), 3)
            curr_right = np.round(step.get(key_right), 3)
            current_14d = np.concatenate([curr_left, curr_right])
            if np.array_equal(current_14d, previous_14d):
                change = True
                continue

            step_dict[f'step_{step_num}'] = step
            step_num += 1

            prev_left = curr_left.copy()
            prev_right = curr_right.copy()
            previous_14d = np.concatenate([prev_left, prev_right])

        if len(step_dict) < min_length:
            change = True
            continue

        episode_dict[f'episode_{episode_num}'] = step_dict
        episode_num += 1

    if change == True:
        return episode_dict
    else:
        return []


def aa(root_dirs):
    if ',' in root_dirs:
        root_dirs = root_dirs.split(',')
    else:
        root_dirs = [root_dirs]

    return root_dirs


def main():
    # for root_dir in aa(
    #         '/data/share_nips/robot/ario1/wipe_the_table/,/data/share_nips/robot/ario1/take_the_pen_out_of_the_pen_holder/,/data/share_nips/robot/ario1/take_the_glasses_out_of_the_glasses_case/,/data/share_nips/robot/ario1/sort_the_fruits/,/data/share_nips/robot/ario1/sort_the_blocks/,/data/share_nips/robot/ario1/put_the_pen_into_the_pen_holder/,/data/share_nips/robot/ario1/put_the_glasses_into_the_glasses_case/,/data/share_nips/robot/ario1/place_the_fork/,/data/share_nips/robot/ario1/fold_the_towel/,/data/share_nips/robot/ario1/fold_the_shorts/,/data/share_nips/robot/ario1/cap_the_pen/'):
    #     for dir_path in find_first_level_dirs(root_dir):
    #         # for dir_path in ['/data/share_nips/robot/aidlux/cxg/sandwich_reset/']:
    #         data = load_and_filter_episodes(dir_path)
    #         if not data:
    #             continue
    #         with open(os.path.join(dir_path, 'data.json'), 'w') as f:
    #             json.dump(data, f, indent=2)
    #         print(f"成功保存🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️, {dir_path}")

    for dir_path in aa('/data/share_nips/robot/aidlux/cxg/sandwich_1113/'):
        data = load_and_filter_episodes(dir_path)
        if not data:
            continue
        with open(os.path.join(dir_path, 'data.json'), 'w') as f:
            json.dump(data, f, indent=2)
        print(f"成功保存🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️, {dir_path}")


if __name__ == "__main__":
    tyro.cli(main)
