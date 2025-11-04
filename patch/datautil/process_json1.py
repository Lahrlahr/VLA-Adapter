import json
import os.path

import numpy as np
from tqdm import tqdm
import tyro
from process_util import find_second_level_dirs, find_first_level_dirs


def load_and_filter_episodes(json_path, min_length=10, pad=True):
    with open(json_path, 'r') as f:
        data = json.load(f)  # dict: {episode_0: {step_0: {}, step_1: {}}, ...}

    key_left = 'puppet/joint_position_left'
    key_right = 'puppet/joint_position_right'

    filtered_episodes = {}

    total_original_steps = 0
    total_filtered_steps = 0
    dropped_episodes = 0

    for ep_name, episode in tqdm(data.items(), desc="Processing Episodes", total=len(data)):
        steps = list(episode.items())
        if not steps:
            continue  # 空 episode 直接跳过

        if not os.path.isdir(os.path.join('/data/share_nips/robot/aidlux/cxg/sandwich/',ep_name)):
            continue

        total_original_steps += len(steps)

        # 初始化第一个 step
        prev_step_name, prev_step_data = steps[0]
        prev_left = np.round(prev_step_data.get(key_left), 3)
        prev_right = np.round(prev_step_data.get(key_right), 3)

        filtered_step_list = [prev_step_data]  # 保留第一个 step
        # if pad:
        #     for i in range(50):
        #         filtered_step_list.append(prev_step_data)
        #     renamed_steps = {f"step_{i}": step_data for i, step_data in enumerate(filtered_step_list)}
        #     filtered_episodes[ep_name] = renamed_steps
        #     continue

        for step_name, step_data in steps[1:]:
            curr_left = np.round(step_data.get(key_left), 3)
            curr_right = np.round(step_data.get(key_right), 3)

            current_14d = np.concatenate([curr_left, curr_right])
            previous_14d = np.concatenate([prev_left, prev_right])

            if not np.array_equal(current_14d, previous_14d):
                filtered_step_list.append(step_data)
                prev_left = curr_left.copy()
                prev_right = curr_right.copy()

        total_filtered_steps += len(filtered_step_list)

        if len(filtered_step_list) >= min_length:
            # 重新命名 step 为 step_0, step_1, ...
            renamed_steps = {f"step_{i}": step_data for i, step_data in enumerate(filtered_step_list)}
            filtered_episodes[ep_name] = renamed_steps
        else:
            dropped_episodes += 1

    # 打印统计信息
    removed_count = total_original_steps - total_filtered_steps
    print(f"原始总 step 数: {total_original_steps}")
    print(f"去重后总 step 数: {total_filtered_steps}")
    print(f"共过滤掉 {removed_count} 个重复的 step")
    print(f"共丢弃 {dropped_episodes} 个长度小于 {min_length} 的 episode")
    print(f"最终保留 {len(filtered_episodes)} 个 episode")

    return filtered_episodes

def main():
    # for dir_path in find_first_level_dirs(root_dir):
    for dir_path in ['/data/share_nips/robot/aidlux/cxg/sandwich_reset/']:
        # if not os.path.exists(os.path.join(dir_path, 'data.json')):
        #     continue
        data = load_and_filter_episodes(os.path.join(dir_path, 'episode.json'))
        with open(os.path.join(dir_path, 'data.json'), 'w') as f:
            json.dump(data, f, indent=2)
        print(f"成功保存🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️, {dir_path}")

if __name__ == "__main__":
     tyro.cli(main)