import os
import json
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from process_util import find_second_level_dirs, find_first_level_dirs

def extract_episode_number(path):
    # 从路径的文件夹名中提取数字
    basename = os.path.basename(path)
    match = re.search(r'(\d+)', basename)
    return int(match.group(1)) if match else float('inf')  # 没有数字的排最后

def load_and_filter_episodes(data, min_length=100):
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

        total_original_steps += len(steps)

        # 初始化第一个 step
        prev_step_name, prev_step_data = steps[0]
        prev_left = np.round(prev_step_data.get(key_left), 2)
        prev_right = np.round(prev_step_data.get(key_right), 2)

        filtered_step_list = [prev_step_data]  # 保留第一个 step

        for step_name, step_data in steps[1:]:
            curr_left = np.round(step_data.get(key_left), 2)
            curr_right = np.round(step_data.get(key_right), 2)

            # 安全检查：防止 None
            if curr_left is None or curr_right is None:
                continue

            current_14d = np.concatenate([curr_left, curr_right])
            previous_14d = np.concatenate([prev_left, prev_right])

            if not np.array_equal(current_14d, previous_14d):
                filtered_step_list.append(step_data)
                prev_left = curr_left.copy()
                prev_right = curr_right.copy()

        total_filtered_steps += len(filtered_step_list)

        # 🔥 关键：只保留长度 >= min_length 的 episode
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

prompt_map = {
    'put_pen_left_arm_pure_white_bg': 'Use the left arm to put the pen into the pen holder.',
    'put_pen_left_arm_gray_white_bg': 'Use the left arm to put the pen into the pen holder.',
    'put_pen_left_arm_cyan_white_bg': 'Use the left arm to put the pen into the pen holder.',
    'put_pen_right_arm_pure_white_bg': 'Use the right arm to put the pen into the pen holder.',
    'put_pen_right_arm_gray_white_bg': 'Use the right arm to put the pen into the pen holder.',
    'put_pen_right_arm_cyan_white_bg': 'Use the right arm to put the pen into the pen holder.',

    'take_pen_left_arm_black_white_bg': 'Use the left arm to take the pen out of the pen holder.',
    'take_pen_left_arm_gray_white_bg': 'Use the left arm to take the pen out of the pen holder.',
    'take_pen_left_arm_green_white_bg': 'Use the left arm to take the pen out of the pen holder.',
    'take_pen_right_arm_black_white_bg': 'Use the right arm to take the pen out of the pen holder.',
    'take_pen_right_arm_gray_white_bg': 'Use the right arm to take the pen out of the pen holder.',
    'take_pen_right_arm_green_white_bg': 'Use the right arm to take the pen out of the pen holder.',

    'sort_blocks_green_bg': 'Sort the blocks by color.',
    'sort_blocks_gray_white_bg': 'Sort the blocks by color.',
    'sort_blocks_kd': 'Sort the blocks by color.',

    'set_the_cup_upright_black_white_bg': 'Set the cup upright.',
    'set_the_cup_upright_gray_white_bg': 'Set the cup upright.',
    'set_the_cup_upright_cyan_white_bg': 'Set the cup upright.',

    'cap_the_pen_gray_white_plaid': 'Cap the pen.',
    'cap_the_pen_black_white_plaid': 'Cap the pen.',
    'cap_the_pen_green_white_plaid': 'Cap the pen.',

    'close_the_book_gray_white_bg': 'Close the book.',
    'close_the_book_black_white_bg': 'Close the book.',
    'close_the_book_green_white_bg': 'Close the book.',

    'fold_the_shorts_gray_white_bg': 'Fold the shorts.',
    'fold_the_shorts_green_white_bg': 'Fold the shorts.',
    'fold_the_shorts_cyan_white_bg': 'Fold the shorts.',

    'fold_the_towel_gray_white_bg': 'Fold the towel.',
    'fold_the_towel_green_white_bg': 'Fold the towel.',
    'fold_the_towel_pure_white_bg': 'Fold the towel.',
    'fold_the_towel_black_white_plaid': 'Fold the towel.',

    'wipe_the_table_coffee_3': 'Wipe the table.',
    'wipe_the_table_coffee_2': 'Wipe the table.',
    'wipe_the_table': 'Wipe the table.',

    'turn_the_pages_of_the_book_black_white_bg': 'Turn the pages of the book',
    'turn_the_pages_of_the_book_gray_white_bg': 'Turn the pages of the book',
    'turn_the_pages_of_the_book_green_white_bg': 'Turn the pages of the book',

    'sort_the_fruits': 'Sort the fruits',
    'sort_the_fruits_black_white_bg': 'Sort the fruits',
    'sort_the_fruits_green_white_bg': 'Sort the fruits',

    'place_the_fork': 'Place the fork',
    'place_the_fork_cyan_white_plaid': 'Place the fork',
    'place_the_fork_green_bg': 'Place the fork',
    'place_the_fork_grey_bg': 'Place the fork',
    'place_the_fork_brown_bg': 'Place the fork',
    'place_the_fork_blue_bg': 'Place the fork',
    'place_the_fork_black_white_bg': 'Place the fork',

    'open_the_shoebox': 'Open the shoebox',
    'open_the_shoebox_black_white_bg': 'Open the shoebox',
    'open_the_shoebox_cyan_white_bg': 'Open the shoebox',

    'close_the_shoebox': 'Close the shoebox',
    'close_the_shoebox_black_white_bg': 'Close the shoebox',
    'close_the_shoebox_cyan_white_bg': 'Close the shoebox',

    'take_the_glasses_out_of_the_glasses_case': 'Take the glasses out of the glasses case',
    'take_the_glasses_out_of_the_glasses_case_green_white_bg': 'Take the glasses out of the glasses case',
    'take_the_glasses_out_of_the_glasses_case_gray_white_bg': 'Take the glasses out of the glasses case',

    'put_the_glasses_into_the_glasses_case': 'Put the glasses into the glasses case',
    'put_the_glasses_into_the_glasses_case_green_white_bg': 'Put the glasses into the glasses case',
    'put_the_glasses_into_the_glasses_case_cyan_white_bg': 'Put the glasses into the glasses case',

}
joint_left = "arm/jointState/puppetLeft/"
joint_right = "arm/jointState/puppetRight/"
camera_front_image = "camera/color/front/"
camera_left_image = "camera/color/left/"
camera_right_image = "camera/color/right/"
txt_file = "sync.txt"
for root_dir in find_first_level_dirs('/data/share_nips/robot/ario1/open_the_shoebox'):
# for root_dir in ['/data/share_nips/robot/ario1/place_the_fork/place_the_fork_green_bg']:
    all_episodes = [
        os.path.join(root_dir, f) for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f)) and re.search(r'^episode\d+$', f)
    ]
    all_episodes = sorted(all_episodes, key=extract_episode_number)

    data = defaultdict(dict)
    for episode in tqdm(all_episodes, desc=f"Processing episodes in {root_dir}😁😁", total=len(all_episodes),
                        colour='green'):
        total_steps = 0
        paths = {
            'joint_left': os.path.join(episode, joint_left, txt_file),
            'joint_right': os.path.join(episode, joint_right, txt_file),
            'camera_front': os.path.join(episode, camera_front_image, txt_file),
            'camera_left': os.path.join(episode, camera_left_image, txt_file),
            'camera_right': os.path.join(episode, camera_right_image, txt_file),
        }
        with open(paths['joint_left'], 'r') as f:
            jl_lines = [line.strip() for line in f if line.strip()]
        with open(paths['joint_right'], 'r') as f:
            jr_lines = [line.strip() for line in f if line.strip()]
        with open(paths['camera_front'], 'r') as f:
            cf_lines = [line.strip() for line in f if line.strip()]
        with open(paths['camera_left'], 'r') as f:
            cl_lines = [line.strip() for line in f if line.strip()]
        with open(paths['camera_right'], 'r') as f:
            cr_lines = [line.strip() for line in f if line.strip()]
        lengths = [len(jl_lines), len(jr_lines), len(cf_lines), len(cl_lines), len(cr_lines)]
        if len(set(lengths)) != 1:
            raise ValueError(f"txt文件数量不一致: {lengths}")

        for jl_ts, jr_ts, cf_ts, cl_ts, cr_ts in zip(jl_lines, jr_lines, cf_lines, cl_lines, cr_lines):
            jl_path = os.path.join(episode, joint_left, jl_ts)
            jr_path = os.path.join(episode, joint_right, jr_ts)
            with open(jl_path, 'r') as f:
                jl_pos = json.load(f).get("position")
            with open(jr_path, 'r') as f:
                jr_pos = json.load(f).get("position")

            front_img = os.path.join(os.path.basename(episode), camera_front_image, cf_ts)
            left_wrist_img = os.path.join(os.path.basename(episode), camera_left_image, cl_ts)
            right_wrist_img = os.path.join(os.path.basename(episode), camera_right_image, cr_ts)

            prompt = prompt_map[os.path.basename(root_dir)]

            data[os.path.basename(episode).replace('episode', 'episode_')][f'step_{total_steps}'] = {
                'observations_rgb_images_camera_front_image': front_img,
                'observations_rgb_images_camera_left_wrist_image': left_wrist_img,
                'observations_rgb_images_camera_right_wrist_image': right_wrist_img,
                'puppet/joint_position_left': jl_pos,
                'puppet/joint_position_right': jr_pos,
                'prompt': prompt
            }
            total_steps += 1

    data = load_and_filter_episodes(data)

    output_json = os.path.join(root_dir, "data.json")
    with open(output_json, 'w') as out_f:
        json.dump(data, out_f, indent=2)
    print(f"🤣🤣🤣 Saved {len(data)} combined episodes to {output_json}")
