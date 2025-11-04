import json
import os
import cv2
from tqdm import tqdm
from glob import glob
from process_util import find_second_level_dirs, find_first_level_dirs ,get_last_two
import tyro

def main(fps=30):
    root_dir = '/data/share_nips/robot/aidlux/cxg/sort_fruit_and_bread/'
    video_root = os.path.join(root_dir, 'video')
    os.makedirs(video_root, exist_ok=True)
    # for dir_path in find_first_level_dirs(root_dir):
    for dir_path in ['/data/share_nips/robot/aidlux/cxg/sort_fruit_and_bread/']:
        data_files = glob(os.path.join(dir_path, 'data.json'))
        if not data_files:
            continue
        with open(data_files[0], 'r') as f:
            data = json.load(f)
        session_name = os.path.basename(dir_path)
        num = 0
        for episode_name, episode_data in tqdm(data.items(), desc=f"Episodes in ({dir_path})", total=3):
            # front_video_file = os.path.join(video_root, session_name, episode_name, 'front.mp4')
            # left_video_file = os.path.join(video_root, session_name, episode_name, 'left.mp4')
            # right_video_file = os.path.join(video_root, session_name, episode_name, 'right.mp4')

            front_video_file = os.path.join(video_root, episode_name, 'front.mp4')
            left_video_file = os.path.join(video_root,  episode_name, 'left.mp4')
            right_video_file = os.path.join(video_root,  episode_name, 'right.mp4')

            os.makedirs(os.path.dirname(front_video_file), exist_ok=True)
            os.makedirs(os.path.dirname(left_video_file), exist_ok=True)
            os.makedirs(os.path.dirname(right_video_file), exist_ok=True)

            front_video = None
            left_video = None
            right_video = None
            for step_name, step_data in episode_data.items():
                front_img_path = os.path.join(dir_path, get_last_two(step_data['observations_rgb_images_camera_front_image']))
                left_img_path = os.path.join(dir_path,get_last_two(step_data['observations_rgb_images_camera_left_wrist_image']))
                right_img_path = os.path.join(dir_path, get_last_two(step_data['observations_rgb_images_camera_right_wrist_image']))

                front_img = cv2.imread(front_img_path)
                left_img = cv2.imread(left_img_path)
                right_img = cv2.imread(right_img_path)

                if front_video is None:
                    h, w = front_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    front_video = cv2.VideoWriter(front_video_file, fourcc, fps, (w, h))

                if left_video is None:
                    h, w = left_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    left_video = cv2.VideoWriter(left_video_file, fourcc, fps, (w, h))

                if right_video is None:
                    h, w = right_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    right_video = cv2.VideoWriter(right_video_file, fourcc, fps, (w, h))

                front_video.write(front_img)
                left_video.write(left_img)
                right_video.write(right_img)

            front_video.release()
            left_video.release()
            right_video.release()
            print(f"✅ 视频已保存: {front_video_file}, {left_video_file}, {right_video_file}")
            num += 1
            if num == 3:
                break

if __name__ == "__main__":
     tyro.cli(main)