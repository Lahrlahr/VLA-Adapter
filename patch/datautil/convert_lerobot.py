#!/usr/bin/env python3
# 注意这是agilex 双臂的转换脚本，单臂的转换脚本请参考 convert_taihu_data_to_lerobot.py
import sys
import json
import numpy as np
from PIL import Image
import tyro
import shutil
import logging
from tqdm import tqdm
import os
from typing import Optional, List
import gc
import sys
from glob import glob

sys.path.insert(0, "/data/pengtao/robot/convert_agilex_lerobot/lerobot/src")  # 添加上级目录到路径中

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import re
import pathlib
from pathlib import Path

# 定义各个关节的合法弧度范围
JOINT_RANGES = [
    (-2.618, 2.618),  # Joint1
    (0.0, 3.14),  # Joint2
    (-2.967, 0.0),  # Joint3
    (-1.745, 1.745),  # Joint4
    (-1.22, 1.22),  # Joint5
    (-2.0944, 2.0944),  # Joint6
    (0.0, 0.07)  # Joint7， gripper
]

image_size = [480, 640]  # 图像的高度和宽度

REPO_NAME = "agilex"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("convert_agilex_to_lerobot_cxg.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


def get_last_two(path):
    p = Path(path)
    if len(p.parts) < 2:
        return None  # 路径太短
    return "/".join(p.parts[-2:])


def find_first_level_dirs(root_dir):
    first_level_dirs = []
    for item_name in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item_name)
        if os.path.isdir(item_path):
            first_level_dirs.append(item_path)
    return first_level_dirs


def main(root: Optional[str] = "/data/share_nips/robot/aidlux/cxg/sandwich_1010/", prompt: Optional[
    str] = "Use the right arm to grab the breads and use the left arm to grab the meat and vegetables to make a sandwich."):
    dataset = LeRobotDataset.create(
        repo_id="agilex",
        robot_type="agilex",
        use_videos=False,
        fps=10,
        root=os.path.join(root, 'lerobot_en'),
        features={
            "observation.images.front": {  # 前置摄像头图像
                "dtype": "image",
                "shape": (image_size[0], image_size[1], 3),  # (3, 480, 640)
                "names": ["height", "width", "channels"],
            },
            "observation.images.left_wrist": {  # 左腕摄像头图像
                "dtype": "image",
                "shape": (image_size[0], image_size[1], 3),  # (3, 480, 640)
                "names": ["height", "width", "channels"],
            },
            "observation.images.right_wrist": {  # 右腕摄像头图像
                "dtype": "image",
                "shape": (image_size[0], image_size[1], 3),  # (3, 480, 640)
                "names": ["height", "width", "channels"],
            },
            "observation.state": {  # 左手关节位置
                "dtype": "float32",
                "shape": (14,),
                "names": {
                    "motors": [
                        "left_waist",
                        "left_shoulder",
                        "left_elbow",
                        "left_forearm_roll",
                        "left_wrist_angle",
                        "left_wrist_rotate",
                        "left_gripper",
                        "right_waist",
                        "right_shoulder",
                        "right_elbow",
                        "right_forearm_roll",
                        "right_wrist_angle",
                        "right_wrist_rotate",
                        "right_gripper"
                    ],
                },
            },
            "action": {  # 动作，包含左右手的动作
                "dtype": "float32",
                "shape": (14,),  # 7左 + 7右
                "names": {
                    "motors": [
                        "left_waist",
                        "left_shoulder",
                        "left_elbow",
                        "left_forearm_roll",
                        "left_wrist_angle",
                        "left_wrist_rotate",
                        "left_gripper",
                        "right_waist",
                        "right_shoulder",
                        "right_elbow",
                        "right_forearm_roll",
                        "right_wrist_angle",
                        "right_wrist_rotate",
                        "right_gripper"
                    ],
                },
            },
        },
        image_writer_threads=4,
        image_writer_processes=0
    )

    os.makedirs(os.path.join(root, 'lerobot_en'), exist_ok=True)
    for dir_path in [root, ]:
        # for dir_path in tqdm(find_first_level_dirs(root), desc=f"Processing dirs in {root}", position=0, ):
        json_files = glob(os.path.join(dir_path, "data.json"))
        if not json_files:
            continue
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        for episode_name, episode_data in tqdm(data.items(), desc=f"Episodes in {os.path.basename(dir_path)}",
                                               position=1, ):
            episode_data_list = list(episode_data.values())
            try:
                for i in range(len(episode_data_list) - 1):
                    current_step = episode_data_list[i]
                    next_step = episode_data_list[i + 1]

                    front_image_path = os.path.join(dir_path,
                                                    get_last_two(
                                                        current_step["observations_rgb_images_camera_front_image"]))
                    left_image_path = os.path.join(dir_path,
                                                   get_last_two(
                                                       current_step["observations_rgb_images_camera_left_wrist_image"]))
                    right_image_path = os.path.join(dir_path,
                                                    get_last_two(current_step[
                                                                     "observations_rgb_images_camera_right_wrist_image"]))
                    front_image = np.array(Image.open(front_image_path).convert("RGB"))
                    left_image = np.array(Image.open(left_image_path).convert("RGB"))
                    right_image = np.array(Image.open(right_image_path).convert("RGB"))  # (H, W, 3)

                    left_joint = np.array(current_step["puppet/joint_position_left"], dtype=np.float32)
                    right_joint = np.array(current_step["puppet/joint_position_right"], dtype=np.float32)
                    joint = np.concatenate([left_joint, right_joint], axis=0)

                    next_left_joint = np.array(next_step["puppet/joint_position_left"], dtype=np.float32)
                    next_right_joint = np.array(next_step["puppet/joint_position_right"], dtype=np.float32)
                    action = np.concatenate([next_left_joint, next_right_joint], axis=0)
                    frame_data = {
                        "observation.images.front": front_image,
                        "observation.images.left_wrist": left_image,
                        "observation.images.right_wrist": right_image,
                        "observation.state": joint,
                        "action": action,
                    }
                    dataset.add_frame(frame_data, task=prompt)
            except Exception as e:
                print(f"Unexpected error at step : {type(e).__name__}: {e}")
                try:
                    dataset.save_episode()
                    gc.collect()
                except Exception as e1:
                    pass
                continue

            dataset.save_episode()
            gc.collect()


if __name__ == "__main__":
    tyro.cli(main)
