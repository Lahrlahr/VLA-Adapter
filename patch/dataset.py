import torch
import numpy as np
from pathlib import Path
import os
import json
import random
from PIL import Image
from dataclasses import dataclass
from .data_transform import process_data1, process_data2
from patch.constant import *

def get_last_two(path):
    p = Path(path)
    if len(p.parts) < 2:
        return None  # 路径太短
    return "/".join(p.parts[-2:])

def get_stats(all_episodes):
    all_data = []
    for episode in all_episodes:
        for step_data in episode:
            joint_position_left = step_data.get(DATASET_LEFT_JOINT)  # 长度应为7
            joint_position_right = step_data.get(DATASET_RIGHT_JOINT)  # 长度应为7

            full_state = np.concatenate([joint_position_left, joint_position_right])
            all_data.append(full_state)
    all_data = np.array(all_data)  # N x 14

    norm_stats = {
        'qpos': {
            'mean': all_data.mean(axis=0).tolist(),  # 每个关节的均值，形状 [14]
            'std': all_data.std(axis=0).tolist(),  # 每个关节的标准差，形状 [14]
            'q01': np.percentile(all_data, 1, axis=0).tolist(),  # 1% 分位数，形状 [14]
            'q99': np.percentile(all_data, 99, axis=0).tolist()  # 99% 分位数，形状 [14]
        }
    }

    # stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    # with open(stats_path, 'wb') as f:
    #     pickle.dump(stats, f)
    #
    return norm_stats

def preprocess_dataset(json_paths, seed: int = 42):
    if ',' in json_paths:
        json_paths = json_paths.split(',')
    else:
        json_paths = [json_paths]
    all_episodes = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
        json_dir = os.path.dirname(json_path)
        for episode_data in data.values():
            all_steps = []
            for step_data in episode_data.values():
                for key, value in step_data.items():
                    if key.startswith('observations'):
                        full_path = os.path.join(json_dir, get_last_two(value))
                        step_data[key] = full_path
                all_steps.append(step_data)
            all_episodes.append(all_steps)

    norm_stats = get_stats(all_episodes)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(all_episodes), generator=generator).tolist()
    train_episodes = [all_episodes[i] for i in indices]

    return train_episodes, norm_stats

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, num_queries):
        self.episode_list , self.norm_stats = preprocess_dataset(json_path)
        self.num_queries = num_queries

    def __len__(self) -> int:
        return len(self.episode_list)

    def __getitem__(self, idx: int):
        episode_data = self.episode_list[idx]
        start_idx = random.randint(0, len(episode_data) - 1)
        end_idx = start_idx + self.num_queries + 1
        indices = [min(i, len(episode_data) - 1) for i in range(start_idx, end_idx)]
        chunk = [episode_data[i] for i in indices]

        current_step = chunk[0]
        front_image = np.array(Image.open(current_step[DATASET_FRONT_IMAGE]).convert("RGB"))
        left_image = np.array(Image.open(current_step[DATASET_LEFT_IMAGE]).convert("RGB"))
        right_image = np.array(Image.open(current_step[DATASET_RIGHT_IMAGE]).convert("RGB"))
        proprio_left = np.array(current_step[DATASET_LEFT_JOINT])
        proprio_right = np.array(current_step[DATASET_RIGHT_JOINT])
        prompt = current_step[DATASET_PROMPT]

        left_list = []
        right_list = []
        for step in chunk[1:]:
            left_list.append(step[DATASET_LEFT_JOINT])
            right_list.append(step[DATASET_RIGHT_JOINT])
        actions = np.concatenate([np.array(left_list), np.array(right_list)], axis=1)

        result = {
            OBS_FRONT_IMAGE:  front_image,
            OBS_LEFT_IMAGE:  left_image,
            OBS_RIGHT_IMAGE:  right_image,
            OBS_LEFT_JOINT: proprio_left,
            OBS_RIGHT_JOINT: proprio_right,
            OBS_PROMPT: prompt,
            'actions': actions
        }
        return result



class VLAAdapterDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, num_queries, processor):
        self.dataset = EpisodicDataset(json_path, num_queries)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        result = process_data1(self.processor, self.dataset.norm_stats, self.dataset[idx])
        return result

    @property
    def dataset_statistics(self):
        return self.dataset.norm_stats

@dataclass
class VLAAdapterCollator:
    pad_token_id: int = 151643
    padding_side: str = "right"
    def __call__(self, items):
        result = process_data2(items)
        return result
