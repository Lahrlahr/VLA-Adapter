import numpy as np
import pandas as pd
from tabulate import tabulate
import os
from datetime import datetime
import glob
import json

json_files = glob.glob(os.path.join('/data/share_nips/robot/aidlux/bread/', "*.json"))
log_path = 'hhh.log'
with open(json_files[0], 'r') as f:
    episodes = json.load(f)


def collect_joint_data(episodes, key_left='puppet/joint_position_left', key_right='puppet/joint_position_right'):
    """
    从所有 episode 中收集左右手关节数据
    """
    all_left_joints = []
    all_right_joints = []

    for episode_name, episode in episodes.items():
        for step, obs in episode.items():
            all_left_joints.append(obs[key_left])
            all_right_joints.append(obs[key_right])

    return np.array(all_left_joints), np.array(all_right_joints)


def summarize_joint_stats(joint_array, side='Left', joint_prefix='Joint'):
    """
    生成每个关节的统计信息表格
    """
    num_joints = joint_array.shape[1]
    stats = []

    for i in range(num_joints):
        data = joint_array[:, i]
        stats.append({
            'Side': side,
            'Joint': f'{joint_prefix}_{i}',
            'Mean': data.mean(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Median': np.median(data),
            'Range': data.max() - data.min(),
            'Q10': np.percentile(data, 10),
            'Q90': np.percentile(data, 90)
        })

    return pd.DataFrame(stats)


# === 主流程 ===
all_left_joints, all_right_joints = collect_joint_data(episodes)

# 生成统计表
df_left = summarize_joint_stats(all_left_joints, side='Left', joint_prefix='L')
df_right = summarize_joint_stats(all_right_joints, side='Right', joint_prefix='R')

# 合并为一个大表
stats_summary = pd.concat([df_left, df_right], ignore_index=True)
table_str = tabulate(stats_summary, headers="keys", tablefmt="grid", floatfmt=".6f")

print("📊 所有 Episode 中关节角度统计信息")
print("=" * 80)
print(table_str)