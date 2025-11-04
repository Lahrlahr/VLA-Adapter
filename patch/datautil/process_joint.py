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

joint_data = []
n = 0
for step, obs in episodes['episode_0'].items():
    joint_data.append([f"Pred_{n:03d}"] + obs['puppet/joint_position_left'] + obs['puppet/joint_position_right'])
    n += 1

columns = [f"L{i}" for i in range(7)] + [f"R{i}" for i in range(7)]
df = pd.DataFrame(joint_data, columns=["Step"] + columns)
table_str = tabulate(df, headers="keys", tablefmt="grid", floatfmt=".6f")

with open(log_path, "a") as f:
    f.write(f"\n时间: {datetime.now()}\n")
    f.write(table_str)
    f.write("\n")
