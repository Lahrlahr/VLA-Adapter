from openpi_client.websocket_client_policy import WebsocketClientPolicy
import torch
import numpy as np
from constant import *
from dataset import EpisodicDataset
import pandas as pd
from tabulate import tabulate

client = WebsocketClientPolicy(host="0.0.0.0", port=2251)

obs = {
    OBS_FRONT_IMAGE: np.full((480, 640, 3), 200, dtype=np.uint8),
    OBS_LEFT_IMAGE: np.full((480, 640, 3), 200, dtype=np.uint8),
    OBS_RIGHT_IMAGE: np.full((480, 640, 3), 200, dtype=np.uint8),
    OBS_LEFT_JOINT: np.full((7,), 0.5),
    OBS_RIGHT_JOINT: np.full((7,), 0.5),
    OBS_PROMPT: 'task'
}
action = client.infer(obs)

dataset = EpisodicDataset('/data/share_nips/robot/aidlux/cxg/sandwich_1010/data.json,/data/share_nips/robot/aidlux/cxg/sandwich_1009/data.json,/data/share_nips/robot/aidlux/cxg/sandwich/data.json,/data/share_nips/robot/aidlux/cxg/sandwich2/data.json,/data/share_nips/robot/aidlux/cxg/sandwich_reset/data.json', 50)
for num in range(0, len(dataset), 30):
    item = dataset[num]
    action1 = item['actions']
    item.pop('actions')
    action = client.infer(item)['actions']

    diffs = action - action1

    data = []
    data.append([f"Pred_mean"] + np.abs(diffs).mean(0).tolist())
    for t in range(diffs.shape[0]):
        diff = diffs[t]
        data.append([f"Pred_{t + 1:02d}"] + diff.tolist())
    data.append([f"Pred_mean"] + np.abs(diffs).mean(0).tolist())

    joint = [f"L{i}" for i in range(7)] + [f"R{i}" for i in range(7)]
    df = pd.DataFrame(data, columns=["Step"] + joint)
    table_str = tabulate(df, headers="keys", tablefmt="grid", floatfmt=".6f")

    print(f"\n🔍 Step {num} - 关节位置轨迹")
    print(table_str)

    with open('log', "a") as f:
        f.write("\n")
        f.write(table_str)
        f.write("\n")