import numpy as np
import pandas as pd
from tabulate import tabulate
import os
from datetime import datetime, timezone, timedelta
from PIL import Image
from patch.constant import *


class ImageBuffer:
    def __init__(self, buffer_dir):
        self.buffer_dir = buffer_dir
        self.counter = 0

    def add_image(self, img):
        os.makedirs(self.buffer_dir, exist_ok=True)
        filepath = os.path.join(self.buffer_dir, f"frame_{self.counter:06d}.jpg")
        Image.fromarray(img).save(filepath)
        self.counter += 1


class JointVisualizer:
    def __init__(self, save_dir, filename='joint.log'):
        self.save_dir = save_dir
        self.log_path = os.path.join(self.save_dir, filename)
        self.log_path1 = os.path.join(self.save_dir, 'debug.log')

        # 列名：左臂 L0-L6，右臂 R0-R6
        self.columns = [f"L{i}" for i in range(7)] + [f"R{i}" for i in range(7)]
        self.last_action = None

    def visualize(self, joint, actions, step=0):
        os.makedirs(self.save_dir, exist_ok=True)
        bj_time = datetime.now(timezone(timedelta(hours=8)))

        if self.last_action is not None:
            error = np.abs(self.last_action - joint)
            mean_error = np.mean(error)
            max_error = np.max(error)
            argmax_dim = np.argmax(error)

            with open(self.log_path1, "a") as f:
                f.write(f"\n[Step {step}] 时间: {bj_time}\n")
                f.write(f"Error (|last_action - current_action|):\n")
                f.write(f"  Mean: {mean_error:.6f}, Max: {max_error:.6f} (dim {argmax_dim})\n")
                f.write("  Per-dim: " + "  ".join([f"{e:.6f}" for e in error]) + "\n")
        self.last_action = actions[-1].copy()

        data = []
        data.append(["Current"] + joint.tolist())
        for t in range(actions.shape[0]):
            action = actions[t]
            data.append([f"Pred_{t + 1:02d}"] + action.tolist())

        df = pd.DataFrame(data, columns=["Step"] + self.columns)
        table_str = tabulate(df, headers="keys", tablefmt="grid", floatfmt=".6f")

        print(f"\n🔍 Step {step} - 关节位置轨迹")
        print(table_str)

        with open(self.log_path, "a") as f:
            f.write(f"\n[Step {step}] 时间: {bj_time}\n")
            f.write(table_str)
            f.write("\n")


class ServerLog:
    def __init__(self, dir, img_dirs=None, joint_dir='joint'):
        self.img_buffers = []
        if img_dirs is None:
            img_dirs = ['image_front', 'image_left', 'image_right']
        for img_dir in img_dirs:
            self.img_buffers.append(ImageBuffer(os.path.join(dir, img_dir)))

        self.joint_visualizer = JointVisualizer(os.path.join(dir, joint_dir))
        self.step = 0  # 记录当前步数

    def write(self, obs, actions):
        img_list = [obs[OBS_FRONT_IMAGE], obs[OBS_LEFT_IMAGE], obs[OBS_RIGHT_IMAGE]]
        for i, img in enumerate(img_list):
            self.img_buffers[i].add_image(img)

        joint = np.concatenate([obs[OBS_LEFT_JOINT], obs[OBS_RIGHT_JOINT]])
        self.joint_visualizer.visualize(joint, actions, step=self.step)
        self.step += 1
