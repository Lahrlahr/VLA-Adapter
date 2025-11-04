import json
import os
from glob import glob
import numpy as np
from PIL import Image
import h5py
from tqdm import tqdm
import traceback
from process_util import find_second_level_dirs, find_first_level_dirs

for root_dir in [
                 '/data/share_nips/robot/ario1/sort_the_blocks',
                 ]:
    os.makedirs(os.path.join(root_dir, 'hdf5'), exist_ok=True)
    num = 0
    for dir_path in tqdm(find_first_level_dirs(root_dir), desc=f"Processing dirs in {root_dir}", position=0, ):
    # root_dir='/data/share_nips/robot/ario1/sort_the_fruits/'
    # for dir_path in tqdm(['/data/share_nips/robot/aidlux/bread_full'], desc=f"Processing dirs in {root_dir}", position=0, ):
        num = 0
        json_files = glob(os.path.join(dir_path, "data.json"))
        if not json_files:
            continue
        with open(json_files[0], 'r') as f:
            data = json.load(f)

    for episode_name, episode_data in tqdm(data.items(), desc=f"Episodes in {os.path.basename(dir_path)}",
                                           position=1, ):
        try:
            front_image = []
            left_image = []
            right_image = []
            joint_all = []
            for step_name, step_data in episode_data.items():
                front_image_path = os.path.join(dir_path, step_data["observations_rgb_images_camera_front_image"])
                left_image_path = os.path.join(dir_path, step_data["observations_rgb_images_camera_left_wrist_image"])
                right_image_path = os.path.join(dir_path, step_data["observations_rgb_images_camera_right_wrist_image"])
                front_image.append(np.array(Image.open(front_image_path).convert("RGB")))
                left_image.append(np.array(Image.open(left_image_path).convert("RGB")))
                right_image.append(np.array(Image.open(right_image_path).convert("RGB")))  # (H, W, 3)

                left_joint = np.array(step_data["puppet/joint_position_left"])
                right_joint = np.array(step_data["puppet/joint_position_right"])
                joint = np.concatenate([left_joint, right_joint], axis=0)
                joint_all.append(joint)
                lan = step_data['prompt']

            # hdf5_path = os.path.join(root_dir, f'hdf5/episode_{num:06d}.hdf5')
            hdf5_path = f'/data/share_nips/robot/ario1/put_the_bread_into_the_basket/hdf5/episode_{num:06d}.hdf5'
            num += 1
            with h5py.File(hdf5_path, 'w') as f:
                group = f.create_group("obs")
                group.create_dataset("front_image", data=np.stack(front_image[:-1], dtype='uint8'))
                group.create_dataset("left_image", data=np.stack(left_image[:-1], dtype='uint8'))
                group.create_dataset("right_image", data=np.stack(right_image[:-1], dtype='uint8'))

                group.create_dataset("state", data=np.stack(joint_all[:-1]), dtype='float32')

                f.create_dataset("action", data=np.stack(joint_all[1:]), dtype='float32')
                f.create_dataset("timestamp", data=np.stack([0 for f in joint_all[1:]]), dtype='float32')

                f.create_dataset("prompt", data='Put the bread into the basket.')
        except Exception as e:
            error_msg = f"[ERROR] Failed to process episode '{episode_name}' in {dir_path}"
            tqdm.write(error_msg)
            tqdm.write(f"        Exception: {type(e).__name__}: {e}")
            tqdm.write("        Traceback:")
            tqdm.write(traceback.format_exc())
            continue  # Proceed to next episode