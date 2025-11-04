import os
from pathlib import Path
def find_second_level_dirs(root_dir):
    second_level_dirs = []

    # 遍历一级子目录（depth=1）
    for dir1_name in os.listdir(root_dir):
        dir1_path = os.path.join(root_dir, dir1_name)
        if not os.path.isdir(dir1_path):
            continue  # 跳过文件

        # 遍历一级目录下的内容（即二级目录）
        for dir2_name in os.listdir(dir1_path):
            dir2_path = os.path.join(dir1_path, dir2_name)
            if os.path.isdir(dir2_path):
                second_level_dirs.append(dir2_path)

    return second_level_dirs


def find_first_level_dirs(root_dir):
    first_level_dirs = []
    for item_name in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item_name)
        if os.path.isdir(item_path):
            first_level_dirs.append(item_path)
    return first_level_dirs

def get_last_two(path):
    p = Path(path)
    if len(p.parts) < 2:
        return None  # 路径太短
    return "/".join(p.parts[-2:])