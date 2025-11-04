import torch
import numpy as np
import cv2
from torch.nn.utils.rnn import pad_sequence
from patch.constant import *

def process_data1(processor, stats, item):
    prompt = item['prompt']
    lines = "<|im_start|>system\n" \
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" \
            "<|im_start|>user\n" \
            f"What action should the robot take to{prompt} ?<|im_end|>\n" \
            "<|im_start|>assistant\n"
    input_ids = processor.tokenizer(lines, add_special_tokens=False)['input_ids']

    input_ids = torch.tensor(input_ids + 64 * [151530])
    labels = input_ids.clone()
    labels[: -(64 + 1)] = -100

    q01 = torch.tensor(stats['qpos']['q01'])
    q99 = torch.tensor(stats['qpos']['q99'])
    proprio = torch.from_numpy(
        np.concatenate([item[OBS_LEFT_JOINT], item[OBS_RIGHT_JOINT]]))
    proprio = torch.clamp(2 * (proprio - q01) / (q99 - q01) - 1, -1, 1)
    if 'actions' in item:
        actions = torch.from_numpy(item['actions'])
        actions = torch.clamp(2 * (actions - q01) / (q99 - q01) - 1, -1, 1)
    else:
        actions = None

    img_listt = []
    for img in [item[OBS_FRONT_IMAGE], item[OBS_LEFT_IMAGE],
                item[OBS_RIGHT_IMAGE]]:
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0

        mean1 = np.array([0.485, 0.456, 0.406])
        std1 = np.array([0.229, 0.224, 0.225])
        img1 = (img - mean1) / std1

        mean2 = np.array([0.5, 0.5, 0.5])
        std2 = np.array([0.5, 0.5, 0.5])
        img2 = (img - mean2) / std2
        img_listt.append(torch.from_numpy(np.concatenate([img1, img2], axis=-1)).permute(2, 0, 1))
    pixel_values = torch.cat(img_listt, dim=0)

    result = {
        'input_ids': input_ids,
        'pixel_values': pixel_values,
        'proprio': proprio,
        'actions': actions,
        'labels': labels,
    }

    return result

def process_data2(items):
    pad_token_id = 151643
    IGNORE_INDEX = -100

    input_ids = [item['input_ids'] for item in items]
    pixel_values = [item['pixel_values'] for item in items]
    proprio = [item['proprio'] for item in items]
    labels = [item['labels'] for item in items]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(pad_token_id)

    pixel_values = torch.stack(pixel_values, dim=0)
    proprio = torch.stack(proprio, dim=0)
    if items[0]['actions'] != None:
        actions = [item['actions'] for item in items]
        actions = torch.stack(actions, dim=0)
    else:
        actions = None

    result = dict(
        pixel_values=pixel_values,
        proprio=proprio,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        actions=actions,
    )

    return result

def unnormolize_data(stats, actions):
    q01 = torch.tensor(stats['qpos']['q01'])
    q99 = torch.tensor(stats['qpos']['q99'])
    actions = 0.5 * (actions + 1) * (q99 - q01 + 1e-8) + q01
    return actions