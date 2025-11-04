from openpi_client.websocket_client_policy import WebsocketClientPolicy
import torch
import numpy as np
from constant import *

client = WebsocketClientPolicy(host = "0.0.0.0", port= 2251 )

obs = {
    OBS_FRONT_IMAGE: np.full((480, 640, 3), 200, dtype=np.uint8),
    OBS_LEFT_IMAGE: np.full((480, 640, 3), 200, dtype=np.uint8),
    OBS_RIGHT_IMAGE: np.full((480, 640, 3), 200, dtype=np.uint8),
    OBS_LEFT_JOINT: np.full((7,), 0.5),
    OBS_RIGHT_JOINT: np.full((7,), 0.5),
    OBS_PROMPT: 'task'
}
action = client.infer(obs)
