from openpi_client import base_policy as _base_policy

BasePolicy = _base_policy.BasePolicy

import json
import os
import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector

from patch.data_transform import process_data1, process_data2, unnormolize_data
from data_visual import ServerLog

from dataset import EpisodicDataset

DEVICE = torch.device("cuda:0")
ACTION_DIM = 14
proprio_dim = 14
llm_dim = 896


def load_component_state_dict(checkpoint_path):
    state_dict = torch.load(checkpoint_path, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def get_vla(model_path):
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        load_in_4bit=False,
        low_cpu_mem_usage=False,
        trust_remote_code=False,
    )

    vla.vision_backbone.set_num_images_in_input(3)
    vla.eval()
    vla = vla.to(DEVICE)

    return vla


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str):
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            checkpoint_path = os.path.join(pretrained_checkpoint, filename)
            return checkpoint_path


def get_proprio_projector(model_path):
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    )
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()

    checkpoint_path = find_checkpoint_file(model_path, 'proprio_projector')
    state_dict = load_component_state_dict(checkpoint_path)
    proprio_projector.load_state_dict(state_dict)

    return proprio_projector


def get_action_head(model_path):
    action_head = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=ACTION_DIM,
        use_pro_version=True,
        num_task_tokens=768
    )
    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    checkpoint_path = find_checkpoint_file(model_path, 'action_head')
    state_dict = load_component_state_dict(checkpoint_path)
    action_head.load_state_dict(state_dict)

    return action_head

def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    else:
        return obj

class VlaAdapter(BasePolicy):
    def __init__(
            self,
            model_path,
            log_path,
    ):
        self.vla = get_vla(model_path)
        self.proprio_projector = get_proprio_projector(model_path)
        self.action_head = get_action_head(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)

        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        with open(dataset_statistics_path, "r") as f:
            self.stats = json.load(f)

        if log_path:
            self.log = ServerLog(log_path)
        else:
            self.log = None

    def infer(self, obs: dict) -> dict:
        data = process_data1(self.processor, self.stats, obs)
        data = process_data2([data])
        data = to_device(data,0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                input_embeddings = self.vla.get_input_embeddings()(data["input_ids"])
                text_len = input_embeddings.shape[1]
                action_queries = self.vla.action_queries.weight.unsqueeze(0).repeat(input_embeddings.size(0), 1, 1)
                projected_patch_embeddings = self.vla._process_vision_features(data['pixel_values'].to(dtype=torch.bfloat16), None, False)
                input_embeddings = torch.cat(
                    (input_embeddings[:, :1], projected_patch_embeddings, input_embeddings[:, 1:], action_queries),
                    dim=1)

                batch_size, total_seq_len, _ = input_embeddings.shape
                attention_mask = torch.ones(batch_size, total_seq_len, device=DEVICE)

                language_model_output = self.vla.language_model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=input_embeddings,
                    labels=None,
                    use_cache=None,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

                multi_layer_hidden_states = torch.cat(language_model_output.hidden_states, dim=0).unsqueeze(0)
                multi_layer_hidden_states = torch.cat((multi_layer_hidden_states[:, :, :768],
                                                       multi_layer_hidden_states[:, :,
                                                       768 + text_len: 768 + text_len + 64]), dim=2)

                actions = self.action_head.predict_action(multi_layer_hidden_states,
                                                          proprio=data['proprio'],
                                                          proprio_projector=self.proprio_projector)
                actions = actions.reshape(50, ACTION_DIM)
                actions = actions.float().cpu()

        actions = unnormolize_data(self.stats, actions).detach().numpy()
        if self.log:
            self.log.write(obs, actions)
        return {"actions": actions}

    @property
    def metadata(self):
        return {}
