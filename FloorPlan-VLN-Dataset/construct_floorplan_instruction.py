import os
import copy
import json
import random
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


VLN_DATASET_PATH = ""
QWEN_LOCAL_PATH = "models"
FLOORPLAN_PATH = "data/mp3d_floorplan"


class InstructionCreater:
    def __init__(self) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(QWEN_LOCAL_PATH)
        
    def create_floorplan_description(self, scan, level_id):
        fp_path = os.path.join(FLOORPLAN_PATH, scan, "floorplan.json")
        with open(fp_path, 'r') as f:
            floorplan = json.load(f)
        
        lines = []
        regions = floorplan["levels"][str(level_id)]["regions"]
        for region_id, region in regions.items():
            region_type = region["label"]
            center = region["center"]
            connectivity = region["connectivity"]
            line = f"- Region {region_id}: id={region_id}, type='{region_type}', center='{center}', connected_to={connectivity}"
            lines.append(line)
        res = "\n".join(lines)
        
        return res
    
    def reformulate_raw_vln_instr(self, task_name: str, split: str, add_template):
        fn_path = os.path.join(VLN_DATASET_PATH, task_name, f"{task_name}_{split}.json")
        with open(fn_path, 'r') as f:
            dataset = json.load(f)
        episodes = dataset["episodes"]
        new_episodes = []
        exist_trajs = []
        print(len(episodes))
        # random.shuffle(episodes)
        for episode in tqdm(episodes):
            traj_id = episode["trajectory_id"]
            if traj_id not in exist_trajs:
                exist_trajs.append(traj_id)
                
                level = episode["level"]
                scan = episode["scene_id"].split('/')[1]
                instruction = episode["instruction"]
                traj_regions = episode["traj_regions"]
                start_region = traj_regions[0]
                end_region = traj_regions[-1]
                new_instr = self.qwen_instr(instruction, start_region, end_region)
                if add_template:
                    floorplan_description = self.create_floorplan_description(scan, level)
                    new_instr = f"<image>\n<video>\nImagine you are a robot programmed for navigation tasks. You have been given a image of the floorplan a navigation video and the floorplan sturcture description:\n{floorplan_description}.\nYour assigned task is: '{new_instr}'. Decide your next action, which could involve turning left or right by a specific degree or moving forward a certain distance."
                new_episode = copy.deepcopy(episode)
                new_episode["instruction"] = new_instr
                new_episodes.append(new_episode)
        print(len(new_episodes))
        save_path = os.path.join(VLN_DATASET_PATH, task_name, f"floorplan_vln_instructions_{task_name}_{split}.json")
        with open(save_path, 'w') as f:
            json.dump(new_episodes, f, indent=1)
            
    def qwen_instr(self, instruction: str, start_region:str, end_region:str):
        answer_templates = [
            f"You are now at ({start_region[0]}, {start_region[1]}), please go to ({end_region[0]}, {end_region[1]}), and ",
            f"From ({start_region[0]}, {start_region[1]}), make your way to ({end_region[0]}, {end_region[1]}), and ",
            f"Head from ({start_region[0]}, {start_region[1]}) to ({end_region[0]}, {end_region[1]}), and ",
            f"Start at ({start_region[0]}, {start_region[1]}), proceed to ({end_region[0]}, {end_region[1]}), and ",
            f"Make your way from ({start_region[0]}, {start_region[1]}) to ({end_region[0]}, {end_region[1]}), and ", 
            f"Navigate from ({start_region[0]}, {start_region[1]}) toward ({end_region[0]}, {end_region[1]}), and ",
            f"Begin at ({start_region[0]}, {start_region[1]}), move to ({end_region[0]}, {end_region[1]}), and ",
            f"Travel from ({start_region[0]}, {start_region[1]}) to ({end_region[0]}, {end_region[1]}), and ",
            f"Departing from ({start_region[0]}, {start_region[1]}), reach ({end_region[0]}, {end_region[1]}), and ",
            f"Go from ({start_region[0]}, {start_region[1]}) to ({end_region[0]}, {end_region[1]}), "
        ]
        answer_template = random.sample(answer_templates, 1)[0]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are given a raw navigation instruction from a Vision-and-Language Navigation (VLN) task with the identifier of [RAW INSTRUCTION].\nYou task is to: Extract from the original instruction the most specific description of the final stopping point (e.g., objects, landmarks, or scene details).\nExamples: \nUser: [RAW INSTRUCTION]: Go past the small couch and turn slightly right to go into the hallway, pass the glass doors. Then head straight into the first room on the right. Then turn left and wait in the hallway that is painted beige and has two paintings one on each side of a piece of furniture on the left.\nYou: stop beside the furniture that has a painting on both sides.\nAttention: Output ONLY the generated instruction, without any additional explanation or text.\nLet's start:\n[RAW INSTRUCTION]: {instruction}\nYou:"
                    },
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        new_instruction = answer_template + output_text
        # print(f"raw instruction: {instruction}\nnew isntruction: {new_instruction}\n\n")
        
        return new_instruction


if __name__ == "__main__":
    IC = InstructionCreater()
    IC.reformulate_raw_vln_instr(task_name="rxr_from_merge", split="val_unseen", add_template=False)