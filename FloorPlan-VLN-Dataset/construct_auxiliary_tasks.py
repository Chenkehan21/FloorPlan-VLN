import os
import copy
import json
import time
import random
from PIL import Image
from tqdm import tqdm
from itertools import groupby

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from decord import VideoReader, cpu


VLN_VIDEO_PATH = "data/videos_split"
VLN_DATASET_PATH = "data/navigation_datasets"
FLOORPLAN_PATH = "data/mp3d_floorplan_graph"
QA_DATASET_PATH = "data/QA_datasets"
QWEN_LOCAL_PATH = "models"


class Reasoner:
    def __init__(self) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(QWEN_LOCAL_PATH)
        self.processor.tokenizer.padding_side = 'left'
        
    def pad_step_region(self, traj_regions, actions):
        step_regions = [traj_regions[0]]
        n = 1
        for i, act in enumerate(actions):
            if act == 1:
                step_regions.append(traj_regions[n])
                n += 1
            else:
                step_regions.append(step_regions[i])
        
        return step_regions

    def define_next_explore_region(self, explore_plan, n):
        if n + 1 < len(explore_plan):
            next_explore_region = explore_plan[n + 1]
        else:
            next_explore_region = explore_plan[n]
        
        return next_explore_region
    
    def get_explored_description(self, explored):
        regions = []
        for item in explored:
            if item not in regions:
                regions.append(item)
        explored_description = ""
        for i, item in enumerate(regions):
            if i == len(explored) - 1:
                explored_description += f"({item[0]}, {item[1]})."
            else:
                explored_description += f"({item[0]}, {item[1]}), "
        
        return explored_description
    
    def describe_explore_plan(self, region_explore_plan):
        res = f"According to the instruction and floorplan, I plan to explore the regions in this order: "
        for i, item in enumerate(region_explore_plan):
            if i == len(region_explore_plan) - 1:
                res += f"({item[0]}, {item[1]})."
            else:
                res += f"({item[0]}, {item[1]}), "
        
        return res
    
    def create_obs_caption(self, frame_img_batch: Image, step_region_types: str):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": obs,
                        },
                        {"type": "text", 
                        "text": f"You are a navigation robot and now in the {current_region_type}. Describe your observation briefly and concisely. Only describe the elements that you can clearly recognize and are certain about. If parts of the image are blurry, ambiguous, or hard to interpret, ignore them completely. Avoid speculation or assumptions. Focus solely on what you are most confident in identifying. Finally prove that you are in the {current_region_type}. You should answer like: 'I can see <your descriptions> so I may be in the {current_region_type}.'"},
                    ],
                },
            ]
            for obs, current_region_type in zip(frame_img_batch, step_region_types)
        ]
        texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                 for message in messages]
        image_input_list, video_input_list = [], None
        for message in messages:
            image_inputs, video_inputs = process_vision_info(message)
            image_input_list.append(image_inputs[0])
            
        inputs = self.processor(
            text=texts,
            images=image_input_list,
            videos=video_input_list,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        
        return output_text
    
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
    
    def create_qa_sample(self, episode_id, video_path, floorplan_path, level, 
                         instruction, action, floorplan_description, step_reasoning):
        user_template = f"<image>\n<video>\nImagine you are a robot programmed for navigation tasks. You have been given a image of the floorplan a navigation video and the floorplan sturcture description:\n{floorplan_description}.\nYour assigned task is: '{instruction}'. Decide your next action, which could involve turning left or right by a specific degree or moving forward a certain distance."
        
        gpt_response_template = "The next action is"
        action2instruction = {
            "0": "stop.",
            "1": "move forward 25 cm.",
            "2": "turn left 15 degrees.",
            "3": "turn right 15 degrees.",
            "4": "move forward 50 cm.",
            "5": "move forward 75 cm.",
            "6": "turn left 30 degrees.",
            "7": "turn left 45 degrees.",
            "8": "turn right 30 degrees.",
            "9": "turn right 45 degrees."
        }
        action_description = action2instruction[str(action)]
        item = {
                "id": episode_id,
                "video": video_path, # navigation video
                "floorplan": floorplan_path, # floorplan image
                "level": level,
                "conversations": [
                    {
                        "from": "human",
                        "value": user_template.format(instruction.strip()),
                    },
                    {
                        "from": "gpt",
                        "value": f"{step_reasoning} <Action>{gpt_response_template} {action_description}</Action>"
                    }
                ]
            }

        return item
    
    def split_list_into_batches(self, list, batchsize):
        return [list[i : i + batchsize] for i in range(0, len(list), batchsize)]
    
    def calculate_total_steps(self, task_name: str, split: str,):
        gt_fp = os.path.join(VLN_DATASET_PATH, task_name, f"{task_name}_{split}_gt.json")
        with open(gt_fp, 'r') as f:
            gt = json.load(f)
        gt
        res = 0
        all_actions = []
        max_steps = -1
        for i, item in enumerate(gt.values()):
            if i % 3 == 0:
                actions = item["actions"]
                all_actions += actions
                res += len(actions)
                if len(actions) > max_steps:
                    max_steps = len(actions)
        print(res, max_steps, len(all_actions))
        return res
    
    def create_cot(self, task_name: str, task_name_video: str, split: str, batchsize:int):
        dataset = []
        eps_fp = os.path.join(VLN_DATASET_PATH, task_name, f"floorplan_vln_instructions_{task_name}_{split}.json")
        with open(eps_fp, 'r') as f:
            episodes = json.load(f)
        
        gt_fp = os.path.join(VLN_DATASET_PATH, task_name, f"{task_name}_{split}_gt.json")
        with open(gt_fp, 'r') as f:
            gt = json.load(f)
        
        # random.shuffle(episodes)
        processed_trajs = {}
        start_idx = 0
        for idx, episode in enumerate(tqdm(episodes)):
            level = episode["level"]
            eps_id = episode["episode_id"]
            traj_id = episode["trajectory_id"]
            instruction = episode["instruction"]
            scan = episode["scene_id"].split('/')[1]
            traj_regions = episode["traj_regions"]
            target_region = traj_regions[-1]
            stop_description = instruction.split(',')[-1].strip()
            if stop_description.lower().startswith('and'):
                stop_description = stop_description[3:].strip()
            
            floorplan_description = self.create_floorplan_description(scan, level)
            
            region_explore_plan = [key for key, _ in groupby(traj_regions)]
            navigation_plan = self.describe_explore_plan(region_explore_plan)
            
            actions = gt[str(eps_id)]["actions"]
            step_regions = self.pad_step_region(traj_regions, actions)[:-1]
            
            print(f"================ episode: {eps_id} traj: {traj_id} actions: {len(actions)}================")
            
            video_fp = os.path.join(VLN_VIDEO_PATH, task_name_video, split, scan, str(traj_id), f"{len(actions) - 1}.mp4")
            vr = VideoReader(video_fp, ctx=cpu(0))
            total_frames = len(vr)
            assert total_frames == len(step_regions)
            last_region = None
            explored = []
            n = 0
            
            bs = min(batchsize, total_frames)
            frame_images = [Image.fromarray(vr[i].asnumpy()) for i in range(total_frames)]
            frame_img_batch = self.split_list_into_batches(frame_images, bs)
            step_regions_batch = self.split_list_into_batches(step_regions, bs)
            frame_index_batch = self.split_list_into_batches(list(range(total_frames)), bs)
            action_batch = self.split_list_into_batches(actions, bs)
            if traj_id not in processed_trajs:
                processed_trajs[traj_id] = []
                caption_batch = None
            else:
                caption_batch = processed_trajs[traj_id]
            for i, batch in enumerate(frame_img_batch):
                actions_i = action_batch[i]
                frame_index_i = frame_index_batch[i]
                step_regions_i = step_regions_batch[i]
                step_region_types_i = [item[1] for item in step_regions_i]
                # import pdb;pdb.set_trace()
                if len(processed_trajs[traj_id]) < len(action_batch):
                    print("qwen captionning")
                    captions = self.create_obs_caption(batch, step_region_types_i)
                    processed_trajs[traj_id].append(captions)
                else:
                    captions = caption_batch[i]
                assert len(actions_i) == len(frame_index_i) == len(step_regions_i) == len(captions)
                for action, frame_index, caption, region_id_type in zip(actions_i, frame_index_i, captions, step_regions_i):
                    if last_region is None or (region_id_type == last_region and n == 0):
                        next_explore_region = self.define_next_explore_region(region_explore_plan, n)
                        explore_plan = f"I haven't explored any regions. I am exploring ({region_id_type[0]}, {region_id_type[1]}) and I need to explore ({next_explore_region[0]}, {next_explore_region[1]}) next."
                        last_region = region_id_type
                        explored.append(region_id_type)
                    else:
                        if region_id_type != last_region:
                            n += 1
                            last_region = region_id_type
                            explored.append(region_id_type)
                        explored_description = self.get_explored_description(explored[:-1])
                        if region_id_type == target_region:
                            explore_plan = f"I have explored {explored_description} I am exploring ({region_id_type[0]}, {region_id_type[1]}) and I need to {stop_description}"
                        else:
                            next_explore_region = self.define_next_explore_region(region_explore_plan, n)
                            explore_plan = f"I have explored {explored_description} I am exploring ({region_id_type[0]}, {region_id_type[1]}) and I need to explore ({next_explore_region[0]}, {next_explore_region[1]}) next."
                    reasoning = f"<Plan>{navigation_plan}</Plan> <Think>{caption} {explore_plan}</Think>"
                    
                    floorplan_path = os.path.join(FLOORPLAN_PATH, scan, f"floorplan_region_graph{level}.png")
                    video_path = os.path.join(VLN_VIDEO_PATH, task_name_video, split, scan, str(traj_id), f"{frame_index}.mp4")
                    sample = self.create_qa_sample(eps_id, video_path, floorplan_path, level, 
                                                instruction, action, floorplan_description, reasoning)
                    dataset.append(sample)
            
            if idx % 500 == 0 or idx == len(episodes) - 1:
                with open(os.path.join(QA_DATASET_PATH, f"{task_name}_{split}_qwen_{start_idx}_{idx}.json"), 'w') as f:
                    json.dump(dataset, f, indent=1)
                dataset = []
                start_idx = idx + 1
                torch.cuda.empty_cache()
    

if __name__ == "__main__":
    reasoner = Reasoner()
    reasoner.calculate_total_steps(task_name="r2r_from_merge", split="train")
    # reasoner.create_cot(task_name="r2r_from_merge", task_name_video="r2r", split="val_unseen", batchsize=256)