import os
import re
import cv2
import json
import copy
import math
import time
import torch
import random
import imageio
import numpy as np
from PIL import Image
from tqdm import trange
from typing import List, Dict, Sequence
from scipy.spatial.transform import Rotation as R

from habitat import Env
from habitat.core.agent import Agent
from habitat.utils.visualizations import maps

from transformers.feature_extraction_utils import BatchFeature
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from floorplan_nav import FloorplanNavigator


FLOORPLAN_PATH = "data/mp3d_floorplan_graph"


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Qwen_Agent(Agent):
    def __init__(self, model_path, result_path, require_map=False):
        print("Initialize Qwen")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = 'left'
        
        self.floorplan_navigator = FloorplanNavigator()
        
        self.result_path = result_path
        self.require_map = require_map
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        self.nav_list = []
        self.topdown_map_list = []

        self.count_id = 0
        self.reset()
    
    def get_heading(self, rotation):
        r = R.from_quat(rotation)
        angle_radians = r.as_euler('zyx', degrees=False)
        angle_degrees = r.as_euler('zyx', degrees=True)
        change_direction = (angle_degrees[0] == -180 and angle_degrees[2] == 180)
        if change_direction:
            heading = -1 * angle_radians[1] - math.pi * 0.5
        else:
            heading = angle_radians[1] + math.pi * 0.5
            
        return heading
    
    def preprocess_qwen_2_visual(
        self,
        content,
        tokenizer,
        grid_thw_image: List = [],
        grid_thw_video: List = [],
    ) -> Dict:
        system_message = "You are a helpful assistant."

        tokenizer = copy.deepcopy(tokenizer)
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        visual_replicate_index_image = 0
        visual_replicate_index_video = 0
        input_ids = []

        input_ids += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )

        role = "user"
        if "<image>" in content:
            parts = content.split("<image>")
            new_parts = []
            for i in range(len(parts) - 1):
                new_parts.append(parts[i])
                replacement = (
                    "<|vision_start|>"
                    + f"<|image_pad|>"
                    * grid_thw_image[visual_replicate_index_image] # grid_thw_image=[tensor(64)]
                    + "<|vision_end|>"
                )
                new_parts.append(replacement)
                visual_replicate_index_image += 1
            new_parts.append(parts[-1])
            content = "".join(new_parts)
        if "<video>" in content:
            parts = content.split("<video>")
            new_parts = []
            for i in range(len(parts) - 1):
                new_parts.append(parts[i])
                replacement = (
                    "<|vision_start|>"
                    + f"<|video_pad|>"
                    * grid_thw_video[visual_replicate_index_video] # grid_thw_video=[tensor(100)]
                    + "<|vision_end|>"
                )
                new_parts.append(replacement)
                visual_replicate_index_video += 1
            new_parts.append(parts[-1])
            content = "".join(new_parts)

        conv = [{"role": role, "content": content}]
        encode_id = tokenizer.apply_chat_template(conv)
        input_ids += encode_id
        input_ids = torch.tensor([input_ids], dtype=torch.long)
            
        return dict(
            input_ids=input_ids,
        )
    
    def process_video_frames(self, video, frame_idx, video_length, processor):
        image_processor = copy.deepcopy(processor.image_processor)
        image_processor.max_pixels = 352800
        image_processor.min_pixels = 200704
        image_processor.size["longest_edge"] = image_processor.max_pixels # 25088
        image_processor.size["shortest_edge"] = image_processor.min_pixels # 3136
        fps = len(frame_idx) / video_length
        video_processed = image_processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        
        return video_tensor, grid_thw, second_per_grid_ts

    def process_inputs_navid(self, nav_list, instruction, processor):
        total_frames = len(nav_list)
        avg_fps = 1
        video_length = total_frames / avg_fps
        interval = 1

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = 1
        video_max_frames = 6

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = [nav_list[i] for i in frame_idx]
        
        video, video_grid_thw, second_per_grid_ts = self.process_video_frames(video, frame_idx, video_length, processor)
        video = [video]
        
        video_grid_thw_merged = copy.deepcopy(video_grid_thw)
        if not isinstance(video_grid_thw, Sequence):
            video_grid_thw_merged = [video_grid_thw_merged]
            video_grid_thw = [video_grid_thw]
        video_grid_thw_merged = [
            merged_thw.prod() // processor.image_processor.merge_size**2
            for merged_thw in video_grid_thw_merged
        ]
        
        content = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video and your assigned task is: {instruction}. Decide your next action, which could involve turning left or right by a specific degree or moving forward a certain distance or stop."
        inputs = self.preprocess_qwen_2_visual(
            content,
            processor.tokenizer,
            grid_thw_image=None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["pixel_values_videos"] = torch.cat(video, dim=0)
        inputs["video_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in video_grid_thw], dim=0)
        inputs["second_per_grid_ts"] = second_per_grid_ts
        
        return inputs
    
    def predict_inference(self, instruction, navigation_video):
        inputs = self.process_inputs_navid(navigation_video, instruction, self.processor)
        inputs = BatchFeature(inputs)
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 3, float(match)

        return None, None

    def addtext(self, image, instruction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instruction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instruction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line
        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def reset(self):
        if self.require_map:
            if len(self.topdown_map_list)!=0:
                output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))
                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.transformation_list = []
        self.nav_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.first_forward = False
        
    def act(self, observations, concat_img, info, episode_id):
        self.episode_id = episode_id
        rgb = observations["rgb"]

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]['text'], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            return {"action": temp_action}
        self.nav_list.append(concat_img)
        instruction = observations["instruction"]['text']
        navigation = self.predict_inference(instruction, self.nav_list)
        action_index, num = self.extract_result(navigation)
        # print(f"{episode_id}: {instruction}\n{navigation}\n{action_index}, {num}")
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]['text'], navigation)
            self.topdown_map_list.append(img)
            
        if action_index == 0:
            self.pending_action_list.append(0)
        elif action_index == 1:
            for _ in range(min(3, int(num/25))):
                self.pending_action_list.append(1)

        elif action_index == 2:
            for _ in range(min(3,int(num/15))):
                self.pending_action_list.append(2)

        elif action_index == 3:
            for _ in range(min(3,int(num/15))):
                self.pending_action_list.append(3)
        
        if action_index is None or len(self.pending_action_list)==0:
            self.pending_action_list.append(random.randint(1, 3))
            # Primarily unused, intended to complete the pipeline logic.
        
        return {"action": self.pending_action_list.pop(0)}


def add_mask(image, mask_ratio=0.25):
    h, w, c = image.shape
    patch_size = 32
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    num_patches = num_patches_h * num_patches_w
    num_mask = int(mask_ratio * num_patches)
    mask_indices = random.sample(range(num_patches), num_mask)
    masked_image = image.copy()
    for idx in mask_indices:
        i = idx // num_patches_w
        j = idx % num_patches_w
        masked_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :] = 255
    return masked_image

def add_mask_to_floorplan_nav(frame, mask_ratio=0.25):
    assert frame.shape[1] == 448 * 2
    floorplan_nav = frame[:, 448:, :]
    noised_fp_nav = add_mask(floorplan_nav, mask_ratio=mask_ratio)
    frame[:, 448:, :] = noised_fp_nav
    
    return frame


def evaluate_agent(config, split_id, dataset, model_path, result_path) -> None:
    env = Env(config.TASK_CONFIG, dataset)
    agent = Qwen_Agent(model_path, result_path)
    num_episodes = len(env.episodes)
    
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS
    
    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}
    count = 0
    all_scenes = os.listdir(FLOORPLAN_PATH)
    
    for i in trange(num_episodes, desc=config.EVAL.IDENTIFICATION+"-{}".format(split_id), ncols=80):
        obs = env.reset()
        iter_step = 0
        agent.reset()
        scene_id = env.current_episode.scene_id.split('/')[-2]
        level = env.current_episode.level
        with open(os.path.join(FLOORPLAN_PATH, scene_id, "floorplan.json"), "r") as f:
            floorplan = json.load(f)
        traj = []
        state = env.sim.get_agent_state()
        position = state.position.tolist()
        traj.append(position)
        rotation = state.rotation
        rotation = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
        heading = agent.get_heading(rotation)
        floorplan_nav_img = agent.floorplan_navigator.plot_floorplan_traj(floorplan, level, np.array(traj)[:1], heading)
        floorplan_nav_img = floorplan_nav_img[..., ::-1]
        concat_img = cv2.hconcat([obs['rgb'], floorplan_nav_img])
         
        continuse_rotation_count = 0
        last_dtg = 999
        while not env.episode_over:
            info = env.get_metrics()
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            action = agent.act(obs, concat_img, info, env.current_episode.episode_id)
            if continuse_rotation_count > EARLY_STOP_ROTATION: 
                print(f"{env.current_episode.episode_id} early stop rotation")
                action = {"action": 0}
            if iter_step > EARLY_STOP_STEPS:
                print("early stop step")
                action = {"action": 0}
            iter_step+=1
            obs = env.step(action)
            
            state = env.sim.get_agent_state()
            position = state.position.tolist()
            traj.append(position)
            if action['action'] == 2:
                heading += math.radians(15)
            elif action['action'] == 3:
                heading -= math.radians(15)
            floorplan_nav_img = agent.floorplan_navigator.plot_floorplan_traj(floorplan, level, np.array(traj), heading)
            floorplan_nav_img = floorplan_nav_img[..., ::-1]
            concat_img = cv2.hconcat([obs['rgb'], floorplan_nav_img])
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1
        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)