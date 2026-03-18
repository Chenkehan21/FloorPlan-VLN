import os
import json
import random
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from collections import Counter
from collect_navigation_datasets import NavigationDatasetsCollector as NDC


FLOORPLAN_PATH = ""
R2R_FILE_PATH = ""
SAVE_PATH = ""
R2R_STOP_BALANCE_PATH = ""
RXR_FILE_PATH = ""


class ActionProcessor:
    def __init__(self) -> None:
        pass
    
    def pad_locations(self, locations: List, actions: List) -> List:
        padded_locations = []
        padded_locations.append(locations[0]) # start position
        i = 0
        for action in actions:
            if action == 1:
                i += 1
            padded_locations.append(locations[i])
        
        return padded_locations
    
    def chunk_actions(self, actions: List) -> List[List]:
        start = actions[0]
        group, groups = [], []
        for i, action in enumerate(actions):
            if action == start:
                group.append(action)
            else:
                groups.append(group)
                start = action
                group = [action]
            if i == len(actions) - 1:
                groups.append(group)
        
        return groups

    def chunk_by_priority(self, lst: List, priorities: List[int] = [3, 2, 1], ratio: float=0.7) -> List[List]:
        n = len(lst)
        chunks = []
        i = 0
        if random.random() > ratio:
            priorities = [2, 1]
        while i < n:
            for size in priorities:
                if n - i >= size:
                    chunks.append(size)
                    i += size
                    break
                
        return chunks

    def merge_actions(self, actions: List, padded_locations: List, merge_steps: List=[1,2]) -> Tuple[List, List]:
        res_actions, res_action_ids, res_locations = [], [], [padded_locations[0]]
        kept_frames = [0]
        action_space = {
            0: [0],
            1: [1, 4, 5],
            2: [2, 6],
            3: [3, 7],
        }
        current_id = 0
        action_groups = self.chunk_actions(actions)
        for group in action_groups:
            if group[0] == 1:
                merge_choice = self.chunk_by_priority(group, priorities=[3, 2, 1], ratio=0.7)
            else:
                merge_choice = self.chunk_by_priority(group, priorities=[2, 1], ratio=2)
            for merge_step in merge_choice:
                current_id += merge_step
                kept_frames.append(current_id)
                res_action_ids.append(current_id - 1)
                res_actions.append(action_space[actions[res_action_ids[-1]]][merge_step - 1])
                res_locations.append(padded_locations[current_id])
        
        return res_actions, res_locations, kept_frames[:-1]
    
    def identify_location_region(self, floorplan: dict, locations: List, traj_id) -> List:
        regions_height_range, regions_level, regions_boundaries = NDC.extract_regions_info(floorplan)
        traj_region_ids, traj_levels = [], []
        for point in locations:
            point_region, point_level, _ = NDC.check_point_roomtype(point, 
                                                            regions_height_range, 
                                                            regions_boundaries, 
                                                            regions_level)
            traj_region_ids.append(point_region)
            traj_levels.append(point_level)
        assert len(set(traj_levels)) == 1
        traj_level = traj_levels[0]
        traj_region_labels = [floorplan["levels"][traj_level]["regions"][i]["label"] for i in traj_region_ids]
        
        return list(zip(traj_region_ids, traj_region_labels))
        
    def create_merged_dataset(self, stop_balance_path: str, file_path: str, name: str, split: str="train", 
                                    merge_action: bool=False, stop_balancing: bool=True, plot: bool=True) -> None:
        with open(os.path.join(file_path, f"{name}_{split}_gt.json"), "r") as f:
            gt_data = json.load(f)
        
        with open(os.path.join(file_path, f"{name}_{split}.json"), "r") as f:
            episodes = json.load(f)["episodes"]
        
        all_actions = []
        merged_gt_data = {}
        merged_trajs = {}
        for episode_id, episode_gt in tqdm(gt_data.items()):
            if "rxr" in name:
                episode = episodes[eval(episode_id) - 9002]
            else:
                episode = episodes[eval(episode_id)]
            assert episode["episode_id"] == eval(episode_id)
            traj_id = str(episode["trajectory_id"])
            if traj_id not in merged_trajs:
                merged_trajs[traj_id] = []
            scene = episode["scene_id"].split('/')[1]
            
            actions = episode_gt["actions"]
            locations = episode_gt["locations"]
            padded_locations = self.pad_locations(locations, actions)
            if merge_action and len(merged_trajs[traj_id]) == 0:
                merged_actions, merged_locations, merged_action_ids = self.merge_actions(actions, padded_locations)
                if stop_balancing:
                    stop_balance_steps = os.listdir(os.path.join(stop_balance_path, split, scene, traj_id))
                    stop_num = min(len(stop_balance_steps), 5)
                    merged_actions += [0] * len(stop_balance_steps[:stop_num])
                    merged_action_ids += [merged_action_ids[-1]] * len(stop_balance_steps[:stop_num])
                merged_trajs[traj_id].append((merged_actions, merged_locations, merged_action_ids))
            elif merge_action and len(merged_trajs[traj_id]) > 0:
                merged_actions, merged_locations, merged_action_ids = merged_trajs[traj_id][0]
            
            all_actions += merged_actions
            
            with open(os.path.join(FLOORPLAN_PATH, scene, "floorplan.json"), "r") as f:
                floorplan = json.load(f)
            traj_regions = self.identify_location_region(floorplan, merged_locations, traj_id)
        
            merged_gt_data[episode_id] = {
                "locations": merged_locations,
                "actions": merged_actions,
                "frame_ids": merged_action_ids,
                "regions": traj_regions
            }
        
        with open(os.path.join(SAVE_PATH, f"combine_actions_{name}_{split}_gt.json"), "w") as f:
            json.dump(merged_gt_data, f, indent=2)
        print(len(all_actions))
        if plot:
            actions_counter = Counter(all_actions)
            print(actions_counter)
            action_space = list(actions_counter.keys())
            action_space.sort()
            action_distribution = [actions_counter[i] for i in action_space]
            plt.pie(action_distribution,
                    labels=action_space,
                    autopct="%.2f%%",
                    radius=1, 
                    labeldistance=1,
                    # colors=["#a75f9d", "#007dcc", "#008c64"],
                    # explode=[0,0.1,0],
                    textprops={'fontsize': '12'},
                    wedgeprops=dict(width=1,edgecolor='white'))
            plt.savefig("./rxr_merged_action_distribution.png")
    
    def create_merged_dagger_dataset(self, stop_balance_path: str, file_path: str, name: str, split: str="train", 
                                    merge_action: bool=False, stop_balancing: bool=True, plot: bool=True) -> None:
        with open(os.path.join(file_path, f"{name}_{split}_gt.json"), "r") as f:
            gt_data = json.load(f)
        
        with open(os.path.join(file_path, f"{name}_{split}.json"), "r") as f:
            episodes = json.load(f)["episodes"]
        
        all_actions = []
        merged_gt_data = {}
        merged_trajs = {}
        for episode_id, episode_gt in tqdm(gt_data.items()):
            if "rxr" in name:
                episode = episodes[eval(episode_id) - 9002]
            else:
                episode = episodes[eval(episode_id)]
            assert episode["episode_id"] == eval(episode_id)
            traj_id = str(episode["trajectory_id"])
            if traj_id not in merged_trajs:
                merged_trajs[traj_id] = []
            scene = episode["scene_id"].split('/')[1]
            
            actions = episode_gt["actions"]
            locations = episode_gt["locations"]
            padded_locations = self.pad_locations(locations, actions)
            if merge_action and len(merged_trajs[traj_id]) == 0:
                merged_actions, merged_locations, merged_action_ids = self.merge_actions(actions, padded_locations)
                if stop_balancing:
                    stop_balance_steps = os.listdir(os.path.join(stop_balance_path, split, scene, traj_id))
                    stop_num = min(len(stop_balance_steps), 5)
                    merged_actions += [0] * len(stop_balance_steps[:stop_num])
                    merged_action_ids += [merged_action_ids[-1]] * len(stop_balance_steps[:stop_num])
                merged_trajs[traj_id].append((merged_actions, merged_locations, merged_action_ids))
            elif merge_action and len(merged_trajs[traj_id]) > 0:
                merged_actions, merged_locations, merged_action_ids = merged_trajs[traj_id][0]
            
            all_actions += merged_actions
            
            with open(os.path.join(FLOORPLAN_PATH, scene, "floorplan.json"), "r") as f:
                floorplan = json.load(f)
            traj_regions = self.identify_location_region(floorplan, merged_locations, traj_id)
        
            merged_gt_data[episode_id] = {
                "locations": merged_locations,
                "actions": merged_actions,
                "frame_ids": merged_action_ids,
                "regions": traj_regions
            }
        
        with open(os.path.join(SAVE_PATH, f"combine_actions_{name}_{split}_gt.json"), "w") as f:
            json.dump(merged_gt_data, f, indent=2)
        print(len(all_actions))
        if plot:
            actions_counter = Counter(all_actions)
            print(actions_counter)
            action_space = list(actions_counter.keys())
            action_space.sort()
            action_distribution = [actions_counter[i] for i in action_space]
            plt.pie(action_distribution,
                    labels=action_space,
                    autopct="%.2f%%",
                    radius=1, 
                    labeldistance=1,
                    # colors=["#a75f9d", "#007dcc", "#008c64"],
                    # explode=[0,0.1,0],
                    textprops={'fontsize': '12'},
                    wedgeprops=dict(width=1,edgecolor='white'))
            plt.savefig("./rxr_merged_action_distribution.png")


if __name__ == "__main__":
    random.seed(1)
    action_processor = ActionProcessor()
    config = {
        "stop_balance_path": R2R_STOP_BALANCE_PATH, 
        "file_path": RXR_FILE_PATH, 
        "name": "rxr_from_merge", 
        "split": "train", 
        "merge_action": True, 
        "stop_balancing": False, 
        "plot": True
    }
    action_processor.create_merged_dataset(**config)