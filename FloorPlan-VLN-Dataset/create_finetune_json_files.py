import os
import re
import json
import random
from copy import deepcopy
from collections import defaultdict, Counter


FLOORPLAN_IMAGE = "mp3d_floorplan"


def raw_instr_dataset(data_path, save_path):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    for item in data:
        video_fp = item['video']
        new_video_fp = video_fp.replace("videos_action_balance", "concate_floorplan_navigation_video")
        item['video'] = new_video_fp

    with open(save_path, "w") as f:
        json.dump(data, f)


def extract_raw_instruction(text):
    match = re.search(r"your assigned task is:\s*'([^']*)'", text.lower())
    if match:
        instruction = match.group(1)
    else:
        instruction = ""
    
    return instruction


def extract_raw_rxr_instruction(text):
    pattern = r"your assigned task is:(.*?)(?=Decide your next action)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        result = match.group(1).strip()
    else:
        result = ""
    
    return result


def extract_plan_part(answer):
        match = re.search(r"<Plan>(.*?)</Plan>", answer, re.DOTALL)
        if match:
            plan = match.group(1).strip()
        else:
            plan = ""
        
        return plan


def extract_think_part(answer):
        match = re.search(r"<Think>(.*?)</Think>", answer, re.DOTALL)
        if match:
            think = match.group(1).strip()
            localizations = []
            summarys = []
            for item in think.split('. '):
                if "explor" in item and "I " in item:
                    summarys.append(item)
                else:
                    localizations.append(item)
            localization = '. '.join(localizations)
            summary = '. '.join(summarys)
        else:
            think = ""
            localization = ""
            summary = ""
        
        return localization, summary


def extract_action_part(answer):
        match = re.search(r"<Action>(.*?)</Action>", answer, re.DOTALL)
        if match:
            action = match.group(1).strip()
        else:
            action = ""
        
        return action


def fp_instr_dataset(data_v1_path, data_v2_path, save_path):
    with open(data_v1_path, "r") as f:
        data_v1 = json.load(f)

    with open(data_v2_path, "r") as f:
        data_v2 = json.load(f)
    print(len(data_v1), len(data_v2))
    
    for i, item in enumerate(data_v1):
        video_v1 = '/'.join(item['video'].split('/')[-5:])
        video_v2 = data_v2[i]['video']
        path = video_v2.split('/')
        video_name = path[-1].split('.')[0].split('_')[0]
        path[-1] = f'{video_name}.mp4'
        video_v2 = '/'.join(path[-5:])
        assert video_v1 == video_v2
        instr2 = extract_raw_instruction(data_v2[i]['conversations'][0]['value'])
        item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video and your assigned task is: {instr2} Decide your next action, which could involve turning left or right by a specific degree or moving forward a certain distance or stop."
    
    with open(save_path, "w") as f:
        json.dump(data_v1, f)
        

def fp_cot_instr_dataset(data_v1_path, data_v2_path, save_path):
    with open(data_v1_path, "r") as f:
        data_v1 = json.load(f)

    with open(data_v2_path, "r") as f:
        data_v2 = json.load(f)
    print(len(data_v1), len(data_v2))
    
    for i, item in enumerate(data_v1):
        video_v1 = '/'.join(item['video'].split('/')[-5:])
        video_v2 = data_v2[i]['video']
        path = video_v2.split('/')
        video_name = path[-1].split('.')[0].split('_')[0]
        path[-1] = f'{video_name}.mp4'
        video_v2 = '/'.join(path[-5:])
        assert video_v1 == video_v2
        instr2 = extract_raw_instruction(data_v2[i]['conversations'][0]['value'])
        answer2 = data_v2[i]['conversations'][1]['value']
        item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Your assigned task is: {instr2} Decide your next action, which could involve turning left or right by a specific degree or moving forward a certain distance or stop."
        if "<Action>The next action is stop.</Action>" not in answer2:
            item['conversations'][1]['value'] = answer2
        else:
            answer2_stop = answer2.replace("The next action is stop", "I think I should stop because I have finished the instruction")
            item['conversations'][1]['value'] = answer2_stop
    
    with open(save_path, "w") as f:
        json.dump(data_v1, f)


def fp_aux_instr_dataset(data_v1_path, data_v2_path, save_path):
    with open(data_v1_path, "r") as f:
        data_v1 = json.load(f)
        
    with open(data_v2_path, "r") as f:
        data_v2 = json.load(f)
    
    res = []
    count_action, count_plan, count_localization, count_summary, count_inst = 0, 0, 0, 0, 0
    for i, item in enumerate(data_v2):
        res.append(data_v1[i])
        count_action += 1
        
        video_v1 = '/'.join(data_v1[i]['video'].split('/')[-5:])
        video_v2 = item['video']
        path = video_v2.split('/')
        aug_video_namev2 = path[-1].split('.')[0]
        video_name = path[-1].split('.')[0].split('_')[0]
        path[-1] = f'{video_name}.mp4'
        video_v2 = '/'.join(path[-5:])
        assert video_v1 == video_v2
        
        instr2 = extract_raw_instruction(item['conversations'][0]['value'])
        answer = item['conversations'][1]['value']
        plan = extract_plan_part(answer)
        localization, summary = extract_think_part(answer)
        
        if video_name == "0":
            plan_item = deepcopy(data_v1[i])
            plan_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Your assigned task is: {instr2} Describe your navigation plan."
            plan_item['conversations'][1]['value'] = plan
            res.append(plan_item)
            count_plan += 1
        
        if random.random() <= 0.50:
            localization_item = deepcopy(data_v1[i])
            localization_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Describe your location."
            localization_item['conversations'][1]['value'] = localization
            count_localization += 1
            
            summary_item = deepcopy(data_v1[i])
            summary_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Your assigned task is: {instr2} Summarize your navigation process and describe your next plan."
            summary_item['conversations'][1]['value'] = summary
            count_summary += 1
            
            res.append(localization_item)
            res.append(summary_item)
        
        if "I think I should stop" in data_v1[i]['conversations'][1]['value'] and "_" not in aug_video_namev2:
            instruction_reasoning_item = deepcopy(data_v1[i])
            instruction_reasoning_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Based on this video, describe the navigation trajctory."
            instruction_reasoning_item['conversations'][1]['value'] = instr2
            res.append(instruction_reasoning_item)
            count_inst += 1
    
    print(len(res))
    print(f"count_action: {count_action}\nplan: {count_plan}\nlocalization: {count_localization}\nsummary: {count_summary}\ninstruction reasoning: {count_inst}")
    with open(save_path, "w") as f:
        json.dump(res, f)

def fp_aux_instr_ablation_dataset(data_v1_path, data_v2_path, save_path):
    with open(data_v1_path, "r") as f:
        data_v1 = json.load(f)

    with open(data_v2_path, "r") as f:
        data_v2 = json.load(f)
    
    res = []
    count_action, count_plan, count_localization, count_summary, count_inst = 0, 0, 0, 0, 0
    for i, item in enumerate(data_v2):
        res.append(data_v1[i])
        count_action += 1
        
        video_v1 = '/'.join(data_v1[i]['video'].split('/')[-5:])
        video_v2 = item['video']
        path = video_v2.split('/')
        aug_video_namev2 = path[-1].split('.')[0]
        video_name = path[-1].split('.')[0].split('_')[0]
        path[-1] = f'{video_name}.mp4'
        video_v2 = '/'.join(path[-5:])
        assert video_v1 == video_v2
        
        instr2 = extract_raw_instruction(item['conversations'][0]['value'])
        answer = item['conversations'][1]['value']
        plan = extract_plan_part(answer)
        localization, summary = extract_think_part(answer)
        
        if video_name == "0":
            plan_item = deepcopy(data_v1[i])
            plan_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Your assigned task is: {instr2} Describe your navigation plan."
            plan_item['conversations'][1]['value'] = plan
            # res.append(plan_item)
            count_plan += 1
        
        if random.random() <= 0.50:
            localization_item = deepcopy(data_v1[i])
            localization_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Describe your location."
            localization_item['conversations'][1]['value'] = localization
            count_localization += 1
            
            summary_item = deepcopy(data_v1[i])
            summary_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Your assigned task is: {instr2} Summarize your navigation process and describe your next plan."
            summary_item['conversations'][1]['value'] = summary
            count_summary += 1
            
            # res.append(localization_item)
            # res.append(summary_item)
        
        if "I think I should stop" in data_v1[i]['conversations'][1]['value'] and "_" not in aug_video_namev2:
            instruction_reasoning_item = deepcopy(data_v1[i])
            instruction_reasoning_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Based on this video, describe the navigation trajctory."
            instruction_reasoning_item['conversations'][1]['value'] = instr2
            # res.append(instruction_reasoning_item)
            count_inst += 1
    
    print(len(res))
    print(f"count_action: {count_action}\nplan: {count_plan}\nlocalization: {count_localization}\nsummary: {count_summary}\ninstruction reasoning: {count_inst}")
    with open(save_path, "w") as f:
        json.dump(res, f)
    

def fp_img_dataset(data_path, save_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    for item in data:
        item['conversations'][0]['value'] = item['conversations'][0]['value'].replace("<video>\n", "<image>\n<video>\n")
        level = item["level"]
        video = item["video"]
        scan = video.split('/')[-3]
        floorplan_img_path = os.path.join(FLOORPLAN_IMAGE, scan, f"floorplan_level{level}.png")
        item["image"] = floorplan_img_path
        item["video"] = video.replace("concate_floorplan_navigation_video", "videos_action_balance")
    
    with open(save_path, "w") as f:
        json.dump(data, f)


def fp_img_test_action_dataset(data_path, save_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    for item in data:
        item['conversations'][0]['value'] = item['conversations'][0]['value'].replace("<video>\n", "<image>\n<video>\n")
        level = item["level"]
        video = item["video"]
        scan = video.split('/')[-3]
        floorplan_img_path = os.path.join(FLOORPLAN_IMAGE, scan, f"floorplan_level{level}.png")
        item["image"] = floorplan_img_path
        item["video"] = video.replace("concate_floorplan_navigation_video", "videos_action_balance")
    
    with open(save_path, "w") as f:
        json.dump(data, f)
        

def fp_rxr_dataset(data_path, save_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    res = []
    count_action, count_plan, count_localization, count_summary, count_inst = 0, 0, 0, 0, 0
    for item in data:
        video_path = item["video"]
        video_path = video_path.replace("rxr_videos_action_balance", "rxr_from_merge_concate_floorplan_navigation_video")
        item["video"] = video_path
        video_name = item["video"].split('/')[-1].split('.')[0]
        prompt = item['conversations'][0]['value']
        raw_instr = extract_raw_rxr_instruction(prompt)[:-1]
        answer = item['conversations'][1]['value']
        plan = extract_plan_part(answer)
        localization, summary = extract_think_part(answer)
        item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video and your assigned task is: {raw_instr} Decide your next action, which could involve turning left or right by a specific degree or moving forward a certain distance or stop."
        item['conversations'][1]['value'] = extract_action_part(answer)
        if "The next action is stop" in answer:
            item['conversations'][1]['value'] = "I think I should stop because I have finished the instruction."
        res.append(item)
        count_action += 1
        if "60" in answer:
            for _ in range(8):
                res.append(item)
                count_action += 1
        
        if video_name == "0":
            plan_item = deepcopy(item)
            plan_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Your assigned task is: {raw_instr} Describe your navigation plan."
            plan_item['conversations'][1]['value'] = plan
            for _ in range(1):
                res.append(plan_item)
                count_plan += 1
        
        # localization_item = deepcopy(item)
        # localization_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Describe your location."
        # localization_item['conversations'][1]['value'] = localization
        # count_localization += 1
        # res.append(localization_item)
        if random.random() <= 0.50:
            summary_item = deepcopy(item)
            summary_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Your assigned task is: {raw_instr} Summarize your navigation process and describe your next plan."
            summary_item['conversations'][1]['value'] = summary
            count_summary += 1
            res.append(summary_item)
        
        if "The next action is stop" in answer:
            instruction_reasoning_item = deepcopy(item)
            instruction_reasoning_item['conversations'][0]['value'] = f"<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video including observation and floorplan guidance. Based on this video, describe the navigation trajctory."
            instruction_reasoning_item['conversations'][1]['value'] = raw_instr
            
            # repeat stop sample
            for _ in range(1):
                res.append(instruction_reasoning_item)
                count_inst += 1
        
    
    print(len(res))
    print(f"count_action: {count_action}\nplan: {count_plan}\nlocalization: {count_localization}\nsummary: {count_summary}\ninstruction reasoning: {count_inst}")
    with open(save_path, "w") as f:
        json.dump(res, f)
        
        
def fp_understanding_dataset(data_path, save_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    random.shuffle(data)
    
    exist = []
    res = []
    for item in data:
        scene = item['video'].split('/')[-3]
        level = item["level"]
        if scene not in exist or random.random() >= 0.96:
            with open(os.path.join(FLOORPLAN_IMAGE, scene, "floorplan.json"), "r") as f:
                floorplan = json.load(f)
            
            df_q, df_a = describe_floorplan_qa(floorplan, level)
            df_item = deepcopy(item)
            df_item['conversations'][0]['value'] = f"<video>\n{df_q}"
            df_item['conversations'][1]['value'] = df_a
            res.append(df_item)
            
            rnc_q, rnc_a = count_regions(floorplan, level)
            rnc_item = deepcopy(item)
            rnc_item['conversations'][0]['value'] = f"<video>\n{rnc_q}"
            rnc_item['conversations'][1]['value'] = rnc_a
            res.append(rnc_item)
            
            rtc_q, rtc_a = count_region_types(floorplan, level)
            rtc_item = deepcopy(item)
            rtc_item['conversations'][0]['value'] = f"<video>\n{rtc_q}"
            rtc_item['conversations'][1]['value'] = rtc_a
            res.append(rtc_item)

            src_qas = specific_region_count(floorplan, level)
            src_qas = random.sample(src_qas, min(3, len(src_qas)))
            for src_q, src_a in src_qas:
                src_item = deepcopy(item)
                src_item['conversations'][0]['value'] = f"<video>\n{src_q}"
                src_item['conversations'][1]['value'] = src_a
                res.append(src_item)
            
            rti_qas = region_type_identification(floorplan, level)
            rti_qas = random.sample(rti_qas, min(3, len(rti_qas)))
            for rti_q, rti_a in rti_qas:
                rti_item = deepcopy(item)
                rti_item['conversations'][0]['value'] = f"<video>\n{rti_q}"
                rti_item['conversations'][1]['value'] = rti_a
                res.append(rti_item)
            exist.append(scene)
    
    with open(save_path, "w") as f:
        json.dump(res, f)
            
        
def describe_floorplan_qa(floorplan, level):
    df_q = "Describe the floorplan in the form of (region_id, region_type)."
    regions = floorplan["levels"][level]["regions"]
    df_a = ", ".join(f"({id}, {info['label']})" for id, info in regions.items()) + "."
    
    return (df_q, df_a)


def count_regions(floorplan, level: str) -> str:
    rnc_q = "How many regions are there in the floorplan?"
    rnc_a = f"There are {len(floorplan["levels"][level]["regions"])} regions in the floorplan."
    
    return (rnc_q, rnc_a)


def count_region_types(floorplan, level: str) -> str:
    regions = floorplan["levels"][level]["regions"]
    region_types = [region["label"] for region in regions.values()]
    
    rtc_q = "How many different types of regions are there in the floorplan?"
    rtc_a = f"There are {len(set(region_types))} different types of regions in the floorplan."
    
    return (rtc_q, rtc_a)


def count_specific_regions(floorplan, level: str, target_region: str):
    regions = floorplan["levels"][level]["regions"]
    region_types = [region["label"] for region in regions.values()]
    region_counter = Counter(region_types)
    
    return [region_counter.get(region, 0) for region in target_region]

def specific_region_count(floorplan, level):
    qa = []
    regions = floorplan["levels"][level]["regions"]
    region_types = set([region["label"] for region in regions.values()])
    specific_regions = list(region_types)
    true_counts = count_specific_regions(floorplan, level, specific_regions)
    for i, region in enumerate(specific_regions):
        q = f"Identify how many {region}s are there in the floorplan?"
        true_count = true_counts[i]
        a = f"There are {true_count} {region}s in the floorplan."
        qa.append((q, a))
    
    return qa

def identify_region_type(floorplan, level: str, region_id) -> str:
    return [floorplan["levels"][level]["regions"][i]["label"] for i in region_id]

def region_type_identification(floorplan, level):
    qa = []
    regions = floorplan["levels"][level]["regions"]
    region_ids = list(regions.keys())
    true_types = identify_region_type(floorplan, level, region_ids)
    for i, region_id in enumerate(region_ids):
        true_type = true_types[i]
        q = f"What's the type of region {region_id}?"
        a = f"Region {region_id} is a {true_type}."
        qa.append((q, a))
    
    return qa


def split_video_dataset(data_path, save_path):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    for item in data:
        video_path = item['video']
        video_path = video_path.replace("concate_floorplan_navigation_video", "videos_action_balance")
        floorplan_video_path = video_path.replace("concate_floorplan_navigation_video", "floorplan_traj_r2r_from_merge_navigation_video_stepwise")
        item['video'] = [video_path, floorplan_video_path]
        item["conversations"][0]['value'] = item["conversations"][0]['value'].replace(
            "<video>\nImagine you are a robot programmed for navigation tasks. You have been given a navigation video and your assigned task is", 
            "Imagine you are a robot programmed for navigation tasks. You have been given a navigation video <video>\n and a floorplan navigation video <video>.\n Your assigned task is")
    
    with open(save_path, "w") as f:
        json.dump(data, f)


def interleave_dataset(data_path, save_path):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    for item in data:
        item['video'] = item['video'].replace("concate_floorplan_navigation_video", "r2r_from_merge_interleave_floorplan_navigation_video")
    
    with open(save_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    # fp_aux_instr_ablation_dataset()
    # fp_understanding_dataset()
    # split_video_dataset()
    # interleave_dataset()
    fp_aux_instr_dataset()