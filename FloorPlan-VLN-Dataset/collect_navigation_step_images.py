import os
import json
import random
import quaternion
import numpy as np
import habitat_sim
from PIL import Image
from tqdm import tqdm


SCENE_PATH = "datasets/VLN-CE/scene_datasets/"
R2R_FILE_PATH = "datasets/VLN-CE/floorplan_vln_r2r/"
RXR_FILE_PATH = "datasets/VLN-CE/floorplan_vln_rxr/"
VIDEO_PATH = "datasets/VLN-CE/videos_poses/"
VIDEO_STOP_BALANCE_PATH = "datasets/VLN-CE/videos_stop_balance"


action_map = {
    1: "move_forward",
    2: "turn_left",
    3: "turn_right"
}

r2r_sim_settings = {
    "width": 448,  # Spatial resolution of the observations
    "height": 448,
    "hfov": 90,
    "scene": "",  # Scene path
    "agent_height": 1.5,
    "radius": 0.1,
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
    "color_sensor": True,
    "turn_angle": 15,
    "forward_step_size": 0.25
}

rxr_sim_settings = {
    "width": 448,  # Spatial resolution of the observations
    "height": 448,
    "hfov": 79,
    "scene": "",  # Scene path
    "agent_height": 0.88,
    "radius": 0.18,
    "default_agent": 0,
    "sensor_height": 0.88,  # Height of sensors in meters
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
    "color_sensor": True,
    "turn_angle": 30,
    "forward_step_size": 0.25
}


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "hfov": settings["hfov"]
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.parameters['hfov'] = str(sensor_params["hfov"])
            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = settings["agent_height"]
    agent_cfg.radius = settings["radius"]
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=settings["forward_step_size"])
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings["turn_angle"])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings["turn_angle"])
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def pose_augmentation(position, rotation, pos_range=0.25, rot_range=15):
    x, z, y = position
    x_delta = random.uniform(-1 * pos_range, pos_range)
    y_delta = random.uniform(-1 * pos_range, pos_range)
    rot_delta = random.uniform(-1 * rot_range, rot_range)
    ypr = np.degrees(quaternion.as_euler_angles(rotation))
    new_position = np.array([x + x_delta, z, y + y_delta])
    new_ypr = ypr + np.array([0., rot_delta, 0.])
    new_rotation = quaternion.from_euler_angles(np.radians(new_ypr))
    
    return new_position, new_rotation
    

def collect_videos(task="r2r", split="train", stop_balancing=False):
    if task == "r2r":
        file_path = os.path.join(R2R_FILE_PATH, "r2r_from_merge_%s.json"%split)
        gt_file_path = os.path.join(R2R_FILE_PATH, "r2r_from_merge_%s_gt.json"%split)
    elif task == "rxr":
        file_path = os.path.join(RXR_FILE_PATH, "rxr_from_merge_%s.json"%split)
        gt_file_path = os.path.join(RXR_FILE_PATH, "rxr_from_merge_%s_gt.json"%split)
    if not stop_balancing:
        video_save_path = os.path.join(VIDEO_PATH, task, split)
    else:
        video_save_path = os.path.join(VIDEO_STOP_BALANCE_PATH, task, split)
    # os.makedirs(video_save_path, exist_ok=True)
    
    with open(file_path, 'r') as f:
        dataset = json.load(f)["episodes"]
    with open(gt_file_path, 'r') as f:
        gt_dataset = json.load(f)
    
    if task == "r2r":
        sim_settings = r2r_sim_settings
    elif task == "rxr":
        sim_settings = rxr_sim_settings

    # Sort the dataset to process all trajectories within a single scene, 
    # minimizing the need to reinitialize the simulator.
    sorted_dataset = sorted(dataset, key=lambda x: x["scene_id"].split('/')[1])
    current_traj_id = -1
    current_scene = "-1"
    sim = None
    poses = {}
    for data in tqdm(sorted_dataset):
        if data["scene_id"] != current_scene:
            current_scene = data["scene_id"]
            scene = data["scene_id"].split('/')[1]
            scene_path = os.path.join(video_save_path, scene)
            os.makedirs(scene_path, exist_ok=True)
            
            sim_settings["scene"] = os.path.join(SCENE_PATH, data["scene_id"])
            cfg = make_cfg(sim_settings)
            if sim is not None:
                sim.close()
            sim = habitat_sim.Simulator(cfg)
            agent = sim.initialize_agent(sim_settings["default_agent"])
            agent_state = habitat_sim.AgentState()
            
        if data["trajectory_id"] != current_traj_id:
            current_traj_id = data["trajectory_id"]
            poses[current_traj_id] = {}
            traj_path = os.path.join(scene_path, str(current_traj_id))
            os.makedirs(traj_path, exist_ok=True)
            
            start_position = data["start_position"]
            start_rotation = data["start_rotation"]
            agent_state.position = start_position
            agent_state.rotation = np.quaternion(start_rotation[-1], *start_rotation[:-1])
            agent.set_state(agent_state)
            if not stop_balancing:
                obs = sim.get_sensor_observations()
                img = Image.fromarray(obs["color_sensor"])
                img.save(os.path.join(traj_path, "0.png"))
            episode_id = data["episode_id"]
            actions = gt_dataset[str(episode_id)]["actions"]
            actions = [action_map[i] for i in actions[:-1]]
            positions = []
            rotations = []
            agent_state = agent.get_state()
            position = agent_state.position
            rotation = agent_state.rotation
            positions.append(position.tolist())
            rotations.append([rotation.x, rotation.y, rotation.z, rotation.w])
            for i, action in enumerate(actions):
                obs = sim.step(action)
                agent_state = agent.get_state()
                position = agent_state.position
                rotation = agent_state.rotation
                positions.append(position.tolist())
                rotations.append([rotation.x, rotation.y, rotation.z, rotation.w])
                if not stop_balancing:
                    img = Image.fromarray(obs["color_sensor"])
                    img.save(os.path.join(traj_path, "%d.png"%(i+1)))
            poses[current_traj_id]["positions"] = positions
            poses[current_traj_id]["rotations"] = rotations
            if stop_balancing:
                agent_state = agent.get_state()
                position = agent_state.position
                rotation = agent_state.rotation
                balance_num = len(actions) // (len(set(actions)) + 1) # add stop
                for i in range(balance_num):
                    new_position, new_rotation = pose_augmentation(position, rotation)
                    agent_state.position = new_position
                    agent_state.rotation = new_rotation
                    agent.set_state(agent_state)
                    obs = sim.get_sensor_observations()
                    img = Image.fromarray(obs["color_sensor"])
                    img.save(os.path.join(traj_path, "%d.png"%(i+1)))
    
    with open(os.path.join(R2R_FILE_PATH, "pose.json"), "w") as f:
        json.dump(poses, f, indent=1)

if __name__ == "__main__":
    for split in ["train"]:
        collect_videos(task="r2r", split=split, stop_balancing=False)