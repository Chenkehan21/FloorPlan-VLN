import os
import cv2
import json
import gzip
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple
from shapely.geometry import Point, Polygon


MP3D_PATH = "datasets/mp3d/v1/scans"
FLOORPLAN_TRAJ_PATH = "datasets/floorplan_traj"
MP3D_FLOORPLAN_PATH = "datasets/mp3d_floorplan/"

CONNECTIVITY_PATH = "datasets/connectivity"

R2R_DATASET_PATH = "datasets/VLN-CE/R2R_VLNCE_v1-3_preprocessed/"
RXR_DATASET_PATH = "datasets/RxR_VLNCE_v0/"

ONE_FLOOR_SAVE_PATH = "datasets/onefloor"
R2R_SAVE_PATH = "datasets/R2R_onefloor_processed"
RXR_SAVE_PATH = "datasets/RXR_onefloor_processed"
MERGE_SAVE_PATH = "datasets/merged"
R2R_SCAN_SORTED_PATH = "datasets/R2R_scan_sorted"
RXR_SCAN_SORTED_PATH = "datasets/RXR_scan_sorted"
FloorPlan_VLN_R2R_PATH = "datasets/floorplan_vln_r2r"
FloorPlan_VLN_RxR_PATH = "datasets/floorplan_vln_rxr"

ROOM_LABELS = {
    'a': 'bathroom',
    'b': 'bedroom',
    'c': 'closet',
    'd': 'dining room',
    'e': 'entryway',
    'f': 'family room',
    'g': 'garage',
    'h': 'hallway',
    'i': 'library',
    'j': 'laundry room',
    'k': 'kitchen',
    'l': 'living room',
    'm': 'meeting room',
    'n': 'lounge',
    'o': 'office',
    'p': 'porch',
    'r': 'recrec room',
    's': 'stairs',
    't': 'toilet',
    'u': 'tool room',
    'v': 'tv room',
    'w': 'gym',
    'x': 'outdoor',
    'y': 'balcony',
    'z': 'other room',
    'B': 'bar',
    'C': 'classroom',
    'D': 'dining booth',
    'S': 'spa',
    'Z': 'junk',
    '-': 'no label',
}
COLOR_PALETT = [
    "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58230", "#911eb4",
    "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff",
    "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd7b4",
    "#2f4f4f", "#bde0fe", "#7fffd4", "#ff69b4", "#4682b4", "#c71585",
    "#f4a460", "#556b2f", "#40e0d0", "#dda0dd", "#98fb98", "#ffb6c1",
    "#4f5e78"
]
ROOM_COLORS = dict(zip(ROOM_LABELS.values(), COLOR_PALETT))


class NavigationDatasetsCollector:
    def __init__(self, save_onefloor: bool, onefloor_save_path: str,
                 save_r2r: bool, r2r_save_path: str,
                 save_rxr: bool, rxr_save_path: str,
                 save_merge: bool, merge_save_path: str,
                 save_scan_sorted_r2r: bool, scan_sorted_r2r_path: str,
                 save_scan_sorted_rxr: bool, scan_sorted_rxr_path: str,
                 save_split_from_merge: bool, r2r_split_from_merge_path: str, rxr_split_from_merge_path: str,
                 plot_floorplan: bool=False, plot_traj: bool=False, plot_vertex: bool=False, extract_region_graph: bool=False):
        self.save_onefloor = save_onefloor
        self.onefloor_save_path = onefloor_save_path
        self.save_r2r = save_r2r
        self.r2r_save_path = r2r_save_path
        self.save_rxr = save_rxr
        self.rxr_save_path = rxr_save_path
        self.save_merge = save_merge
        self.merge_save_path = merge_save_path
        self.save_scan_sorted_r2r = save_scan_sorted_r2r
        self.scan_sorted_r2r_path = scan_sorted_r2r_path
        self.save_scan_sorted_rxr = save_scan_sorted_rxr
        self.scan_sorted_rxr_path = scan_sorted_rxr_path
        self.save_split_from_merge = save_split_from_merge
        self.r2r_split_from_merge_path = r2r_split_from_merge_path
        self.rxr_split_from_merge_path = rxr_split_from_merge_path
        
        self._plot_floorplan = plot_floorplan
        self._plot_traj = plot_traj
        self._plot_vertex = plot_vertex
        self._extract_region_graph = extract_region_graph
    
    def create_floorplans(self):
        scans = os.listdir(MP3D_PATH)
        for scan in tqdm(scans):
            house_file_path = os.path.join(MP3D_PATH, scan, "house_segmentations", "%s.house"%scan)
            floorplan = {
                "name": scan,
                "total_levels": 0,
                "total_regions": 0,
                "levels": {},
                }
            current_scan_node_region = {}
            scan_path = os.path.join(MP3D_FLOORPLAN_PATH, scan)
            os.makedirs(scan_path, exist_ok=True)
            floorplan_save_path = os.path.join(MP3D_FLOORPLAN_PATH, scan, "floorplan.json")
            
            with open(house_file_path , "r") as f:
                region_level_table = {}
                surface_region_table = {}
                room_label_count = {}
                for line in f:
                    parts = line.split()
                    line_type = parts[0]
                    
                    if line_type == "H":
                        floorplan['total_regions'] = parts[10]
                        floorplan['total_levels'] = parts[12]
                        
                    if line_type == "L":
                        level_id = parts[1]
                        levels = floorplan["levels"]
                        if level_id not in levels.keys():
                            levels[level_id] = {"height_range": (float(parts[9]), float(parts[12])), "regions":{}}
                    
                    if line_type == "R":
                        region_id, level_id, label = parts[1], parts[2], parts[5]
                        if label not in room_label_count.keys():
                            room_label_count[label] = 1
                        else:
                            room_label_count[label] += 1
                        region_level_table[region_id] = level_id
                        levels = floorplan["levels"]
                        regions = levels[level_id]["regions"]
                        if region_id not in regions.keys():
                            regions[region_id] = {"label": ROOM_LABELS[label], "boundaries": []}
                        regions[region_id]["region_height_range"] = (float(parts[11]), float(parts[-6]))
                        regions[region_id]["id"] = room_label_count[label]
                    
                    if line_type == "S":
                        surface_index, region_index = parts[1], parts[2]
                        surface_region_table[surface_index] = region_index
                    
                    if line_type == "V":
                        surface_id = parts[2]
                        region_id = surface_region_table[surface_id]
                        level_id = region_level_table[region_id]
                        coordinates = parts[4 : 7]
                        floorplan["levels"][level_id]["regions"][region_id]["boundaries"].append(coordinates)
                    
                    if line_type == "P":
                        current_scan_node_region[parts[1]] = parts[3]
            
            for regions in levels.values():
                items = regions["regions"]
                regions["regions"] = {key: items[key] for key in sorted(items.keys(), key=int)}
            floorplan["levels"] = {key:levels[key] for key in sorted(levels.keys(), key=int)}
            
            if self._extract_region_graph:
                levels = floorplan["levels"]
                for level_id, level in levels.items():
                    mp3d_topo_path = os.path.join(scan_path, f"floorplan_mp3d_topo{level_id}.png")
                    floorplan_adj_path = os.path.join(scan_path, f"floorplan_region_graph{level_id}.png")
                    regions = level["regions"]
                    region_polys, region_labels = [], []
                    region_ids = list(regions.keys())
                    for id, region in regions.items():
                        region_polys.append(np.array(region["boundaries"]).astype(np.float64))
                        label = region["label"]
                        if label not in ROOM_LABELS.values():
                            label = "no label"
                        region_labels.append(label)
                    
                    graph, node_positions = self.load_connectivity_graph(scan, current_scan_node_region, region_ids)
                    adjacent_matrix, region_connectivity = self.build_region_adjacency(graph, current_scan_node_region, region_ids)
                    if len(node_positions) > 0:
                        nodes = np.vstack(list(node_positions.values()))
                        region_polys, nodes = self.transforms_topo(region_polys, nodes)
                        center_positions = self.calculate_node_abs_position(region_polys, scale=1000)
                        if self._plot_floorplan:
                            node_positions = {k: v for (k, v) in zip(list(node_positions.keys()), nodes)}
                            img = self.plot_floorplan_topo(region_polys, region_labels, region_ids, graph, node_positions, scale=500)
                            cv2.imwrite(mp3d_topo_path, img)
                            
                            img_adj = self.plot_floorplan_adj(region_polys, region_labels, region_ids, adjacent_matrix, scale=500)
                            cv2.imwrite(floorplan_adj_path, img_adj)
                            
                        for id, (region_id, region) in enumerate(regions.items()):
                            region["center"] = center_positions[id]
                            if int(region_id) in region_connectivity:
                                region["connectivity"] = region_connectivity[int(region_id)]
                            else:
                                region["connectivity"] = []
                    else:
                        print(floorplan["name"], level_id)
                        for id, (region_id, region) in enumerate(regions.items()):
                            region["center"] = [500, 500]
                            region["connectivity"] = []
                        if self._plot_floorplan:
                            region_polys = self.transforms(region_polys)
                            img = self.plot_floorplan_som(region_polys, region_labels, region_ids, scale=1000)
                            cv2.imwrite(floorplan_adj_path, img)
                    level["regions"] = regions
                    level["region_graph"] = adjacent_matrix.tolist()
                
                with open(floorplan_save_path, 'w') as f:
                    json.dump(floorplan, f, indent=2)
            elif self._plot_floorplan:
                levels = floorplan["levels"]
                for level_id, level in levels.items():
                    f_path = os.path.join(scan_path, f"floorplan_level{level_id}.png")
                    regions = level["regions"]
                    region_polys, region_labels = [], []
                    for id, region in regions.items():
                        region_polys.append(np.array(region["boundaries"]).astype(np.float64))
                        label = region["label"]
                        if label not in ROOM_LABELS.values():
                            label = "no label"
                        region_labels.append(label)
                    region_ids = list(regions.keys())
                    region_polys = self.transforms(region_polys)
                    img = self.plot_floorplan_som(region_polys, region_labels, region_ids, scale=1000)
                    cv2.imwrite(f_path, img)
    
    def calculate_node_abs_position(self, regions, scale=256):
        region_centers = []
        regions = [(region * scale / 256).round().astype(int) for region in regions]
        for i, region in enumerate(regions):
            mask = np.zeros((scale, scale), dtype=np.uint8)
            cv2.fillPoly(mask, [region], 255)

            dist_map = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_map)

            centroid = np.mean(region, axis=0).astype(int)
            cx, cy = centroid

            candidate_yx = np.argwhere(dist_map >= max_val * 0.95)  # shape: (K, 2)
            
            dists = np.linalg.norm(candidate_yx - [cy, cx], axis=1)
            best_idx = np.argmin(dists)
            best_point = candidate_yx[best_idx]
            center = (float(best_point[1]), float(best_point[0]))
            center_flipped = (scale - center[0], center[1])
            region_centers.append(center_flipped)

        return region_centers
    
    def build_region_adjacency(self, graph: nx.Graph, node_to_region: dict, region_ids: int):
        num_regions = len(region_ids)
        # adj_matrix = np.zeros((num_regions, num_regions), dtype=int) + np.diag(np.ones(num_regions, dtype=int))
        adj_matrix = np.zeros((num_regions, num_regions), dtype=int)
        int_region_ids = [int(i) for i in region_ids]
        sorted_region_ids = sorted(int_region_ids)
        region_connectivity = defaultdict(list)
        for u, v in graph.edges():
            if node_to_region[u] in region_ids and node_to_region[v] in region_ids:
                region_u = int(node_to_region[u])
                region_v = int(node_to_region[v])
                
                if region_u != region_v:
                    region_connectivity[region_u].append(region_v)
                    region_connectivity[region_v].append(region_u)
                    id_u = sorted_region_ids.index(region_u)
                    id_v = sorted_region_ids.index(region_v)
                    adj_matrix[id_u][id_v] = 1
                    adj_matrix[id_v][id_u] = 1
        region_connectivity = {k: list(set(v)) for k, v in region_connectivity.items()}
        
        return adj_matrix, region_connectivity
    
    def load_connectivity_graph(self, scan, node_region, region_ids):
        with open(os.path.join(CONNECTIVITY_PATH, '%s_connectivity.json' % scan)) as f:
            data = json.load(f)
        
        graph = nx.Graph()
        positions = {}
        for i,item in enumerate(data):
            if item['included']:
                for j,conn in enumerate(item['unobstructed']):
                    if conn and data[j]['included']:
                        if node_region[item['image_id']] in region_ids:
                            positions[item['image_id']] = np.array([item['pose'][3], item['pose'][7]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            graph.add_edge(item['image_id'],data[j]['image_id'],weight=self.distance(item,data[j]))
        nx.set_node_attributes(graph, values=positions, name='position')
        
        return graph, positions
    
    def distance(self, pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5
    
    def process_r2r_dataset(self, dataset: json, gt_data: json, split: str):
        episodes = dataset["episodes"]
        res = {}
        for episode in episodes:
            episode_id = episode["episode_id"]
            trajectory_id = episode["trajectory_id"]
            scene_id = episode["scene_id"]
            scene = scene_id.split('/')[1]
            
            if scene not in res.keys():
                res[scene] = {}
            if trajectory_id not in res[scene].keys():
                res[scene][trajectory_id] = {
                    "episode_ids": [],
                    "instruction": [],
                    "gt_path": gt_data[str(episode_id)]["locations"],
                    "reference_path": episode["reference_path"]
                }
            if episode_id not in res[scene][trajectory_id]["episode_ids"]:
                res[scene][trajectory_id]["episode_ids"].append(episode_id)
                res[scene][trajectory_id]["instruction"].append(episode["instruction"]["instruction_text"])
        
        # sort
        for scene, trajs in res.items():
            trajs = {key: trajs[key] for key in sorted(trajs.keys(), key=int)}
            
        if self.save_scan_sorted_r2r:
            with open(os.path.join(self.scan_sorted_r2r_path, "%s_%s.json"%(split, "scan_sorted_r2r")), "w") as f:
                json.dump(res, f, indent=2)
                
        return res
    
    def process_rxr_dataset(self, dataset: json, gt_data: json, split: str):
        episodes = dataset["episodes"]
        res = {}
        for episode in episodes:
            language = episode["instruction"]["language"]
            if "en" in language:
                episode_id = episode["episode_id"]
                trajectory_id = episode["trajectory_id"]
                scene_id = episode["scene_id"]
                scene = scene_id.split('/')[1]
                
                if scene not in res.keys():
                    res[scene] = {}
                if trajectory_id not in res[scene].keys():
                    res[scene][trajectory_id] = {
                        "episode_ids": [],
                        "instruction": [],
                        "gt_path": gt_data[str(episode_id)]["locations"],
                        "reference_path": episode["reference_path"]
                    }
                if episode_id not in res[scene][trajectory_id]["episode_ids"]:
                    res[scene][trajectory_id]["episode_ids"].append(episode_id)
                    res[scene][trajectory_id]["instruction"].append(episode["instruction"]["instruction_text"])
                
        # sort
        for scene, trajs in res.items():
            trajs = {key: trajs[key] for key in sorted(trajs.keys(), key=int)}
        
        if self.save_scan_sorted_rxr:
            with open(os.path.join(self.scan_sorted_rxr_path, "%s_%s.json"%(split, "scan_sorted_rxr")), "w") as f:
                json.dump(res, f, indent=2)
        
        return res
    
    @staticmethod
    def check_point_roomtype(point: np.ndarray, regions_height_range: Dict, 
                             regions_boundaries: Dict, regions_level: Dict,
                             check_radius: float=0.01, check_step: float=0.25) -> Tuple[List, List]:
        position_level, position_region = -1, -1
        position_height = point[1]
        point = np.array([point[0], -1 * point[2]]) # habitat use left-hand coordination
        check_point = Point(point)
        check_circle = check_point.buffer(check_radius)
        
        while position_region == -1 and position_level == -1:
            for region_id, height_range in regions_height_range.items():
                polygon = Polygon(regions_boundaries[region_id]) # polygon.intersects(check_circle)
                flag1 = (position_height >= height_range[0] and 
                                position_height <= height_range[1] and 
                                polygon.intersects(check_circle))
                flag2 = (position_height >= height_range[0] - 0.25 and 
                            position_height <= height_range[1] + 0.25 and 
                            polygon.intersects(check_circle))
                if (flag1 or flag2):
                    position_region = region_id
                    position_level = regions_level[region_id]
                    break
            check_radius += check_step
            check_circle = check_point.buffer(check_radius)
            
        return position_region, position_level, check_radius - check_step

    @staticmethod
    def extract_regions_info(floorplan: Dict) -> Tuple[Dict, Dict, Dict]:
        levels = floorplan["levels"]
        regions_height_range = {}
        regions_level = {}
        regions_boundaries = {}
        
        for level_id, level in levels.items():
            regions = level["regions"]
            for region_id, region_info in regions.items():
                regions_level[region_id] = level_id
                boundaries = np.array(region_info["boundaries"]).astype(float)[:, :2]
                height_range = region_info["region_height_range"] 
                regions_height_range[region_id] = np.array(height_range).astype(float)
                regions_boundaries[region_id] = boundaries

        return regions_height_range, regions_level, regions_boundaries
    
    def segment_traj_level(self, traj_levels: List) -> List:
        start = traj_levels[0]
        group, groups = [], []
        for i, level in enumerate(traj_levels):
            if level == start:
                group.append(level)
            else:
                groups.append(group)
                start = level
                group = [level]
            if i == len(traj_levels) - 1:
                groups.append(group)
        
        return groups
    
    def transforms(self, regions: List[np.ndarray], traj: np.ndarray=None, width: int=256, height: int=256) -> List:
        region_lengths = [len(region) for region in regions]
        regions = np.concatenate(regions, axis=0)
        regions = regions[:, :2]
        
        if traj is not None:
            traj = traj[:, [0,2]]
            traj[:, 1] *= -1
            regions = np.concatenate([regions, traj], axis=0)
            region_lengths.append(len(traj))
        
        image_res = np.array([width, height])
        
        region_min = np.min(regions, axis=0)
        region_max = np.max(regions, axis=0)
        max_min = region_max - region_min
        region_min -= 0.01 * max_min
        region_max += 0.01 * max_min
        
        min_id = np.argmin(max_min)
        ratio = np.min(max_min) / np.max(max_min)
        image_res[min_id] *= ratio
        regions = (regions - region_min) / (region_max - region_min)
        regions = regions * image_res
        
        floorplan_center = np.array([width, height], dtype=np.float32) / 2.
        region_center = (np.max(regions, axis=0) - np.min(regions, axis=0)) / 2 + np.min(regions, axis=0)
        delta = floorplan_center - region_center
        regions += delta
        regions = np.round(regions).astype(int)
        
        split_indices = np.cumsum(region_lengths[:-1])
        regions = np.split(regions, split_indices)
        if traj is not None:
            trajs = regions[-1]
            regions = regions[:-1]
            
            return regions, trajs
        else:
            return regions

    def transforms_topo(self, regions: List[np.ndarray], nodes:np.ndarray, width: int=256, height: int=256) -> List:
        
        region_lengths = [len(region) for region in regions]
        regions = np.concatenate(regions, axis=0)
        regions = regions[:, :2]
        
        regions = np.concatenate([regions, nodes], axis=0)
        region_lengths.append(len(nodes))
        
        image_res = np.array([width, height])
        
        region_min = np.min(regions, axis=0)
        region_max = np.max(regions, axis=0)
        max_min = region_max - region_min
        
        region_min -= 0.01 * max_min
        region_max += 0.01 * max_min
        
        min_id = np.argmin(max_min)
        ratio = np.min(max_min) / np.max(max_min)
        image_res[min_id] *= ratio
        regions = (regions - region_min) / (region_max - region_min)
        regions = regions * image_res
        
        floorplan_center = np.array([width, height], dtype=np.float32) / 2.
        region_center = (np.max(regions, axis=0) - np.min(regions, axis=0)) / 2 + np.min(regions, axis=0)
        delta = floorplan_center - region_center
        regions += delta
        regions = np.round(regions).astype(int)

        split_indices = np.cumsum(region_lengths[:-1])
        regions = np.split(regions, split_indices)
        nodes = regions[-1]
        regions = regions[:-1]
        
        return regions, nodes
    
    def plot_floorplan(self, regions: List[np.ndarray], room_labels: List, 
                       traj: np.ndarray, region_ids: List[int], scale: int=256):
        """Draw floorplan map where different colors indicate different rooms
        """
        regions = [(region * scale / 256).round().astype(int) for region in regions]
        traj = (traj * scale / 256).round().astype(int)
        # define the color map
        room_colors = [ROOM_COLORS[label] for label in room_labels]
        colorMap = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in room_colors]
        colorMap = np.asarray(colorMap)
        if len(regions) > 0:
            colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap], axis=0).astype(
                np.uint8)
        else:
            colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0)], axis=0).astype(
                np.uint8)
            
        # from RGB to BGR
        colorMap = colorMap[:, ::-1]
        alpha_channels = np.zeros(colorMap.shape[0], dtype=np.uint8)
        alpha_channels[1:len(regions) + 1] = 150
        colorMap = np.concatenate([colorMap, np.expand_dims(alpha_channels, axis=-1)], axis=-1)
        
        room_map = np.zeros([scale, scale]).astype(np.int32)
        for idx, polygon in enumerate(regions):
            cv2.fillPoly(room_map, [polygon], color=idx + 1)
        image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 4))

        pointColor = (0,0,0,255)
        lineColor = (0,0,0,255)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        for region in regions:
            for i, point in enumerate(region):
                if i == len(region)-1:
                    cv2.line(image, tuple(point), tuple(region[0]), color=lineColor, thickness=5, lineType=cv2.LINE_AA)
                else:    
                    cv2.line(image, tuple(point), tuple(region[i+1]), color=lineColor, thickness=5, lineType=cv2.LINE_AA)
        if self._plot_traj:
            for i in range(len(traj)):
                if i + 1 <= len(traj) - 1:
                    cv2.line(image, tuple(traj[i]), tuple(traj[i + 1]), color=(255, 255, 255, 255), thickness=5, lineType=cv2.LINE_AA)
                if i == 0:
                    cv2.circle(image, tuple(traj[i]), color=(255, 0, 0, 255), radius=12, thickness=-1)
                if i + 1 == len(traj) - 1:
                    cv2.circle(image, tuple(traj[i + 1]), color=(0, 0, 255, 255), radius=12, thickness=-1)
        if self._plot_vertex:
            for region in regions:
                for i, point in enumerate(region):
                    cv2.circle(image, tuple(point), color=pointColor, radius=12, thickness=-1)
                    cv2.circle(image, tuple(point), color=(255, 255, 255, 0), radius=6, thickness=-1)
        image = np.ascontiguousarray(np.fliplr(image))
        
        for i, region in enumerate(regions):
            center = np.mean(region, axis=0).astype(int)
            text = region_ids[i]

            text_color = (255, 255, 255, 255)  # 白色
            cv2.putText(image, text, (scale - center[0], center[1]), font, font_scale, text_color, font_thickness)
        
        return image

    def plot_floorplan_traj(self, traj_groups: List[List], floorplan: Dict, 
                            gt_traj: List, traj_id: str, scene_path: str):
        start_id = 0
        for i, group in enumerate(traj_groups):
            grouped_gt_path = np.array(gt_traj[start_id : start_id + len(group)]).astype(float)
            start_id += len(group)
            level_id = group[0]
            regions = floorplan["levels"][level_id]["regions"]
            region_polys = []
            region_labels = []
            for region in regions.values():
                boundaries = np.array(region["boundaries"]).astype(np.float64)
                region_polys.append(boundaries)
                region_labels.append(region["label"])
            region_polys, traj = self.transforms(region_polys, grouped_gt_path)
            img = self.plot_floorplan(region_polys, region_labels, traj, list(regions.keys()), scale=1000)
            img_path = os.path.join(scene_path, "traj%s_"%traj_id + "part%s_"%i + "level%s.png"%level_id)
            cv2.imwrite(img_path, img)
    
    def plot_floorplan_som(self, regions: List[np.ndarray], room_labels: List, 
                region_ids: List[int], scale: int=256):
        """Draw floorplan map where different colors indicate different rooms"""
        regions = [(region * scale / 256).round().astype(int) for region in regions]

        # define the color map
        room_colors = [ROOM_COLORS[label] for label in room_labels]
        colorMap = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in room_colors]
        colorMap = np.asarray(colorMap)
        if len(regions) > 0:
            colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap], axis=0).astype(
                np.uint8)
        else:
            colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0)], axis=0).astype(
                np.uint8)

        colorMap = colorMap[:, ::-1]  # RGB to BGR
        colorMap[0] = np.array([255, 255, 255], dtype=np.uint8)  # set background to white

        room_map = np.zeros([scale, scale], dtype=np.int32)
        for idx, polygon in enumerate(regions):
            cv2.fillPoly(room_map, [polygon], color=idx + 1)
        image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 3))

        lineColor = (0, 0, 0)  # black
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        for region in regions:
            for i, point in enumerate(region):
                pt1 = tuple(point)
                pt2 = tuple(region[0]) if i == len(region) - 1 else tuple(region[i+1])
                cv2.line(image, pt1, pt2, color=lineColor, thickness=2, lineType=cv2.LINE_AA)

        image = np.ascontiguousarray(np.fliplr(image))
        
        for i, region in enumerate(regions):
            region_id = str(region_ids[i])

            mask = np.zeros((scale, scale), dtype=np.uint8)
            cv2.fillPoly(mask, [region], 255)

            dist_map = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_map)

            centroid = np.mean(region, axis=0).astype(int)
            cx, cy = centroid

            candidate_yx = np.argwhere(dist_map >= max_val * 0.95)  # shape: (K, 2)
            
            dists = np.linalg.norm(candidate_yx - [cy, cx], axis=1)
            best_idx = np.argmin(dists)
            best_point = candidate_yx[best_idx]
            center = (best_point[1], best_point[0])  # x, y 格式
            radius = dist_map[best_point[0], best_point[1]]

            font = cv2.FONT_HERSHEY_DUPLEX
            font_thickness = 1
            font_scale = 1.
            text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
            text_w, text_h = text_size

            while (text_w**2 + text_h**2)**0.5 / 2 > radius * 0.8:
                font_scale -= 0.1
                if font_scale < 0.3:
                    break
                text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
                text_w, text_h = text_size

            text_org = (scale - center[0] - text_w // 2, center[1] + text_h // 2)
            top_left = (text_org[0], text_org[1] - text_h)
            bottom_right = (text_org[0] + text_w, text_org[1] + int(text_h * 0.3))

            cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

            cv2.putText(image, region_id, text_org, font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        
        return image
    
    def plot_floorplan_topo(self, regions: List[np.ndarray], room_labels: List, region_ids: List, graph, node_positions, scale: int=256):
        """Draw floorplan map where different colors indicate different rooms"""
        regions = [(region * scale / 256).round().astype(int) for region in regions]

        # define the color map
        room_colors = [ROOM_COLORS[label] for label in room_labels]
        colorMap = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in room_colors]
        colorMap = np.asarray(colorMap)
        if len(regions) > 0:
            colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap], axis=0).astype(
                np.uint8)
        else:
            colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0)], axis=0).astype(
                np.uint8)

        colorMap = colorMap[:, ::-1]  # RGB to BGR
        colorMap[0] = np.array([255, 255, 255], dtype=np.uint8)  # set background to white

        room_map = np.zeros([scale, scale], dtype=np.int32)
        for idx, polygon in enumerate(regions):
            cv2.fillPoly(room_map, [polygon], color=idx + 1)
        image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 3))

        lineColor = (0, 0, 0)  # black
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        for region in regions:
            for i, point in enumerate(region):
                pt1 = tuple(point)
                pt2 = tuple(region[0]) if i == len(region) - 1 else tuple(region[i+1])
                cv2.line(image, pt1, pt2, color=lineColor, thickness=2, lineType=cv2.LINE_AA)

        node_color = (0, 255, 0, 255)
        node_radius = 3
        edge_color = (255, 0, 0, 255)
        edge_thickness = 2

        for u, v in graph.edges():
            if u in node_positions and v in node_positions:
                pt1 = tuple((node_positions[u] * scale / 256).round().astype(int))
                pt2 = tuple((node_positions[v] * scale / 256).round().astype(int))
                cv2.line(image, pt1, pt2, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)

        for node, pos in node_positions.items():
            pt = tuple((pos * scale / 256).round().astype(int))
            cv2.circle(image, pt, node_radius, node_color, thickness=-1)

        image = np.ascontiguousarray(np.fliplr(image))
        
        for i, region in enumerate(regions):
            region_id = str(region_ids[i])

            mask = np.zeros((scale, scale), dtype=np.uint8)
            cv2.fillPoly(mask, [region], 255)

            dist_map = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_map)

            centroid = np.mean(region, axis=0).astype(int)
            cx, cy = centroid

            candidate_yx = np.argwhere(dist_map >= max_val * 0.95)  # shape: (K, 2)
            
            dists = np.linalg.norm(candidate_yx - [cy, cx], axis=1)
            best_idx = np.argmin(dists)
            best_point = candidate_yx[best_idx]
            center = (best_point[1], best_point[0])  # x, y
            radius = dist_map[best_point[0], best_point[1]]

            font = cv2.FONT_HERSHEY_DUPLEX
            font_thickness = 1
            font_scale = 1.
            text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
            text_w, text_h = text_size

            while (text_w**2 + text_h**2)**0.5 / 2 > radius * 0.8:
                font_scale -= 0.1
                if font_scale < 0.3:
                    break
                text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
                text_w, text_h = text_size

            text_org = (scale - center[0] - text_w // 2, center[1] + text_h // 2)
            top_left = (text_org[0], text_org[1] - text_h)
            bottom_right = (text_org[0] + text_w, text_org[1] + int(text_h * 0.3))

            cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 0), thickness=-1)

            cv2.putText(image, region_id, text_org, font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        
        return image
    
    def plot_floorplan_adj(self, regions: List[np.ndarray], room_labels: List, region_ids: List[int], adj_matrix, scale: int=256):
        """Draw floorplan map where different colors indicate different rooms"""
        regions = [(region * scale / 256).round().astype(int) for region in regions]

        # define the color map
        room_colors = [ROOM_COLORS[label] for label in room_labels]
        colorMap = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in room_colors]
        colorMap = np.asarray(colorMap)
        if len(regions) > 0:
            colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap], axis=0).astype(
                np.uint8)
        else:
            colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0)], axis=0).astype(
                np.uint8)

        colorMap = colorMap[:, ::-1]  # RGB to BGR
        colorMap[0] = np.array([255, 255, 255], dtype=np.uint8)  # set background to white

        room_map = np.zeros([scale, scale], dtype=np.int32)
        for idx, polygon in enumerate(regions):
            cv2.fillPoly(room_map, [polygon], color=idx + 1)
        image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 3))

        lineColor = (0, 0, 0)  # black
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        for region in regions:
            for i, point in enumerate(region):
                pt1 = tuple(point)
                pt2 = tuple(region[0]) if i == len(region) - 1 else tuple(region[i+1])
                cv2.line(image, pt1, pt2, color=lineColor, thickness=2, lineType=cv2.LINE_AA)

        image = np.ascontiguousarray(np.fliplr(image))
        region_centers = []
        for i, region in enumerate(regions):
            region_id = str(region_ids[i])

            mask = np.zeros((scale, scale), dtype=np.uint8)
            cv2.fillPoly(mask, [region], 255)

            dist_map = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_map)

            centroid = np.mean(region, axis=0).astype(int)
            cx, cy = centroid

            candidate_yx = np.argwhere(dist_map >= max_val * 0.95)  # shape: (K, 2)
            
            dists = np.linalg.norm(candidate_yx - [cy, cx], axis=1)
            best_idx = np.argmin(dists)
            best_point = candidate_yx[best_idx]
            center = (best_point[1], best_point[0])  # x, y
            center_flipped = (scale - center[0], center[1])
            region_centers.append(center_flipped)
        
        line_color = (255, 0, 0)
        exist_lines = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if adj_matrix[i, j]:
                    pt1 = region_centers[i]
                    pt2 = region_centers[j]
                    if [pt1, pt2] not in exist_lines:
                        cv2.line(image, pt1, pt2, color=line_color, thickness=2, lineType=cv2.LINE_AA)
                        cv2.circle(image, pt1, 3, (0, 255, 0), thickness=-1)
                        cv2.circle(image, pt2, 3, (0, 255, 0), thickness=-1)
                        exist_lines.append([pt1, pt2])
                        exist_lines.append([pt2, pt1])
            
        for i, region in enumerate(regions):
            region_id = str(region_ids[i])

            mask = np.zeros((scale, scale), dtype=np.uint8)
            cv2.fillPoly(mask, [region], 255)

            dist_map = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_map)

            centroid = np.mean(region, axis=0).astype(int)
            cx, cy = centroid

            candidate_yx = np.argwhere(dist_map >= max_val * 0.95)  # shape: (K, 2)
            
            dists = np.linalg.norm(candidate_yx - [cy, cx], axis=1)
            best_idx = np.argmin(dists)
            best_point = candidate_yx[best_idx]
            center = (best_point[1], best_point[0])  # x, y
            center_flipped = (scale - center[0], center[1])
            radius = dist_map[best_point[0], best_point[1]]

            font = cv2.FONT_HERSHEY_DUPLEX
            font_thickness = 1
            font_scale = 1.
            text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
            text_w, text_h = text_size

            while (text_w**2 + text_h**2)**0.5 / 2 > radius * 0.8:
                font_scale -= 0.1
                if font_scale < 0.3:
                    break
                text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
                text_w, text_h = text_size

            text_org = (scale - center[0] - text_w // 2, center[1] + text_h // 2)
            top_left = (text_org[0], text_org[1] - text_h)
            bottom_right = (text_org[0] + text_w, text_org[1] + int(text_h * 0.3))

            cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 0), thickness=-1)

            cv2.putText(image, region_id, text_org, font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        
        return image
    
    def exact_onefloor_trajs(self, data: json, task: str="r2r", split: str="train", plot: bool=False):
        print("process onefloor of %s-%s"%(task, split))
        split_path = os.path.join(FLOORPLAN_TRAJ_PATH, "floorplan_traj_%s"%task, split)
        os.makedirs(split_path, exist_ok=True)
        
        not_found = []
        total_trajs = 0
        multifloor_trajs = []
        res = {}
        double_check_trajs = []
        regions_in_traj = []
        
        # data is r2r or rxr json file sorted by scans
        for scene, trajs in tqdm(data.items()):
            scene_path = os.path.join(split_path, scene)
            os.makedirs(scene_path, exist_ok=True)
            floorplan_path = os.path.join(MP3D_FLOORPLAN_PATH, scene, "floorplan.json")
            with open(floorplan_path, 'r') as f:
                floorplan = json.load(f)
            
            regions_height_range, regions_level, regions_boundaries = self.extract_regions_info(floorplan)
                    
            for traj_id, traj_info in trajs.items():
                total_trajs += 1
                gt_path = traj_info["gt_path"] # use gt_path could be more precise than reference_path
                traj_region_ids, traj_levels, check_radius_list, traj_region_type_id = [], [], [], []
                for point in gt_path:
                    point_region, point_level, check_radius = self.check_point_roomtype(point, 
                                                                    regions_height_range, 
                                                                    regions_boundaries, 
                                                                    regions_level)
                    point_region_type = floorplan["levels"][point_level]["regions"][point_region]["label"]
                    traj_region_type_id.append((point_region, point_region_type))
                    traj_region_ids.append(point_region)
                    traj_levels.append(point_level)
                    check_radius_list.append(check_radius)
                traj_groups = self.segment_traj_level(traj_levels)
                if len(traj_groups) >= 2:
                    multifloor_trajs.append(traj_id)
                elif check_radius_list[-1] >= 0.1: # Cases that the end point is located at the boundary of a region.
                    double_check_trajs.append(traj_id)
                else:
                    end_region = traj_region_ids[-1]
                    traj_level = traj_groups[0][0]
                    res[traj_id] = data[scene][traj_id]
                    res[traj_id]["scene"] = scene
                    res[traj_id]["level"] = traj_level
                    res[traj_id]["end_region"] = floorplan["levels"][traj_level]["regions"][end_region]["label"]
                    res[traj_id]["end_region_id"] = end_region
                    res[traj_id]["traj_regions"] = traj_region_type_id
                    
                    regions_in_traj.append(len(set(traj_region_ids)))
                    
                if self._plot_traj:
                    self.plot_floorplan_traj(traj_groups, floorplan, gt_path, traj_id, scene_path)
                    
                if point_region == -1 and point_level == -1:
                    not_found.append(traj_id)
        
        if self.save_onefloor:
            os.makedirs(os.path.join(self.onefloor_save_path, task), exist_ok=True)
            with open(os.path.join(self.onefloor_save_path, task, "%s_%s.json"%(split, "one_floor")), "w") as f:
                json.dump(res, f, indent=2)

        return res        
        
    def generate_floorplan_r2r_dataset(self, split):
        with gzip.open(os.path.join(R2R_DATASET_PATH, split, "%s.json.gz"%split), "rt", encoding='utf-8') as f:
            raw_dataset = json.load(f)
        with gzip.open(os.path.join(R2R_DATASET_PATH, split, "%s_gt.json.gz"%split), "rt", encoding='utf-8') as f:
            gt_dataset = json.load(f)
        
        data = self.process_r2r_dataset(raw_dataset, gt_dataset, split)
        onefloor_dataset = self.exact_onefloor_trajs(data, task="r2r", split=split, plot=False)
        res_episodes = []
        for episode in raw_dataset["episodes"]:
            traj_id = episode["trajectory_id"]
            if traj_id in onefloor_dataset.keys():
                episode["instruction"] = episode["instruction"]["instruction_text"]
                episode["level"] = onefloor_dataset[traj_id]["level"]
                episode["end_region"] = onefloor_dataset[traj_id]["end_region"]
                episode["end_region_id"] = onefloor_dataset[traj_id]["end_region_id"]
                episode["traj_regions"] = onefloor_dataset[traj_id]["traj_regions"]
                res_episodes.append(episode)
        
        res = {"episodes": res_episodes}
        if self.save_r2r:
            save_path = os.path.join(self.r2r_save_path, "r2r_processed_onefloor_%s.json"%split)
            with open(save_path, 'w') as f:
                json.dump(res, f, indent=2)
            
        return res

    def generate_floorplan_rxr_dataset(self, split):
        with gzip.open(os.path.join(RXR_DATASET_PATH, split, "%s_guide.json.gz"%split), "rt", encoding='utf-8') as f:
            raw_dataset = json.load(f)
        with gzip.open(os.path.join(RXR_DATASET_PATH, split, "%s_guide_gt.json.gz"%split), "rt", encoding='utf-8') as f:
            gt_dataset = json.load(f)
        
        data = self.process_rxr_dataset(raw_dataset, gt_dataset, split)
        onefloor_dataset = self.exact_onefloor_trajs(data, task="rxr", split=split, plot=False)
        
        episode_ids = []
        for traj_id, info in onefloor_dataset.items():
            episode_ids += info["episode_ids"]
            
        res_episodes = []
        for episode in raw_dataset["episodes"]:
            traj_id = episode["trajectory_id"]
            episode_id = episode["episode_id"]
            if traj_id in onefloor_dataset.keys() and episode_id in episode_ids:
                assert "en" in episode["instruction"]["language"]
                del episode["info"]
                episode["instruction"] = episode["instruction"]["instruction_text"]
                episode["level"] = onefloor_dataset[traj_id]["level"]
                episode["end_region"] = onefloor_dataset[traj_id]["end_region"]
                episode["end_region_id"] = onefloor_dataset[traj_id]["end_region_id"]
                episode["traj_regions"] = onefloor_dataset[traj_id]["traj_regions"]
                res_episodes.append(episode)
        
        res = {"episodes": res_episodes}
        if self.save_rxr:
            save_path = os.path.join(self.rxr_save_path, "rxr_processed_onefloor_%s.json"%split)
            with open(save_path, 'w') as f:
                json.dump(res, f, indent=2)
        
        return res
            
    def prepare_data(self, eps_path, gt_path):
        with open(eps_path, "r") as f:
            episodes = json.load(f)["episodes"]
        with gzip.open(gt_path, "rt", encoding="utf-8") as f:
            gt = json.load(f)
        
        return episodes, gt

    def create_traj_eps_ids(self, eps):
        trajs = []
        for episode in eps:
            trajs.append(int(episode["trajectory_id"]))
        trajs.sort()
        idx = 0
        traj_eps_ids = []
        id2id = {}
        start = trajs[0]
        for traj_id in trajs:
            if traj_id == start:
                traj_eps_ids.append(idx)
            else:
                idx += 1
                traj_eps_ids.append(idx)
                start = traj_id
            if traj_id not in id2id.keys():
                id2id[traj_id] = idx
        assert len(traj_eps_ids) == len(eps)
        
        return traj_eps_ids, id2id

    def merge_r2r_rxr(self, split):
        r2r_episodes = self.generate_floorplan_r2r_dataset(split)["episodes"]
        rxr_episodes = self.generate_floorplan_rxr_dataset(split)["episodes"]
        with gzip.open(os.path.join(R2R_DATASET_PATH, split, "%s_gt.json.gz"%split), "rt", encoding='utf-8') as f:
            r2r_gt = json.load(f)
        with gzip.open(os.path.join(RXR_DATASET_PATH, split, "%s_guide_gt.json.gz"%split), "rt", encoding='utf-8') as f:
            rxr_gt = json.load(f)
        
        r2r_traj_eps_ids, id2id = self.create_traj_eps_ids(r2r_episodes)
        rxr_traj_eps_ids, _ = self.create_traj_eps_ids(rxr_episodes)
        if self._plot_traj:
            split_path = os.path.join(FLOORPLAN_TRAJ_PATH, "floorplan_traj_r2r", split)
            scans = os.listdir(split_path)
            for scan in scans:
                fns = os.listdir(os.path.join(split_path, scan))
                for fn in fns:
                    names = fn.split('_')
                    traj_id = int(names[0][4:])
                    new_traj_id = id2id.get(traj_id, f"{traj_id}_drop")
                    names[0] = f"traj{new_traj_id}"
                    new_name = '_'.join(names)
                    os.rename(os.path.join(split_path, scan, fn), os.path.join(split_path, scan, new_name))
        r2r_episodes_split_from_merged, r2r_gt_split_from_merged = [], {}
        rxr_episodes_split_from_merged, rxr_gt_split_from_merged = [], {}
        merged_episodes, merged_gt = [], {}
        for i, episode in enumerate(r2r_episodes):
            episode_id = episode["episode_id"]
            gt = r2r_gt[str(episode_id)]
            episode["episode_id"] = i
            episode["trajectory_id"] = r2r_traj_eps_ids[i]
            merged_episodes.append(episode)
            merged_gt[str(i)] = gt
            r2r_episodes_split_from_merged.append(episode)
            r2r_gt_split_from_merged[str(i)] = gt
        for i, episode in enumerate(rxr_episodes):
            merged_id = i + len(r2r_episodes)
            episode_id = episode["episode_id"]
            gt = rxr_gt[str(episode_id)]
            episode["episode_id"] = merged_id
            episode["trajectory_id"] = rxr_traj_eps_ids[i] + r2r_traj_eps_ids[-1] + 1
            merged_episodes.append(episode)
            merged_gt[str(merged_id)] = gt
            rxr_episodes_split_from_merged.append(episode)
            rxr_gt_split_from_merged[str(merged_id)] = gt
        
        res = {"episodes": merged_episodes}
        res_r2r = {"episodes": r2r_episodes_split_from_merged}
        res_rxr = {"episodes": rxr_episodes_split_from_merged}
        if self.save_merge:
            with open(os.path.join(self.merge_save_path, 'merged_%s.json'%split), 'w') as f:
                json.dump(res, f, indent=2)
            with open(os.path.join(self.merge_save_path, 'merged_%s_gt.json'%split), 'w') as f:
                json.dump(merged_gt, f, indent=2)
        if self.save_split_from_merge:
            with open(os.path.join(self.r2r_split_from_merge_path, 'r2r_from_merge_%s.json'%split), 'w') as f:
                json.dump(res_r2r, f, indent=2)
            with open(os.path.join(self.r2r_split_from_merge_path, 'r2r_from_merge_%s_gt.json'%split), 'w') as f:
                json.dump(r2r_gt_split_from_merged, f, indent=2)
                
            with open(os.path.join(self.rxr_split_from_merge_path, 'rxr_from_merge_%s.json'%split), 'w') as f:
                json.dump(res_rxr, f, indent=2)
            with open(os.path.join(self.rxr_split_from_merge_path, 'rxr_from_merge_%s_gt.json'%split), 'w') as f:
                json.dump(rxr_gt_split_from_merged, f, indent=2)
        
        return res


if __name__ == "__main__":
    config = {
        "save_scan_sorted_r2r": False,
        "scan_sorted_r2r_path": R2R_SCAN_SORTED_PATH,
        "save_scan_sorted_rxr": False,
        "scan_sorted_rxr_path": RXR_SCAN_SORTED_PATH,
        "save_onefloor": False,
        "onefloor_save_path": ONE_FLOOR_SAVE_PATH,
        "save_r2r": False,
        "r2r_save_path": R2R_SAVE_PATH,
        "save_rxr": False,
        "rxr_save_path": RXR_SAVE_PATH,
        "save_merge": False,
        "merge_save_path": MERGE_SAVE_PATH,
        "save_split_from_merge": True,
        "r2r_split_from_merge_path": FloorPlan_VLN_R2R_PATH,
        "rxr_split_from_merge_path": FloorPlan_VLN_RxR_PATH,
        "plot_floorplan": False,
        "plot_traj": False,
        "plot_vertex": False,
        "extract_region_graph": False
    }
    dataset_collector = NavigationDatasetsCollector(**config)
    # dataset_collector.create_floorplans()
    for split in ["val_seen", "val_unseen"]:
        dataset_collector.merge_r2r_rxr(split)