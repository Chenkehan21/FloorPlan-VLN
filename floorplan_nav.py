import os
import cv2
import json
import gzip
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from itertools import groupby
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation as R


FLOORPLAN_TRAJ_PATH = "/data/ckh/layout_diagram/tests/"
MP3D_FLOORPLAN_PATH = "/data/ckh/layout_diagram/data/mp3d_floorplan/"
NAVIGATION_DATASET_PATH = "/data/ckh/layout_diagram/data/navigation_datasets"

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
# random.shuffle(COLOR_PALETT)
ROOM_COLORS = dict(zip(ROOM_LABELS.values(), COLOR_PALETT))


@dataclass
class FloorplanTransform:
    region_min: np.ndarray    # 形状 (2,)
    region_max: np.ndarray    # 形状 (2,)
    image_res: np.ndarray     # 形状 (2,) [H', W']
    delta: np.ndarray         # 形状 (2,)
    noise: np.ndarray = None   # 形状 (2,)，可选的噪声项
    height: int = 448
    width: int = 448

    def apply(self, pts: np.ndarray, add_noise_to_region: bool=False) -> np.ndarray:
        pts = (pts - self.region_min) / (self.region_max - self.region_min)
        
        if self.noise is not None and add_noise_to_region:
            pts = pts + self.noise
        
        pts = pts * self.image_res
        pts = pts + self.delta
        
        pts = np.clip(pts, 0,  [self.height - 1, self.width - 1])  # 确保噪声后仍在图像范围内
        
        return np.round(pts).astype(int)


class FloorplanNavigator:
    def __init__(self):
        self.poly_noise = None
    
    def reset(self):
        self.poly_noise = None
    
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
        
    def compute_floorplan_transform(self, regions: List[np.ndarray], height: int, width: int,
                                    use_poly_noise=False, use_poly_drop=False, 
                                    poly_noise_mu=2, poly_noise_sigma=0.05, poly_drop_ratio=0.05, min_vertexs=8) -> FloorplanTransform:
        if use_poly_drop:
            for region in regions:
                if len(region) > min_vertexs:
                    num_pts = len(region)
                    drop_num = int(num_pts * poly_drop_ratio)
                    drop_indices = np.random.choice(num_pts, size=drop_num, replace=False)
                    region = np.delete(region, drop_indices, axis=0)

        # 1. 合并所有 region 点
        all_pts = np.concatenate(regions, axis=0)[:, :2]
        if use_poly_noise:
            if self.poly_noise is None:
                self.poly_noise = np.random.normal(poly_noise_mu, poly_noise_sigma, size=all_pts.shape)

        image_res = np.array([height, width], dtype=np.float32)

        # 2. 计算包围盒 (只基于 regions)
        region_min = np.min(all_pts, axis=0)
        region_max = np.max(all_pts, axis=0)
        max_min = region_max - region_min

        # padding 1%
        region_min -= 0.01 * max_min
        region_max += 0.01 * max_min
        max_min = region_max - region_min

        # 3. 维持纵横比（假设画布接近正方形）
        min_id = np.argmin(max_min)
        ratio = np.min(max_min) / np.max(max_min)
        image_res[min_id] *= ratio

        # 4. 中心对齐需要先算出归一化后的中心
        pts_norm = (all_pts - region_min) / (region_max - region_min)
        pts_img = pts_norm * image_res
        floorplan_center = np.array([width, height], dtype=np.float32) / 2.
        region_center = (np.max(pts_img, axis=0) + np.min(pts_img, axis=0)) / 2.
        delta = floorplan_center - region_center

        return FloorplanTransform(
            region_min=region_min,
            region_max=region_max,
            image_res=image_res,
            height=height,
            width=width,
            delta=delta,
            noise=self.poly_noise
        )

    def transform_regions(self, tf, regions: List[np.ndarray], add_noise_to_region: bool=False) -> List:
        region_lengths = [len(r) for r in regions]
        all_regions = np.concatenate(regions, axis=0)[:, :2]
        all_regions_img = tf.apply(all_regions, add_noise_to_region=add_noise_to_region)

        split_indices = np.cumsum(region_lengths[:-1])
        regions_tf = np.split(all_regions_img, split_indices)
        
        return regions_tf
    
    def transform_traj(self, tf, traj: np.ndarray) -> np.ndarray:
        traj = traj[..., [0, 2]].copy()
        traj[..., 1] *= -1
        traj_tf = tf.apply(traj)
        
        return traj_tf
    
    def plot_floorplan(self, regions: List[np.ndarray], room_labels: List, 
                       traj: np.ndarray, region_ids: List[int], heading: float, ratio: int=448, scale: int=448, traj_color=(247,247,100)):
        """Draw floorplan map where different colors indicate different rooms
        """
        regions = [(region * ratio).round().astype(int) for region in regions]
        traj = (traj * ratio).round().astype(int)
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
        colorMap[0] = np.array([255, 255, 255], dtype=np.uint8)  # set background to white
        room_map = np.zeros([scale, scale]).astype(np.int32)
        for idx, polygon in enumerate(regions):
            cv2.fillPoly(room_map, [polygon], color=idx + 1)
        
        image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 3))
        lineColor = (0,0,0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        
        for region in regions:
            for i, point in enumerate(region):
                if i == len(region)-1:
                    cv2.line(image, tuple(point), tuple(region[0]), color=lineColor, thickness=1, lineType=cv2.LINE_AA)
                else:    
                    cv2.line(image, tuple(point), tuple(region[i+1]), color=lineColor, thickness=1, lineType=cv2.LINE_AA)
        for i in range(len(traj)):
            if i + 1 <= len(traj) - 1:
                cv2.line(image, tuple(traj[i]), tuple(traj[i + 1]), color=traj_color, thickness=2, lineType=cv2.LINE_AA)

        pose = (traj[-1][0], traj[-1][1], heading)
        agent_arrow = self.get_contour_points(pose, (0,0), size=10)
        cv2.drawContours(image, [agent_arrow], 0, (247,247,100), -1) # draw agent arrow
        cv2.drawContours(image, [agent_arrow], 0, (0,0,0), 1) # draw agent arrow
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
            center_flipped = (scale - center[0], center[1])
            radius = dist_map[best_point[0], best_point[1]]

            font = cv2.FONT_HERSHEY_DUPLEX
            font_thickness = 1
            font_scale = 0.4
            text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
            text_w, text_h = text_size

            while (text_w**2 + text_h**2)**0.5 / 2 > radius * 0.8:
                font_scale -= 0.1
                if font_scale < 0.1:
                    break
                text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
                text_w, text_h = text_size
            if font_scale > 0.2:
                text_org = (scale - center[0] - text_w // 2, center[1] + text_h // 2)
                top_left = (text_org[0], text_org[1] - text_h)
                bottom_right = (text_org[0] + text_w, text_org[1] + int(text_h * 0.3))

                cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 0), thickness=-1)

                cv2.putText(image, region_id, text_org, font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        
        return image
    
    def plot_floorplan_alpha(self, regions: List[np.ndarray], room_labels: List, 
                       traj: np.ndarray, region_ids: List[int], heading: float, ratio: int=448, scale: int=448, traj_color=(247,247,100,255)):
        """Draw floorplan map where different colors indicate different rooms
        """
        regions = [(region * ratio).round().astype(int) for region in regions]
        traj = (traj * ratio).round().astype(int)
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
        colorMap[0] = np.array([255, 255, 255], dtype=np.uint8)  # set background to white
        
        alpha_channels = np.zeros(colorMap.shape[0], dtype=np.uint8)
        alpha_channels[1:len(regions) + 1] = 150
        colorMap = np.concatenate([colorMap, np.expand_dims(alpha_channels, axis=-1)], axis=-1)
        
        room_map = np.zeros([scale, scale]).astype(np.int32)
        for idx, polygon in enumerate(regions):
            cv2.fillPoly(room_map, [polygon], color=idx + 1)
        
        image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 4))
        lineColor = (0,0,0,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        
        for region in regions:
            for i, point in enumerate(region):
                if i == len(region)-1:
                    cv2.line(image, tuple(point), tuple(region[0]), color=lineColor, thickness=1, lineType=cv2.LINE_AA)
                else:    
                    cv2.line(image, tuple(point), tuple(region[i+1]), color=lineColor, thickness=1, lineType=cv2.LINE_AA)
        for i in range(len(traj)):
            if i + 1 <= len(traj) - 1:
                cv2.line(image, tuple(traj[i]), tuple(traj[i + 1]), color=traj_color, thickness=2, lineType=cv2.LINE_AA)
        pose = (traj[-1][0], traj[-1][1], heading)
        agent_arrow = self.get_contour_points(pose, (0,0), size=10)
        cv2.drawContours(image, [agent_arrow], 0, (247,247,100, 255), -1) # draw agent arrow
        cv2.drawContours(image, [agent_arrow], 0, (0,0,0, 255), 1) # draw agent arrow
        image = np.ascontiguousarray(np.fliplr(image))
        
        for i, region in enumerate(regions):
            region_id = str(region_ids[i])
            mask = np.zeros((scale, scale), dtype=np.uint8)
            cv2.fillPoly(mask, [region], 255)

            dist_map = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_map)

            # 计算几何中心
            centroid = np.mean(region, axis=0).astype(int)
            cx, cy = centroid

            # 搜索所有接近最大值的点（避免局部极值）
            candidate_yx = np.argwhere(dist_map >= max_val * 0.95)  # shape: (K, 2)

            # 选出离几何中心最近的点作为新的圆心
            dists = np.linalg.norm(candidate_yx - [cy, cx], axis=1)
            best_idx = np.argmin(dists)
            best_point = candidate_yx[best_idx]
            center = (best_point[1], best_point[0])  # x, y 格式
            center_flipped = (scale - center[0], center[1])
            radius = dist_map[best_point[0], best_point[1]]

            # 文本缩放和写入（略微调整）
            font = cv2.FONT_HERSHEY_DUPLEX
            font_thickness = 1
            font_scale = 0.4
            text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
            text_w, text_h = text_size

            while (text_w**2 + text_h**2)**0.5 / 2 > radius * 0.8:
                font_scale -= 0.1
                if font_scale < 0.1:
                    break
                text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
                text_w, text_h = text_size
            if font_scale > 0.2:
                # 做了 np.fliplr，x 坐标要翻转
                text_org = (scale - center[0] - text_w // 2, center[1] + text_h // 2)
                # cv2.putText(image, region_id, text_org, font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
                top_left = (text_org[0], text_org[1] - text_h)
                bottom_right = (text_org[0] + text_w, text_org[1] + int(text_h * 0.3))  # 下方预留一点空间

                # 先画黑底矩形
                cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 0,255), thickness=-1)

                # 再画白色文字
                cv2.putText(image, region_id, text_org, font, font_scale, (255, 255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        
        return image

    def plot_floorplan_with_multiple_trajs(self, regions: List[np.ndarray], room_labels: List, 
                       trajs: List[np.ndarray], region_ids: List[int], heading: List[float], 
                       ratio: int=448, scale: int=448, traj_colors=[(247,247,100,255)]):
        """Draw floorplan map where different colors indicate different rooms
        """
        regions = [(region * ratio).round().astype(int) for region in regions]
        trajs = [(traj * ratio).round().astype(int) for traj in trajs]
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
        colorMap[0] = np.array([255, 255, 255], dtype=np.uint8)  # set background to white
        alpha_channels = np.zeros(colorMap.shape[0], dtype=np.uint8)
        alpha_channels[1:len(regions) + 1] = 150
        colorMap = np.concatenate([colorMap, np.expand_dims(alpha_channels, axis=-1)], axis=-1)
        
        room_map = np.zeros([scale, scale]).astype(np.int32)
        for idx, polygon in enumerate(regions):
            cv2.fillPoly(room_map, [polygon], color=idx + 1)
        
        image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 4))
        lineColor = (0,0,0,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        
        for region in regions:
            for i, point in enumerate(region):
                if i == len(region)-1:
                    cv2.line(image, tuple(point), tuple(region[0]), color=lineColor, thickness=1, lineType=cv2.LINE_AA)
                else:    
                    cv2.line(image, tuple(point), tuple(region[i+1]), color=lineColor, thickness=1, lineType=cv2.LINE_AA)
        for i, traj in enumerate(trajs):
            traj_color = traj_colors[i] if i < len(traj_colors) else traj_colors[-1]
            thickness = 4-i
            if i == 0:
                size = 12
                thickness = 3
            else:
                size = 8
                thickness = 2
            
            for i in range(len(traj)):
                if i + 1 <= len(traj) - 1:
                    cv2.line(image, tuple(traj[i]), tuple(traj[i + 1]), color=traj_color, thickness=thickness, lineType=cv2.LINE_AA)

            pose = (traj[-1][0], traj[-1][1], heading)
            agent_arrow = self.get_contour_points(pose, (0,0), size=size)
            cv2.drawContours(image, [agent_arrow], 0, traj_color, -1) # draw agent arrow
            cv2.drawContours(image, [agent_arrow], 0, (0,0,0,255), 1) # draw agent arrow
        image = np.ascontiguousarray(np.fliplr(image))
        
        for i, region in enumerate(regions):
            region_id = str(region_ids[i])
            mask = np.zeros((scale, scale), dtype=np.uint8)
            cv2.fillPoly(mask, [region], 255)

            dist_map = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_map)

            # 计算几何中心
            centroid = np.mean(region, axis=0).astype(int)
            cx, cy = centroid

            # 搜索所有接近最大值的点（避免局部极值）
            candidate_yx = np.argwhere(dist_map >= max_val * 0.95)  # shape: (K, 2)

            # 选出离几何中心最近的点作为新的圆心
            dists = np.linalg.norm(candidate_yx - [cy, cx], axis=1)
            best_idx = np.argmin(dists)
            best_point = candidate_yx[best_idx]
            center = (best_point[1], best_point[0])  # x, y 格式
            center_flipped = (scale - center[0], center[1])
            radius = dist_map[best_point[0], best_point[1]]

            # 文本缩放和写入（略微调整）
            font = cv2.FONT_HERSHEY_DUPLEX
            font_thickness = 1
            font_scale = 0.4
            text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
            text_w, text_h = text_size

            while (text_w**2 + text_h**2)**0.5 / 2 > radius * 0.8:
                font_scale -= 0.1
                if font_scale < 0.1:
                    break
                text_size, _ = cv2.getTextSize(region_id, font, font_scale, font_thickness)
                text_w, text_h = text_size
            if font_scale > 0.2:
                # 做了 np.fliplr，x 坐标要翻转
                text_org = (scale - center[0] - text_w // 2, center[1] + text_h // 2)
                # cv2.putText(image, region_id, text_org, font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
                top_left = (text_org[0], text_org[1] - text_h)
                bottom_right = (text_org[0] + text_w, text_org[1] + int(text_h * 0.3))  # 下方预留一点空间

                # 先画黑底矩形
                cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 0, 255), thickness=-1)

                # 再画白色文字
                cv2.putText(image, region_id, text_org, font, font_scale, (255, 255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

        return image
    
    def plot_floorplan_traj(self, floorplan, level_id, gt_path, heading, alpha=False):
        levels = floorplan["levels"].get(level_id, floorplan["levels"]["0"])
        regions = levels["regions"]
        region_polys = []
        region_labels = []
        for region in regions.values():
            boundaries = np.array(region["boundaries"]).astype(np.float64)
            region_polys.append(boundaries)
            region_labels.append(region["label"])
        
        region_polys, traj = self.transforms(region_polys, gt_path, height=448, width=448)
        if alpha:
            img = self.plot_floorplan_alpha(region_polys, region_labels, traj, list(regions.keys()), heading, ratio=1, scale=448)
        else:
            img = self.plot_floorplan(region_polys, region_labels, traj, list(regions.keys()), heading, ratio=1, scale=448)
        # traj_path =os.path.join(save_path, f"{traj_id}")
        # os.makedirs(traj_path, exist_ok=True)
        # img_path = os.path.join(traj_path, f"{step}.png")
        # cv2.imwrite(img_path, img)
        
        return img

    def plot_floorplan_traj_scale_noise(self, floorplan, level_id, gt_path, heading, 
                                        scale_noise=1.0, height=448, width=448, check=False):
        levels = floorplan["levels"].get(level_id, floorplan["levels"]["0"])
        regions = levels["regions"]
        region_polys = []
        region_labels = []
        for region in regions.values():
            boundaries = np.array(region["boundaries"]).astype(np.float64)
            region_polys.append(boundaries)
            region_labels.append(region["label"])
        
        tf = self.compute_floorplan_transform(region_polys, height, width)
        height_noise = int(height * scale_noise)
        width_noise = int(width * scale_noise)
        tf_noise = self.compute_floorplan_transform(region_polys, height_noise, width_noise)
        
        region_polys_tf = self.transform_regions(tf, region_polys)
        traj_tf_noise = self.transform_traj(tf_noise, gt_path)
        print(traj_tf_noise)
        img = self.plot_floorplan(region_polys_tf, region_labels, traj_tf_noise, list(regions.keys()), heading, ratio=1, scale=448)
        if check:
            traj_tf = self.transform_traj(tf, gt_path)
            img_without_noise = self.plot_floorplan(region_polys_tf, region_labels, traj_tf, 
                                                    list(regions.keys()), heading, ratio=1, scale=448, traj_color=(0,0,255))
            alpha = 0.4
            result = cv2.addWeighted(img, 1 - alpha, img_without_noise, alpha, 0)
        
            return img, result
        else:
            return img
    
    
    def plot_floorplan_traj_with_noise(self, floorplan, level_id, gt_path, heading, 
                                       height=448, width=448, check=False, alpha=False,
                                       use_scale_noise=False, scale_noise_value=1.0, 
                                       use_poly_noise=False, use_poly_drop=False, 
                                       poly_noise_mu=2, poly_noise_sigma=0.05, poly_drop_ratio=0.05):
        levels = floorplan["levels"].get(level_id, floorplan["levels"]["0"])
        regions = levels["regions"]
        region_polys = []
        region_labels = []
        for region in regions.values():
            boundaries = np.array(region["boundaries"]).astype(np.float64)
            region_polys.append(boundaries)
            region_labels.append(region["label"])
        
        tf = self.compute_floorplan_transform(region_polys, height, width, 
                                              use_poly_noise=use_poly_noise, use_poly_drop=use_poly_drop,
                                              poly_noise_mu=poly_noise_mu, poly_noise_sigma=poly_noise_sigma, poly_drop_ratio=poly_drop_ratio)
        region_polys_tf = self.transform_regions(tf, region_polys, use_poly_noise)
        
        if use_scale_noise:
            height_noise = int(height * scale_noise_value)
            width_noise = int(width * scale_noise_value)
            tf_noise = self.compute_floorplan_transform(region_polys, height_noise, width_noise)
            traj_tf = self.transform_traj(tf_noise, gt_path)
        else:
            traj_tf = self.transform_traj(tf, gt_path)
        
        if alpha:            
            img = self.plot_floorplan_alpha(region_polys_tf, region_labels, traj_tf, list(regions.keys()), heading, ratio=1, scale=448)
        else:
            img = self.plot_floorplan(region_polys_tf, region_labels, traj_tf, list(regions.keys()), heading, ratio=1, scale=448)
        
        if check:
            traj_tf = self.transform_traj(tf, gt_path)
            img_without_noise = self.plot_floorplan(region_polys_tf, region_labels, traj_tf, 
                                                    list(regions.keys()), heading, ratio=1, scale=448, traj_color=(0,0,255))
            alpha = 0.4
            result = cv2.addWeighted(img, 1 - alpha, img_without_noise, alpha, 0)
        
            return result
        else:
            return img

    def visualize_floorplan_multi_trajs_with_noise(self, floorplan, level_id, gt_path, heading, 
                                       height=448, width=448, check=False, trajs=None,
                                       use_scale_noise=False, scale_noise_values=1.0, 
                                       use_poly_noise=False, use_poly_drop=False, 
                                       poly_noise_mu=2, poly_noise_sigma=0.05, poly_drop_ratio=0.05):
        levels = floorplan["levels"].get(level_id, floorplan["levels"]["0"])
        regions = levels["regions"]
        region_polys = []
        region_labels = []
        for region in regions.values():
            boundaries = np.array(region["boundaries"]).astype(np.float64)
            region_polys.append(boundaries)
            region_labels.append(region["label"])
        
        tf = self.compute_floorplan_transform(region_polys, height, width, 
                                              use_poly_noise=use_poly_noise, use_poly_drop=use_poly_drop,
                                              poly_noise_mu=poly_noise_mu, poly_noise_sigma=poly_noise_sigma, poly_drop_ratio=poly_drop_ratio)
        region_polys_tf = self.transform_regions(tf, region_polys, use_poly_noise)
        traj_tf = self.transform_traj(tf, gt_path)
        traj_tfs = [traj_tf]
        if trajs is not None:
            for traj in trajs:
                traj_tf = self.transform_traj(tf, traj)
                traj_tfs.append(traj_tf)
            img = self.plot_floorplan_with_multiple_trajs(region_polys_tf, region_labels, traj_tfs, list(regions.keys()), 
                                                          heading, ratio=1, scale=448,
                                                          traj_colors=[(247,247,100,255), (0,0,255,180), (255,0,0,180), (0,255,0,180)])
        if use_scale_noise:
            for scale_noise_value in scale_noise_values:
                height_noise = int(height * scale_noise_value)
                width_noise = int(width * scale_noise_value)
                tf_noise = self.compute_floorplan_transform(region_polys, height_noise, width_noise)
                traj_tf = self.transform_traj(tf_noise, gt_path)
                traj_tfs.append(traj_tf)
            img = self.plot_floorplan_with_multiple_trajs(region_polys_tf, region_labels, traj_tfs, list(regions.keys()), 
                                                          heading, ratio=1, scale=448,
                                                          traj_colors=[(247,247,100,255), (0,0,255,180), (255,0,0,180), (0,255,0,180)])

        return img
        
    def get_contour_points(self, pos, origin, size=5):
        x, y, o = pos
        pt1 = (int(x) + origin[0],
            int(y) + origin[1])
        pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
            int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
        pt3 = (int(x + size * np.cos(o)) + origin[0],
            int(y + size * np.sin(o)) + origin[1])
        pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
            int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

        return np.array([pt1, pt2, pt3, pt4])
    

def main(task="r2r_from_merge", split="train"):
    fp_nav = FloorplanNavigator()
    with open(os.path.join(NAVIGATION_DATASET_PATH, task, f"{task}_{split}.json"), 'r') as f:
        episodes = json.load(f)["episodes"]
    
    with open(os.path.join(NAVIGATION_DATASET_PATH, "combine_actions", f"combine_actions_{task}_{split}_gt.json"), "r") as f:
        episodes_gt = json.load(f)
    
    split_path = os.path.join(FLOORPLAN_TRAJ_PATH, "floorplan_traj_%s_navigationv2"%task, split)
    
    for eps in tqdm(episodes):
        eps_id = eps["episode_id"]
        traj_id = eps["trajectory_id"]
        scene = eps["scene_id"].split('/')[1]
        if traj_id == 581:
            fp_path = os.path.join(MP3D_FLOORPLAN_PATH, scene, "floorplan.json")
            with open(fp_path, 'r') as f:
                floorplan = json.load(f)
            save_path = os.path.join(split_path, scene)
            os.makedirs(save_path, exist_ok=True)
            
            start_position = eps["start_position"]
            start_rotation = eps["start_rotation"]
            r = R.from_quat(start_rotation)
            angle_radians = r.as_euler('zyx', degrees=False)
            angle_degrees = r.as_euler('zyx', degrees=True)
            change_direction = (angle_degrees[0] == -180 and angle_degrees[2] == 180)
            if change_direction:
                heading = -1 * angle_radians[1] - math.pi * 0.5
            else:
                heading = angle_radians[1] + math.pi * 0.5
            # print(traj_id, heading, math.degrees(heading))
            level = eps["level"]
            traj_regions = eps["traj_regions"]
            gt_path = np.array(episodes_gt[str(eps_id)]["locations"]).astype(float)
            actions = episodes_gt[str(eps_id)]["actions"]
            fp_nav.plot_floorplan_traj(floorplan, level, gt_path[:1], heading, save_path, traj_id, 0)
            for step, action in enumerate(actions):
                current_path = gt_path[: step + 2]
                if action == 2:
                    heading += math.radians(15)
                elif action == 3:
                    heading -= math.radians(15)
                elif action == 6:
                    heading += math.radians(30)
                elif action == 7:
                    heading -= math.radians(30)
                fp_nav.plot_floorplan_traj(floorplan, level, current_path, heading, save_path, traj_id, step + 1)
            # grouped_actions = [list(group) for key, group in groupby(actions)]
            # start_rotations = []
            # for group_actions in grouped_actions:
            #     if group_actions[0] in [0, 1, 4, 5]:
            #         break
            #     else:
            #         start_rotations.extend(group_actions)
            # for action in start_rotations:
            #     if action in [0, 1, 4, 5]:
            #         break
            #     elif action == 2:
            #         heading += math.radians(15)
            #     elif action == 3:
            #         heading -= math.radians(15)
            #     elif action == 6:
            #         heading += math.radians(30)
            #     elif action == 7:
            #         heading -= math.radians(30)
            # fp_nav.plot_floorplan_traj(floorplan, level, gt_path, save_path, traj_id, heading)

if __name__ == "__main__":
    main()