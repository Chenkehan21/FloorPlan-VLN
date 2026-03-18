import os
import cv2
from PIL import Image
from typing import List
from concurrent.futures import ThreadPoolExecutor


VIDEO_PATH = ""
VIDEO_SAVE_PATH = ""

def split_video(images: List, save_path: str, save_images: bool=False):
    os.makedirs(save_path, exist_ok=True)
    batch_save_videos(images, save_path)
    
    # save step-wise videos
    # total_images = len(images)
    # for step in range(total_images):
    #     samples = images[:step + 1]
    #     if save_images:
    #         current_save_path = os.path.join(save_path, str(step))
    #         os.makedirs(current_save_path, exist_ok=True)
    #         batch_save_images(samples, current_save_path)
    #     else:
    #         batch_save_videos(samples, save_path)

def batch_save_images(images: List, path: str):
    for i, img in enumerate(images):
        img.save(os.path.join(path, f"{i}.png"))

def batch_save_videos(images: List, path: str, fps:int=1):
    name = len(images) - 1
    output_video_path = os.path.join(path, f"{name}.mp4")
    frame = images[0]
    height, width, channel = frame.shape
    size = (width, height)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for image in images:
        out.write(image)

    out.release()

def process_traj_images(scene, traj, task, split, img_fns: List[str], save_images:bool) -> List:
    images = []
    for img_fn in img_fns:
        img_path = os.path.join(VIDEO_PATH, task, split, scene, traj, img_fn)
        if save_images:
            with Image.open(img_path) as img:
                images.append(img.copy())
        else:
            img = cv2.imread(img_path)
            images.append(img.copy())
    return images

def process_scene_traj(scene: str, traj: str, task: str, split: str, save_images:bool):
    img_fns = os.listdir(os.path.join(VIDEO_PATH, task, split, scene, traj))
    img_fns = sorted(img_fns, key=lambda x: int(x.split('.')[0]))
    print(f"Processing {scene}/{traj}")
    
    images = process_traj_images(scene, traj, task, split, img_fns, save_images)
    
    save_path = os.path.join(VIDEO_SAVE_PATH, task, split, scene, traj)
    split_video(images, save_path, save_images)

def main(task: str = "r2r", split: str = "val_unseen", save_images:bool=False):
    scenes = os.listdir(os.path.join(VIDEO_PATH, task, split))
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for scene in scenes:
            trajs = os.listdir(os.path.join(VIDEO_PATH, task, split, scene))
            for traj in trajs:
                futures.append(executor.submit(process_scene_traj, scene, traj, task, split, save_images))
        
        for future in futures:
            future.result()

if __name__ == "__main__":
    main(task='rxr', split='train')
