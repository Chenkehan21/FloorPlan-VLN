import os
import json
import signal
from tqdm import tqdm
from decord import VideoReader, cpu
import imageio
from PIL import Image
import numpy as np
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


VIDEO_PATH = ""
VIDEO_SAVE_PATH = ""
GT_PATH = ""
EPISODE_PATH = ""
R2R_STOP_BALANCE_PATH = ""


def resample_video(
    input_path: str,
    output_path: str,
    keep_indices: List[int],
    generate_multiple: bool = True,
    stop_aug_images: List=None
):
    try:
        vr = VideoReader(input_path, ctx=cpu(0))
        fps = 1
        kept_frames = [vr[i].asnumpy() for i in keep_indices]
        if generate_multiple:
            for i in range(1, len(kept_frames) + 1):
                sub_frames = kept_frames[:i]
                sub_out = os.path.join(output_path, f"{i - 1}.mp4")
                with imageio.get_writer(sub_out, fps=fps) as writer:
                    for frame in sub_frames:
                        print(frame, frame.shape, frame.dtype)
                        writer.append_data(frame)
            
            if stop_aug_images:
                last_frames = kept_frames[:-1]
                for idx, aug_stop in enumerate(stop_aug_images):
                    sub_out = os.path.join(output_path, f"{len(kept_frames) - 1}_{idx+1}.mp4")
                    with imageio.get_writer(sub_out, fps=fps) as writer:
                        for frame in last_frames:
                            writer.append_data(frame)
                        writer.append_data(aug_stop)
        else:
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in kept_frames:
                    writer.append_data(frame)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def build_tasks(task_name: str, split: str) -> List[Tuple[str, str, List[int]]]:
    with open(GT_PATH, "r") as f:
        gt = json.load(f)
    with open(EPISODE_PATH, "r") as f:
        episodes = json.load(f)["episodes"]
    traj2eps = {str(item["trajectory_id"]): str(item["episode_id"]) for item in episodes}

    task_list = []
    scan_path = os.path.join(VIDEO_PATH, task_name, split)
    scans = os.listdir(scan_path)
    for scan in scans:
        traj_path = os.path.join(scan_path, scan)
        trajs = os.listdir(traj_path)
        for traj in trajs:
            step_path = os.path.join(traj_path, traj)
            step_videos = os.listdir(step_path)
            if "r2r" in task_name:
                if not step_videos:
                    continue
                max_id = max([int(item.split('.')[0]) for item in step_videos if item.endswith('.mp4')])
                video_fp = os.path.join(step_path, f"{max_id}.mp4")
            elif 'rxr' in task_name:
                assert len(step_videos) == 1
                video_fp = os.path.join(step_path, step_videos[0])

            eps_id = traj2eps.get(traj)
            if eps_id is None or eps_id not in gt:
                continue

            keep_indices = gt[eps_id]["frame_ids"]
            if "r2r" in task_name:
                stop_nums = keep_indices.count(max(keep_indices))
                keep_indices =  keep_indices[: -1 * (stop_nums - 1)]
            
            output_fp = os.path.join(VIDEO_SAVE_PATH, task_name, split, scan, traj)
            os.makedirs(output_fp, exist_ok=True)
            # stop_aug_images = [np.array(Image.open(
            #     os.path.join(R2R_STOP_BALANCE_PATH, task_name, split, scan, traj, f"{i+1}.png")
            #     ).convert("RGB")) for i in range(stop_nums - 1)]
            stop_aug_images=None
            task_list.append((video_fp, output_fp, keep_indices, True, stop_aug_images))

    return task_list


def main(task_name: str, split: str, max_workers: int = 4):
    tasks = build_tasks(task_name, split)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(resample_video, *task) for task in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
    except KeyboardInterrupt:
        print("\nstop all subprocesses...")
        executor.shutdown(wait=False)
        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)


if __name__ == "__main__":
    main("rxr", "train", max_workers=8)
