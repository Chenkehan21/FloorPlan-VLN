import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


FLOORPLAN_NAVIGATION_VIDEO = ""
SAVE_PATH = ""
R2R_NAVIGATION_VIDEO = ""


def concat_videos_frame_by_frame(
    obs_video_path: str,
    floorplan_video_path: str,
    output_path: str,
    resize_to: tuple = (448, 448),
    verbose: bool = False
):
    cap_obs = cv2.VideoCapture(obs_video_path)
    cap_floor = cv2.VideoCapture(floorplan_video_path)
    fps = cap_obs.get(cv2.CAP_PROP_FPS)
    obs_frame_count = int(cap_obs.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = resize_to[0] * 2
    frame_height = resize_to[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for i in range(obs_frame_count):
        ret_obs, frame_obs = cap_obs.read()
        ret_floor, frame_floor = cap_floor.read()

        if not ret_obs or not ret_floor:
            if verbose:
                print(f"Warning: failed to load frame {i}, skip.")
            break

        frame_obs = cv2.resize(frame_obs, resize_to)
        frame_floor = cv2.resize(frame_floor, resize_to)

        combined = cv2.hconcat([frame_obs, frame_floor])
        out.write(combined)

    cap_obs.release()
    out.release()
    

def interleave_videos_frame_by_frame(
    obs_video_path: str,
    floorplan_video_path: str,
    output_path: str,
    resize_to: tuple = (448, 448),
    verbose: bool = False
):
    cap_obs = cv2.VideoCapture(obs_video_path)
    cap_floor = cv2.VideoCapture(floorplan_video_path)

    fps = cap_obs.get(cv2.CAP_PROP_FPS)
    obs_frame_count = int(cap_obs.get(cv2.CAP_PROP_FRAME_COUNT))
    floor_frame_count = int(cap_floor.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(obs_frame_count, floor_frame_count) * 2

    frame_width, frame_height = resize_to
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for i in range(total_frames):
        if i % 2 == 0:
            ret, frame = cap_obs.read()
            src = "obs"
        else:
            ret, frame = cap_floor.read()
            src = "floor"

        if not ret:
            if verbose:
                print(f"Warning: {src} failed at {i//2} frame, break.")
            break

        frame = cv2.resize(frame, resize_to)
        out.write(frame)

    cap_obs.release()
    cap_floor.release()
    out.release()


def process_scene(scene: str):
    """处理单个 scene 下的所有 traj/step"""
    scene_fp = os.path.join(R2R_NAVIGATION_VIDEO, scene)
    trajs = os.listdir(scene_fp)
    
    for traj in trajs:
        traj_fp = os.path.join(scene_fp, traj)
        steps = os.listdir(traj_fp)

        # floorplan video
        floorplan_nav_traj_fp = os.path.join(FLOORPLAN_NAVIGATION_VIDEO, scene, traj)
        floorplan_nav_video_fn = os.listdir(floorplan_nav_traj_fp)[0]
        floorplan_nav_video_fp = os.path.join(floorplan_nav_traj_fp, floorplan_nav_video_fn)

        # process all steps
        for step in steps:
            r2r_video_fp = os.path.join(traj_fp, step)
            save_path = os.path.join(SAVE_PATH, scene, traj)
            os.makedirs(save_path, exist_ok=True)
            output_path = os.path.join(save_path, step)

            concat_videos_frame_by_frame(r2r_video_fp, floorplan_nav_video_fp, output_path)

    return scene  # 返回 scene 名字方便日志


def process_scene_interleave(scene: str):
    """处理单个 scene 下的所有 traj/step"""
    scene_fp = os.path.join(R2R_NAVIGATION_VIDEO, scene)
    trajs = os.listdir(scene_fp)
    
    for traj in trajs:
        traj_fp = os.path.join(scene_fp, traj)
        steps = os.listdir(traj_fp)

        # process all steps
        for step in steps:
            r2r_video_fp = os.path.join(traj_fp, step)
            floorplan_nav_video_fp = os.path.join(FLOORPLAN_NAVIGATION_VIDEO, scene, traj, step)
            save_path = os.path.join(SAVE_PATH, scene, traj)
            os.makedirs(save_path, exist_ok=True)
            output_path = os.path.join(save_path, step)

            interleave_videos_frame_by_frame(r2r_video_fp, floorplan_nav_video_fp, output_path)

    return scene  # 返回 scene 名字方便日志


def main(max_workers: int = 8):
    scenes = os.listdir(R2R_NAVIGATION_VIDEO)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # futures = [executor.submit(process_scene, scene) for scene in scenes]
        futures = [executor.submit(process_scene_interleave, scene) for scene in scenes]

        for future in tqdm(as_completed(futures), total=len(futures), desc="scenes", ncols=80):
            try:
                scene_name = future.result()
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main(max_workers=59)  # 根据 CPU/内存情况调整
