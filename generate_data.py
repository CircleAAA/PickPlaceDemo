import os
import random
import math
import numpy as np
from src.macro import *
from src.util import normalize_vector, save_json, read_json
from src.scene import SimScene


def record_episode(episode_save_dir, scene):
    front_view_dir = os.path.join(episode_save_dir, "FrontViewCamera")
    os.makedirs(front_view_dir, exist_ok=True)
    
    side_view_dir = os.path.join(episode_save_dir, "SideViewCamera")
    os.makedirs(side_view_dir, exist_ok=True)
    
    top_view_dir = os.path.join(episode_save_dir, "TopViewCamera")
    os.makedirs(top_view_dir, exist_ok=True)
    
    gripper_state_list = []
    frame_count = 0

    while True:
        gripper_to_pick = scene.gripper_to_pick_location()
        if np.linalg.norm(gripper_to_pick) < GRIPPER_SPEED:
            break

        front_view_save_path = os.path.join(front_view_dir, f"{frame_count}.png")
        scene.shot_front_view(front_view_save_path)
        side_view_save_path = os.path.join(side_view_dir, f"{frame_count}.png")
        scene.shot_side_view(side_view_save_path)
        top_view_save_path = os.path.join(top_view_dir, f"{frame_count}.png")
        scene.shot_top_view(top_view_save_path)

        gripper_state_list.append([*scene.gripper.location, 0, 0])
        
        action = GRIPPER_SPEED * normalize_vector(gripper_to_pick)
        scene.move_gripper(dx=action[0], dy=action[1], dz=action[2])
        frame_count += 1

    catch_frame = frame_count

    while True:
        gripper_to_place = scene.gripper_to_place_location()
        if np.linalg.norm(gripper_to_place) < GRIPPER_SPEED:
            break

        front_view_save_path = os.path.join(front_view_dir, f"{frame_count}.png")
        scene.shot_front_view(front_view_save_path)
        side_view_save_path = os.path.join(side_view_dir, f"{frame_count}.png")
        scene.shot_side_view(side_view_save_path)
        top_view_save_path = os.path.join(top_view_dir, f"{frame_count}.png")
        scene.shot_top_view(top_view_save_path)

        gripper_state_list.append([*scene.gripper.location, 1, 0])

        action = GRIPPER_SPEED * normalize_vector(gripper_to_place)
        scene.move_gripper(dx=action[0], dy=action[1], dz=action[2])
        scene.move_object(dx=action[0], dy=action[1], dz=action[2])
        frame_count += 1

    task_frame = frame_count - 1
    gripper_state_list[-3][-1] = 1
    gripper_state_list[-2][-1] = 1
    gripper_state_list[-1][-1] = 1
    data = {
        "gripper_state_list": gripper_state_list,
        "object_init_location": scene.object_init_location,
        "gripper_init_location": scene.gripper_init_location,
        "catch_frame": catch_frame,
        "task_frame": task_frame
    }

    gripper_state_save_path = os.path.join(episode_save_dir, "gripper_state.json")
    save_json(data, gripper_state_save_path)


def generate_episodes_subprocess(episode_indices, resolution=256, object_location_range_scale=0.8):
    sun_rx_radian = math.pi / 3
    sun_ry_radian = math.pi / 3

    sun_density = 6.0
    background_color = (1.0, 1.0, 1.0)
    background_density = 0.5
    scene = SimScene(resolution=resolution, sun_rx_radian=sun_rx_radian, sun_ry_radian=sun_ry_radian, sun_density=sun_density, 
                     background_color=background_color, background_density=background_density)
            
    for episode_index in episode_indices:
        object_init_x = object_location_range_scale * FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
        object_init_y = object_location_range_scale * FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
        object_init_location = (object_init_x, object_init_y, OBJECT_INIT_Z)
        
        gripper_init_x = -6
        gripper_init_y = -6
        gripper_init_z = 10
        gripper_init_location = (gripper_init_x, gripper_init_y, gripper_init_z)
        
        scene.reset(object_init_location=object_init_location, gripper_init_location=gripper_init_location)

        episode_save_dir = os.path.join(os.path.dirname(__file__), "data/episodes", f"episode_{episode_index}")
        record_episode(episode_save_dir, scene)


def main(episode_from, episode_to, resolution=256, object_location_range_scale=0.8):
    episode_indices = list(range(episode_from, episode_to))
    generate_episodes_subprocess(episode_indices, resolution=resolution, object_location_range_scale=object_location_range_scale)
        
        
if __name__ == "__main__":
    # generate train data
    main(episode_from=1085, episode_to=1200, object_location_range_scale=0.8)
    # generate test data
    main(episode_from=1200, episode_to=1500, object_location_range_scale=0.7)