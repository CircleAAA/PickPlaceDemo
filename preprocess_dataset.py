import collections
import numpy as np
from tqdm import trange
from src.util import read_json, save_json
from src.macro import GRIPPER_SPEED


def prepare_dataset_episode(episode_index, num_sequence_in, num_sequence_out):
    multi_view_list = collections.deque()
    action_list = collections.deque()
    
    state_path = f"data/episodes/episode_{episode_index}/gripper_state.json"
    data = read_json(state_path)
    gripper_state_list = data["gripper_state_list"]
    num_frames = len(gripper_state_list)
    for frame_index in range(num_frames):
        front_view_path = f"data/episodes/episode_{episode_index}/FrontViewCamera/{frame_index}.png"        
        side_view_path = f"data/episodes/episode_{episode_index}/SideViewCamera/{frame_index}.png"
        top_view_path = f"data/episodes/episode_{episode_index}/TopViewCamera/{frame_index}.png"
        multi_view_list.append([front_view_path, side_view_path, top_view_path])

        curr_position = gripper_state_list[frame_index][:3]
        next_position = gripper_state_list[frame_index + 1][:3] if frame_index < num_frames - 1 else curr_position
        catch_state = gripper_state_list[frame_index + 1][3] if frame_index < num_frames - 1 else 1
        task_state = gripper_state_list[frame_index + 1][4] if frame_index < num_frames - 1 else 1
        delta = (np.array(next_position) - np.array(curr_position)) / GRIPPER_SPEED
        delta_x, delta_y, delta_z = delta.tolist()
        action = [delta_x, delta_y, delta_z, catch_state, task_state]
        action_list.append(action)
    
    action_min = np.min(np.array(action_list)[:, :3], axis=0)
    action_max = np.max(np.array(action_list)[:, :3], axis=0)
    
    for _ in range(num_sequence_in - 1):
        multi_view_list.appendleft(multi_view_list[0])
    
    for _ in range(num_sequence_out - 1):
        action_list.append(action_list[-1])
    
    multi_view_list = list(multi_view_list)
    action_list = list(action_list)
    samples = []
    for index in range(num_frames):
        input_frames = multi_view_list[index: index + num_sequence_in]
        target_actions = action_list[index: index + num_sequence_out]
        samples.append({'input_frames': input_frames, 'target_actions': target_actions})
    
    return samples, action_min, action_max

        
def prepare_dataset(episode_from, episode_to, num_sequence_in, num_sequence_out, dataset_type):
    assert dataset_type in ["train", "validate"]
    samples = []
    
    action_min = np.array([np.inf] * 3)
    action_max = np.array([-np.inf] * 3)
    
    for episode_index in trange(episode_from, episode_to):
        episode_samples, episode_action_min, episode_action_max = prepare_dataset_episode(episode_index, num_sequence_in, num_sequence_out)
        samples.extend(episode_samples)
        action_min = np.minimum(episode_action_min, action_min)
        action_max = np.maximum(episode_action_max, action_max)
        
    data = {
        "samples": samples,
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist()
    }

    if dataset_type == "train":
        save_json(data, "data/data_train.json")
    else:
        save_json(data, "data/data_validate.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="prepare dataset")
    parser.add_argument("--dataset_type", type=str, choices=["train", "validate"], default="train")
    parser.add_argument("--num_sequence_in", type=int, default=6)
    parser.add_argument("--num_sequence_out", type=int, default=6)
    args = parser.parse_args()
    if args.dataset_type == "train":
        episode_from = 0
        episode_to = 1200
    else:
        episode_from = 1200
        episode_to = 1500

    prepare_dataset(episode_from=episode_from, episode_to=episode_to, dataset_type=args.dataset_type, 
                    num_sequence_in=args.num_sequence_in, num_sequence_out=args.num_sequence_out)
