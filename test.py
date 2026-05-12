import json
import os
import torch
import cv2
import math
import random
import numpy as np
from src.util import read_json
from src.scene import SimScene
from src.model import VisionActionModel
from src.macro import FLOOR_SIZE, GRIPPER_HEIGHT, GRIPPER_SPEED, OBJECT_INIT_Z


def normalize_image(img):
    img = img.astype(np.float32)
    img = img / 255.0 * 2 - 1
    img = np.transpose(img, [2, 0, 1])
    return img


class MultiViewBuffer:
    def __init__(self, len_sequence_in, device):
        self.len_sequence_in = len_sequence_in
        self.device = device
        self.buffer = []
    
    def add(self, multi_view):
        self.buffer.append(multi_view)
        if len(self.buffer) > self.len_sequence_in:
            self.buffer.pop(0)
        
        if len(self.buffer) < self.len_sequence_in:
            self.buffer.extend([multi_view] * (self.len_sequence_in - len(self.buffer)))
            
    def get(self):
        return torch.tensor(np.array(self.buffer), dtype=torch.float32, device=self.device)
    

class ActionChunkingWithTemporalEnsemble():
    def __init__(self, max_buffer_size, len_sequence, action_dim=3):
        self.chunk_buffer = np.full((max_buffer_size, len_sequence, action_dim), np.nan, dtype=object)
    
    def get_action(self, action_sequence):
        self.chunk_buffer[:-1, :-1, :] = self.chunk_buffer[1:, 1:, :]
        self.chunk_buffer[:, -1, :] = np.nan
        self.chunk_buffer[-1, :, :] = np.array(action_sequence)
        return np.nanmean(self.chunk_buffer[:, 0, :], axis=0)


@torch.inference_mode()
def inference_episode(model, resolution, object_location_range_scale, action_chunking_buffer_size,
                      len_sequence_in, len_sequence_out, device):
    sun_rx_radian = math.pi / 3
    sun_ry_radian = math.pi / 3

    sun_density = 6.0
    background_color = (1.0, 1.0, 1.0)
    background_density = 0.5
    scene = SimScene(resolution=resolution, sun_rx_radian=sun_rx_radian, sun_ry_radian=sun_ry_radian, sun_density=sun_density, 
                     background_color=background_color, background_density=background_density)
            
    object_init_x = object_location_range_scale * FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
    object_init_y = object_location_range_scale * FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
    object_init_location = (object_init_x, object_init_y, OBJECT_INIT_Z)
    
    gripper_init_location = (-6, -6, 10)
    scene.reset(object_init_location=object_init_location, gripper_init_location=gripper_init_location)
    
    catch_state = 0
    task_state = 0
    frame_count = 0
    multi_view_buffer = MultiViewBuffer(len_sequence_in=len_sequence_in, device=device)
    action_chunking = ActionChunkingWithTemporalEnsemble(max_buffer_size=action_chunking_buffer_size, len_sequence=len_sequence_out)
    
    while task_state == 0:
        front_view_save_path = os.path.join(os.path.dirname(__file__), f"test_case/front_view/{frame_count}.png")
        scene.shot_front_view(front_view_save_path)
        front_view = cv2.imread(front_view_save_path)
        front_view = normalize_image(front_view)
        
        side_view_save_path = os.path.join(os.path.dirname(__file__), f"test_case/side_view/{frame_count}.png")
        scene.shot_side_view(side_view_save_path)
        side_view = cv2.imread(side_view_save_path)
        side_view = normalize_image(side_view)
        
        top_view_save_path = os.path.join(os.path.dirname(__file__), f"test_case/top_view/{frame_count}.png")
        scene.shot_top_view(top_view_save_path)
        top_view = cv2.imread(top_view_save_path)
        top_view = normalize_image(top_view)
        
        multi_view_buffer.add([front_view, side_view, top_view])
        multi_view = multi_view_buffer.get().unsqueeze(0)
        
        action_pd = model(multi_view).squeeze(0)
        action_pd[:, 3:] = torch.sigmoid(action_pd[:, 3:])
        action_pd = action_pd.cpu().numpy()
        
        catch_state = 1 if action_pd[0, 3] > 0.5 else 0
        task_state = 1 if action_pd[0, 4] > 0.5 else 0
        action = action_chunking.get_action(action_pd[:, :3])
        dx, dy, dz = action * GRIPPER_SPEED

        if catch_state:
            scene.move_gripper(dx, dy, dz)
            scene.move_object(dx, dy, dz)
        else:
            scene.move_gripper(dx, dy, dz)
        frame_count += 1
        

def main():
    resolution = 256
    object_location_range_scale = 0.7
    action_chunking_buffer_size = 4
    len_sequence_in = 6
    len_sequence_out = 6
    device = "cuda"
    
    os.makedirs("test_case/front_view", exist_ok=True)
    os.makedirs("test_case/side_view", exist_ok=True)
    os.makedirs("test_case/top_view", exist_ok=True)
    
    checkpoint_pth = "checkpoint.pth"
    model = VisionActionModel(num_views=3, len_sequence_out=len_sequence_out, action_dim=5, d_model=128, num_heads=8, num_layers=2, dropout=0.2).to(device)
    assert os.path.exists(checkpoint_pth)
    model_state = torch.load(checkpoint_pth)
    model.load_state_dict(model_state)
        
    inference_episode(model=model, resolution=resolution, object_location_range_scale=object_location_range_scale, 
                      action_chunking_buffer_size=action_chunking_buffer_size, len_sequence_in=len_sequence_in, 
                      len_sequence_out=len_sequence_out, device=device)


if __name__ == "__main__":
    main()
