import cv2
import numpy as np
from torch.utils.data import Dataset
from src.util import read_json


class PickPlaceDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = read_json(data_path)
        self.action_min = np.array(data["action_min"], dtype=np.float32)
        self.action_max = np.array(data["action_max"], dtype=np.float32)
        samples = data["samples"]
        self.input_frames = []
        self.target_actions = []
        for sample in samples:
            input_frames = sample["input_frames"]
            target_actions = sample["target_actions"]
            self.input_frames.append(input_frames)
            self.target_actions.append(target_actions)
        
        self.target_actions = np.array(self.target_actions)
        self.length = len(self.input_frames)

    def __len__(self):
        return self.length

    @staticmethod
    def get_image(img_path):
        img = cv2.imread(img_path)
        img = img.astype(np.float32)
        img = img / 255.0 * 2 - 1
        img = np.transpose(img, [2, 0, 1])
        img = img.reshape((1, *img.shape))
        return img
            
    def get_multi_view_sequence(self, index):
        multi_view_sequence = []
        for front_view_path, side_view_path, top_view_path in self.input_frames[index]:
            front_view = self.get_image(front_view_path)
            side_view = self.get_image(side_view_path)
            top_view = self.get_image(top_view_path)
            multi_view = np.concatenate([front_view, side_view, top_view], axis=0)
            multi_view = multi_view.reshape((1, *multi_view.shape))
            multi_view_sequence.append(multi_view)
        multi_view_sequence = np.concatenate(multi_view_sequence, axis=0)
        return multi_view_sequence

    def __getitem__(self, index):
        return self.get_multi_view_sequence(index), self.target_actions[index]
