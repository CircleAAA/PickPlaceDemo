import os
import cv2
from tqdm import trange


for episode_index in trange(600):
    for view in ["FrontViewCamera", "SideViewCamera", "TopViewCamera"]:
        img_dir = f"data/episodes/episode_{episode_index}/{view}"
        for img_filename in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(img_path, img)
