import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import VisionActionModel
from src.dataset import PickPlaceDataset
from tqdm import tqdm


def gripper_state_loss(action_gt, action_pd):
    action_loss = nn.L1Loss()(action_pd[:, :, :3], action_gt[:, :, :3])   # (pred, target)
    io_loss = nn.BCEWithLogitsLoss()(action_pd[:, :, 3:], action_gt[:, :, 3:])   # (input, target)
    return 0.6 * action_loss + 0.4 * io_loss


@torch.no_grad()
def validate(model, dataloader_validate, device):
    model.eval()
    loss_list = []
    for multi_view_sequence, action_gt in dataloader_validate:
        multi_view_sequence = multi_view_sequence.to(device)
        action_gt = action_gt.to(device)
        action_pd = model(multi_view_sequence)
        loss = gripper_state_loss(action_gt, action_pd)
        loss_list.append(loss.item())
    model.train()
    return sum(loss_list) / len(loss_list)


def train():
    device = "cuda"
    batch_size = 64
    num_epochs = 200
    learning_rate = 1e-3
    weight_decay = 0.01
    checkpoint_pth = "checkpoint.pth"
    save_every = 500
    validate_every = 1000

    model = VisionActionModel(num_views=3, len_sequence_out=6, action_dim=5, d_model=128, num_heads=8, num_layers=2, dropout=0.2).to(device)

    if os.path.exists(checkpoint_pth):
        model_state = torch.load(checkpoint_pth)
        model.load_state_dict(model_state)
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataset_train = PickPlaceDataset(data_path="data/data_train.json")
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    dataset_validate = PickPlaceDataset(data_path="data/data_validate.json")
    dataloader_validate = DataLoader(dataset=dataset_validate, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    count = 0
    validation_loss = validate(model, dataloader_validate, device)
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=120)

        total_loss = 0
        for i, (multi_view_sequence, action_gt) in progress_bar:
            multi_view_sequence = multi_view_sequence.to(device)
            action_gt = action_gt.to(device)
            
            from IPython import embed
            embed()
            
            optimizer.zero_grad()
            action_pd = model(multi_view_sequence)
            loss = gripper_state_loss(action_gt, action_pd)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            progress_bar.set_postfix({
                "iter_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "valid_loss": f"{validation_loss:.4f}"
            })

            if count > 0 and (count % save_every == 0):
                torch.save(model.state_dict(), checkpoint_pth)

            if count > 0 and (count % validate_every == 0):
                validation_loss = validate(model, dataloader_validate, device)

            count += 1


if __name__ == '__main__':
    train()
