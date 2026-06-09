import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data_loader.dsec_full import DSECfull
from utils import writer_add_features_normalized as writer_add_features


@torch.no_grad()
def validate(model: nn.Module, val_loader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_list = []
    bar = tqdm(val_loader, total=len(val_loader), ncols=60, leave=False, desc="Validation")
    for voxel, voxel_gt, _ in bar:
        voxel    = voxel.to(device).float()
        voxel_gt = voxel_gt.to(device).float()
        output = model(voxel)
        loss = F.l1_loss(output, voxel_gt)
        loss_list.append(loss.item())
    return {"Val/Loss": float(np.mean(loss_list))}


@torch.no_grad()
def visualize_output(model: nn.Module, device: torch.device) -> list[tuple[int, np.ndarray]]:
    visualizations = []
    loader = DSECfull("sample")
    for index, (voxel, voxel_gt, img1) in enumerate(loader):
        voxel = voxel.unsqueeze(0).to(device).float()
        output = model(voxel)

        vmin = voxel_gt.min().item()
        vmax = voxel_gt.max().item()
        gt_vis  = writer_add_features(voxel_gt, vmin, vmax)
        in_vis  = writer_add_features(voxel[0].cpu(), vmin, vmax)
        out_vis = writer_add_features(output[0].cpu(), vmin, vmax)

        img = (img1.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        frame = np.vstack([np.hstack([img, gt_vis]), np.hstack([in_vis, out_vis])])
        visualizations.append((index, frame))
    return visualizations
