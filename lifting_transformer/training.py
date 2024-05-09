import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.data import Dataset
from tqdm import tqdm

import lifting_transformer.helper
from lifting_transformer.criterion import mspe_loss


def train(model, dataloader, gt_dataloader, device, criterion, optimizer, loss_weights, relative_displacements,
          all_aggregated_tensor):
    model.train()
    running_loss, running_others = 0.0, np.array([0.0, 0.0, 0.0, 0.0])
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, ncols=150)
    continuity_criterion = nn.MSELoss(reduction='none')
    #previous_mse_gt_loss = None
    for i, (camera1, camera2, labels, gt_labels) in pbar:
        camera1, camera2, labels, gt_labels = camera1.to(device), camera2.to(device), labels.to(device), gt_labels.to(
            device)

        optimizer.zero_grad()
        outputs = model(camera1)  # [128, 3, 25, 3], [128, 3, 36]

        # MSE loss
        mse_loss = criterion(outputs, labels) / relative_displacements[..., np.newaxis]
        mse_loss = torch.mean(mse_loss)

        # MSE loss on GT
        model_out = model(gt_dataloader.all_in)
        mse_loss_gt = criterion(model_out, gt_dataloader.all_gt) / relative_displacements[..., np.newaxis]
        mse_loss_gt = torch.sum(mse_loss_gt)

        # With continouity loss:
        continuity_loss = torch.mean(
            continuity_criterion(outputs[:, 1:], outputs[:, :-1]) / relative_displacements[..., np.newaxis])

        # With connectivity loss:
        output_distances = lifting_transformer.helper.calculate_euclidean_distance_4D_torch(outputs)
        connectivity_loss = mspe_loss(output_distances, all_aggregated_tensor)

        # Combine
        loss = loss_weights['mse'] * mse_loss + loss_weights['continuity'] * continuity_loss + loss_weights[
            'connectivity'] * connectivity_loss + loss_weights['ground_truth'] * mse_loss_gt

        #loss = mse_loss_gt
        #print(loss)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_others += np.array(
            [mse_loss.item(), mse_loss_gt.item(),
             continuity_loss.item(),
             connectivity_loss.item()])
        pbar.set_description(
            f"Train Iter {i+1}/{len(dataloader)} loss: {running_loss/(i+1):.5f} Mse Loss: {running_others[0]/(i+1):.5f} Mse GT: {running_others[1]/(i+1):.5f} Continuity Loss: {running_others[2]/(i+1):.5f} Connectivity Loss: {running_others[3]/(i+1):.5f}"
        )
    return running_loss / len(dataloader)
