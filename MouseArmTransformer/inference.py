import argparse
import os
import pickle

import numpy as np
import pandas as pd
import pkg_resources
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange, repeat
from scipy.spatial.distance import euclidean
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import MouseArmTransformer.criterion
import MouseArmTransformer.data
import MouseArmTransformer.helper
import MouseArmTransformer.model
from MouseArmTransformer.helper import DLC_TO_MUJOCO_MAPPING

__all__ = ["run_inference"]


def _get_weight_path(filename):
    resource_path = f"weights/{filename}"
    resource_stream = pkg_resources.resource_stream(__name__, resource_path)
    return resource_stream


def _assert_correct_length(data_dict, min_length):
    common_length = None
    for key, value in data_dict.items():
        if common_length is None:
            common_length = value.shape[0]

        if (len(value.shape) != 2) or (value.shape[1] != 2):
            raise ValueError(f"Each value in the given camera dict needs to be a "
                             f"2D vector, but got {value.shape} for key {key}.")
        if value.shape[0] < min_length:
            raise ValueError(f"You need to provide at least {min_length} samples "
                             f"for each of the given values. I got {value.shape[0]} "
                             f"samples for key {key}.")
        if value.shape[0] != common_length:
            raise ValueError("All provided arrays need to have the same length. "
                             f"The common length determined is {common_length}, "
                             f"but the array with key {key} as length {value.shape[0]}.")


def _assert_correct_joints(given, reference):
    assert all([reference[i] == given[i] for i in range(len(given))])


def evaluate(model, dataloader, gt_dataloader, device, criterion, relative_displacements=None):
    model.eval()
    running_loss = 0.0
    predictions = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    with torch.no_grad():
        for i, (camera1, camera2, labels, _) in pbar:
            camera1, camera2, labels = camera1.to(device), camera2.to(device), labels.to(device)
            outputs = model(camera1)
            loss = criterion(outputs, labels)
            loss = torch.mean(loss)
            running_loss += loss.item()
            predictions.append(outputs.cpu().numpy())  # store the prediction
            pbar.set_description(f"Eval Iter {i+1}/{len(dataloader)} loss: {running_loss/(i+1):.5f}")
    # Also evaluate on GT:
    if gt_dataloader is None:
        return running_loss / len(dataloader), np.concatenate(predictions, axis=0)

    if not (torch.nansum(gt_dataloader.all_in) == 0.0):
        model_out = model(gt_dataloader.all_in)
        mse_loss_gt_norel = criterion(model_out, gt_dataloader.all_gt)
        mse_loss_gt = criterion(model_out, gt_dataloader.all_gt) / relative_displacements[..., np.newaxis]
    else:
        mse_loss_gt = torch.tensor(0.0)
        mse_loss_gt_norel = torch.tensor(0.0)

    return running_loss / len(dataloader), mse_loss_gt, mse_loss_gt_norel, np.concatenate(predictions,
                                                                                          axis=0), model_out


def run_inference(camera1_dict,
                  model_weights='weights_epoch_5_loss0.11_seq2_cutoff0.999_lossweights[1, 25, 1, 1e-05].pt',
                  seq_length=7):
    # Make sure joints are in the same order as in the training data
    original_joints = MouseArmTransformer.helper.mausspaun_keys
    
    # Dict is sorted based on insert order
    camera1_dict = {key: camera1_dict[key] for key in original_joints}
    joints = list(camera1_dict.keys())
    _assert_correct_joints(joints, original_joints)
    _assert_correct_length(camera1_dict, min_length=seq_length + 1)

    camera1 = np.transpose(np.array([cp for key, cp in camera1_dict.items()]), axes=(1, 0, 2))

    # Load model and weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MouseArmTransformer.model.SimpleTransformer(num_joints=len(joints))
    model.load_state_dict(torch.load(_get_weight_path(model_weights), map_location=device))
    model.to(device)

    # Create dataloader with dummy labels + camera2
    labels = np.zeros((camera1.shape[0], camera1.shape[1], 3))
    camera2 = np.zeros_like(camera1)
    print(camera1.shape)
    inference_dataset = MouseArmTransformer.data.PositionDataset(camera1, camera2, labels, seq_length=seq_length)
    inference_dataloader = DataLoader(inference_dataset, batch_size=128, shuffle=False)

    # Run inference
    inference_loss, inference_preds = evaluate(model,
                                               inference_dataloader,
                                               gt_dataloader=None,
                                               device=device,
                                               criterion=MouseArmTransformer.criterion.masked_loss)
    print(f"Inference loss: {inference_loss:.5f}")

    # Put back into dictionary
    inference_preds = inference_preds[:, -1, :, :]  # Only take last timestep
    # Set the first timesteps of sequence as they are undefined (no previous timestep)
    inference_preds[:seq_length, :, :] = inference_preds[seq_length, :, :]
    inference_preds = {joint: inference_preds[:, i, :] for i, joint in enumerate(joints)}

    return inference_preds


def evaluate_ground_truth(inference_preds, mouse_name, day, attempt, part=0, verbose=1):
    """
    Evaluate the ground truth of the given predictions.
    """
    assert part == 0, "3D ground truth only available for part 0"

    # Convert predictions from dict to numpy
    inference_preds_np = np.transpose(np.array([cp for key, cp in inference_preds.items()]), axes=(1, 0, 2))

    # Load ground truth data
    labeled_3d = MouseArmTransformer.helper.load_and_process_ground_truth(mouse_name, day, attempt, part)
    if labeled_3d is None:
        return -1, pd.DataFrame([])

    temp_dfs = []
    for frame in labeled_3d['frame'].unique():
        for marker in DLC_TO_MUJOCO_MAPPING.keys():
            error = MouseArmTransformer.helper.calculate_gt_error(frame, marker, inference_preds_np, labeled_3d,
                                                                  DLC_TO_MUJOCO_MAPPING)
            temp_df = pd.DataFrame({'frame': [frame], 'marker': [DLC_TO_MUJOCO_MAPPING[marker]], 'error': [error]})
            temp_dfs.append(temp_df)

    errors_df = pd.concat(temp_dfs, ignore_index=True)

    # Compute and print statistics
    MouseArmTransformer.helper.compute_gt_statistics(errors_df, verbose)

    return errors_df['error'].mean(), errors_df
