import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.data import Dataset
from tqdm import tqdm


# Dataset
class PositionDataset(Dataset):

    def __init__(self, camera1, camera2, labels, seq_length):
        self.camera1 = torch.tensor(camera1, dtype=torch.float32)
        self.camera2 = torch.tensor(camera2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.num_joints = self.camera1.shape[1]
        self.seq_length = seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        camera1 = (self.camera1[max(0, idx - self.seq_length + 1):idx + 1]).to(self.device)
        camera2 = (self.camera2[max(0, idx - self.seq_length + 1):idx + 1]).to(self.device)
        label = (self.labels[max(0, idx - self.seq_length + 1):idx + 1]).to(self.device)

        # Pad sequences shorter than seq_length
        if len(camera1) < self.seq_length:
            pad_size = self.seq_length - len(camera1)
            camera1 = torch.cat([torch.zeros(pad_size, *camera1.shape[1:], device=self.device), camera1])
            camera2 = torch.cat([torch.zeros(pad_size, *camera2.shape[1:], device=self.device), camera2])
            label = torch.cat([torch.zeros(pad_size, *label.shape[1:], device=self.device), label])

        # Cut down to the number of joints
        camera1 = camera1[:, 0:self.num_joints, :]
        camera2 = camera2[:, 0:self.num_joints, :]
        label = label[:, 0:self.num_joints, :]

        return camera1, camera2, label, []  # Empty list for gt_label


class PositionDatasetGT(Dataset):

    def __init__(self, camera1, camera2, labels, gt_labels, seq_length):
        self.camera1 = torch.tensor(camera1, dtype=torch.float32)
        self.camera2 = torch.tensor(camera2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Triangulated
        self.gt_labels = torch.tensor(gt_labels, dtype=torch.float32)  # Ground truth

        self.num_joints = self.camera1.shape[1]
        self.seq_length = seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        camera1 = (self.camera1[max(0, idx - self.seq_length + 1):idx + 1]).to(self.device)
        camera2 = (self.camera2[max(0, idx - self.seq_length + 1):idx + 1]).to(self.device)
        label = (self.labels[max(0, idx - self.seq_length + 1):idx + 1]).to(self.device)
        gt_label = (self.gt_labels[max(0, idx - self.seq_length + 1):idx + 1]).to(self.device)

        # Pad sequences shorter than seq_length
        if len(camera1) < self.seq_length:
            pad_size = self.seq_length - len(camera1)
            camera1 = torch.cat([torch.zeros(pad_size, *camera1.shape[1:], device=self.device), camera1])
            camera2 = torch.cat([torch.zeros(pad_size, *camera2.shape[1:], device=self.device), camera2])
            label = torch.cat([torch.zeros(pad_size, *label.shape[1:], device=self.device), label])
            gt_label = torch.cat([torch.zeros(pad_size, *gt_label.shape[1:], device=self.device), gt_label])

        # Cut down to the number of joints
        camera1 = camera1[:, 0:self.num_joints, :]
        camera2 = camera2[:, 0:self.num_joints, :]
        label = label[:, 0:self.num_joints, :]
        gt_label = gt_label[:, 0:self.num_joints, :]

        return camera1, camera2, label, gt_label
