import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as sio
from tqdm import tqdm
from einops import rearrange


def one_hot(y, dim):
    Y = np.zeros((y.shape[0], dim))
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
    return Y


class ToPatch(nn.Module):
    def __init__(self, patch_size):
        super(ToPatch, self).__init__()
        self.input_to_patch = nn.Conv2d(in_channels=3,
                                        out_channels=16,
                                        kernel_size=patch_size,
                                        stride=patch_size)

    def forward(self, x):
        x = self.input_to_patch(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
dataset_dir = '3D_dataset/with_base'

seq_length = 8
num_chan = 4
height = 9
width = 9
num_classes = 8
batch_size = 100

for index, file in enumerate(os.listdir(dataset_dir)):
    file = sio.loadmat(dataset_dir + '/' + file)
    if index == 0:
        data = file['data']
        data = np.reshape(data, newshape=(-1, seq_length, num_chan, height, width))
        # data = np.swapaxes(data, 1, 2)
        random_seq_indices = np.random.permutation(data.shape[0] // seq_length)
        data = data[random_seq_indices, :, :, :, :]
        train_data = data[:int(0.6 * data.shape[0]), :, :, :, :]
        valid_data = data[int(0.6 * data.shape[0]):int(0.8 * data.shape[0]), :, :, :, :]
        test_data = data[:int(0.8 * data.shape[0]), :, :, :, :]
        label = np.int32(file['valence_labels'][0]) \
                + np.int32(file['arousal_labels'][0]) ** 2 \
                + np.int32(file['dominance_labels'][0]) ** 4
        label = one_hot(label, 8)
        seq_label = np.empty([0, num_classes])
        for j in range(int(label.shape[0] // seq_length)):
            seq_label = np.vstack((seq_label, label[j * seq_length]))
        seq_label = seq_label[random_seq_indices, :]
        train_label = seq_label[:int(0.6 * data.shape[0]), :]
        valid_label = seq_label[int(0.6 * data.shape[0]):int(0.8 * data.shape[0]), :]
        test_label = seq_label[:int(0.8 * data.shape[0]), :]
    else:
        data = file['data']
        random_indices = np.random.permutation(data.shape[0])
        data = np.reshape(data, newshape=(-1, seq_length, num_chan, height, width))
        # data = np.swapaxes(data, 1, 2)
        random_seq_indices = np.random.permutation(data.shape[0] // seq_length)
        data = data[random_seq_indices, :, :, :, :]
        train_data = np.concatenate([train_data, data[:int(0.6 * data.shape[0]), :, :, :, :]])
        valid_data = np.concatenate([valid_data, data[int(0.6 * data.shape[0]):int(0.8 * data.shape[0]), :, :, :, :]])
        test_data = np.concatenate([test_data, data[int(0.8 * data.shape[0]):, :, :, :, :]])
        label = np.int32(file['valence_labels'][0]) \
                + np.int32(file['arousal_labels'][0]) ** 2 \
                + np.int32(file['dominance_labels'][0]) ** 4
        label = one_hot(label, 8)
        seq_label = np.empty([0, num_classes])
        for j in range(int(label.shape[0] // seq_length)):
            seq_label = np.vstack((seq_label, label[j * seq_length]))
        seq_label = seq_label[random_seq_indices, :]
        train_label = np.concatenate([train_label, seq_label[:int(0.6 * data.shape[0]), :]])
        valid_label = np.concatenate([valid_label, seq_label[int(0.6 * data.shape[0]):int(0.8 * data.shape[0]), :]])
        test_label = np.concatenate([test_label, seq_label[int(0.8 * data.shape[0]):, :]])

x_train = torch.tensor(train_data, requires_grad=False, dtype=torch.float32, device=device)
vl_train = torch.tensor(train_label, requires_grad=False, dtype=torch.float32, device=device)
x_valid = torch.tensor(valid_data, requires_grad=False, dtype=torch.float32, device=device)
vl_valid = torch.tensor(valid_label, requires_grad=False, dtype=torch.float32, device=device)
x_test = torch.tensor(test_data, requires_grad=False, dtype=torch.float32, device=device)
vl_test = torch.tensor(test_label, requires_grad=False, dtype=torch.float32, device=device)
