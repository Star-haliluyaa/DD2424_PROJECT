import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import scipy.io as sio
from tqdm import tqdm
from einops import repeat, rearrange

torch.cuda.empty_cache()


def one_hot(y, dim):
    Y = np.zeros((y.shape[0], dim))
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
    return Y


class ToPatch(nn.Module):
    def __init__(self, patch_size, token_dim):
        super(ToPatch, self).__init__()
        self.input_to_patch = nn.Conv2d(in_channels=4,
                                        out_channels=token_dim,
                                        kernel_size=patch_size,
                                        stride=patch_size)

    def forward(self, x):
        x = torch.reshape(x, shape=(-1, num_chan, height, width))
        x = self.input_to_patch(x)
        x = torch.reshape(x, shape=(-1, seq_length, x.shape[-3], x.shape[-2], x.shape[-1]))
        x = x.flatten(start_dim=-2, end_dim=-1)
        x = x.transpose(-2, -1)  # batch_size X seq_length X n_tokens X token_dim
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, image_size, patch_size, token_dim):
        super(TransformerEncoder, self).__init__()
        self.to_patch = ToPatch(patch_size, token_dim).to(device)
        n_tokens = (image_size // patch_size) ** 2
        self.space_cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.space_encoder = nn.TransformerEncoderLayer(d_model=token_dim, nhead=8, dim_feedforward=128)
        self.space_transformer_encoder = nn.TransformerEncoder(self.space_encoder, num_layers=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, n_tokens + 1, token_dim))
        self.tem_token = nn.Parameter(torch.randn(1, 1, token_dim))
        # self.tem_token = torch.zeros(10 * seq_length, token_dim, device=device)
        # self.tem_list = torch.arange(0, 10 * seq_length, dtype=torch.float, device=device).view(-1, 1)
        # self.division_term = torch.exp(torch.arange(0, token_dim, 2, device=device).float()
        # * (-math.log(10000.0)) / token_dim)
        # self.tem_token[:, 0::2] = torch.sin(self.tem_list * self.division_term)
        # self.tem_token[:, 1::2] = torch.cos(self.tem_list * self.division_term)
        self.tem_encoder = nn.TransformerEncoderLayer(d_model=token_dim, nhead=8, dim_feedforward=128)
        self.tem_transformer_encoder = nn.TransformerEncoder(self.tem_encoder, num_layers=1)
        self.dropout = nn.Dropout()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, num_classes))

    def forward(self, x):
        x = self.to_patch(x)
        b, t, n, _ = x.shape
        space_cls_tokens = repeat(self.space_cls_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((space_cls_tokens, x), dim=2)
        x += self.pos_embedding
        x = self.dropout(x)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer_encoder(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        tem_tokens = repeat(self.tem_token, '() n d -> b n d', b=b)
        # tem_tokens = repeat(torch.unsqueeze(self.tem_token[:seq_length, :], dim=0), '() n d -> b n d', b=b)
        x = torch.cat((tem_tokens, x), dim=1)
        x = self.tem_transformer_encoder(x)
        x = x.mean(dim=1)  # mean-pooling
        x = self.mlp_head(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
dataset_dir = '../data/3D_dataset/with_base' # You might need to change it to your own directory 

seq_length = 8
num_chan = 4
height = 9
width = 9
num_classes = 8
batch_size = 100

for index, file in enumerate(os.listdir(dataset_dir)):
    file = sio.loadmat(dataset_dir + '/' + file)
    if index == 0:# parse the data from first file
        data = file['data']
        data = np.reshape(data, newshape=(-1, seq_length, num_chan, height, width))
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
    else: # concatenate the data from other files
        data = file['data']
        random_indices = np.random.permutation(data.shape[0])
        data = np.reshape(data, newshape=(-1, seq_length, num_chan, height, width))
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

torch.cuda.empty_cache()
del data, label, random_indices, random_seq_indices, file


x_train = torch.tensor(train_data, requires_grad=False, dtype=torch.float32, device=device)
vl_train = torch.tensor(train_label, requires_grad=False, dtype=torch.float32, device=device)
x_valid = torch.tensor(valid_data, requires_grad=False, dtype=torch.float32, device=device)
vl_valid = torch.tensor(valid_label, requires_grad=False, dtype=torch.float32, device=device)
x_test = torch.tensor(test_data, requires_grad=False, dtype=torch.float32, device=device)
vl_test = torch.tensor(test_label, requires_grad=False, dtype=torch.float32, device=device)

del train_data, train_label, valid_data, valid_label, test_data, test_label

net = TransformerEncoder(image_size=9, patch_size=3, token_dim=16).to(device)
num_train = x_train.shape[0]
num_batches = num_train // batch_size
epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

for epoch in tqdm(range(epochs)):
    train_loss = 0
    valid_loss = 0
    for k in range(num_batches):
        mini_batch_indices = list(range(k * batch_size, (k + 1) * batch_size))
        inputs = x_train[mini_batch_indices, :, :, :, :]
        labels = vl_train[mini_batch_indices, :]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            valid_output = net(x_valid)
            valid_loss += criterion(valid_output, vl_valid).item()
    print('\n epoch: %d training loss: %.3f validation loss: %.3f ' % (epoch + 1, train_loss, valid_loss))

torch.cuda.empty_cache()
del x_train, vl_train, inputs, outputs, labels, valid_output, loss, mini_batch_indices


with torch.no_grad():
    vl_pre = net(x_valid)
    vl_pre = torch.argmax(vl_pre, dim=1)
    vl_valid = torch.argmax(vl_valid, dim=1)
print('Validation acc after training:')
print((vl_valid == vl_pre).float().mean())

torch.cuda.empty_cache()
del x_valid, vl_valid, vl_pre


change_gpu_to_cpu = "yes" # use yes if you got  CUDA out of memory.

if change_gpu_to_cpu == "yes":
    device =torch.device('cpu')
    net    =net.to(device)
    x_test = x_test.to(device)
    vl_test=vl_test.to(device)
    print(device)


with torch.no_grad():
    vl_pre = net(x_test)
    vl_pre = torch.argmax(vl_pre, dim=1)
    vl_test = torch.argmax(vl_test, dim=1)
print('Test acc after training:')
print((vl_test == vl_pre).float().mean())
