import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as sio
from tqdm import tqdm


def one_hot(y, dim):
    Y = np.zeros((y.shape[0], dim))
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
    return Y


class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 24, 5)
        self.conv2 = nn.Conv2d(24, 128, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 96)
        self.fc2 = nn.Linear(96, num_class)

    def forward(self, x):  # input:  9 X 9 after padding
        x = F.relu(self.conv1(x))  # (9-5+1) = 5: 5X5
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))  # (5-2+1)/2 = 2: 2X2
        x = x.view(-1, np.prod(x.shape[1:]))  # get the number of features in a batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
dataset_dir = '3D_dataset/with_base'
for index, file in enumerate(os.listdir(dataset_dir)):
    file = sio.loadmat(dataset_dir + '/' + file)
    if index == 0:
        data = file['data']
        random_indices = np.random.permutation(data.shape[0])
        data = data[random_indices, :, :, :]
        train_data = data[:int(0.6 * data.shape[0]), :, :, :]
        valid_data = data[int(0.6 * data.shape[0]):int(0.8 * data.shape[0]), :, :, :]
        test_data = data[:int(0.8 * data.shape[0]), :, :, :]
        label = np.int32(file['valence_labels'][0]) \
                + np.int32(file['arousal_labels'][0]) ** 2 \
                + np.int32(file['dominance_labels'][0]) ** 4
        label = one_hot(label, 8)
        label = label[random_indices, :]
        train_label = label[:int(0.6 * data.shape[0]), :]
        valid_label = label[int(0.6 * data.shape[0]):int(0.8 * data.shape[0]), :]
        test_label = label[:int(0.8 * data.shape[0]), :]
    else:
        data = file['data']
        random_indices = np.random.permutation(data.shape[0])
        data = data[random_indices, :, :, :]
        train_data = np.concatenate([train_data, data[:int(0.6 * data.shape[0]), :, :, :]])
        valid_data = np.concatenate([valid_data, data[int(0.6 * data.shape[0]):int(0.8 * data.shape[0]), :, :, :]])
        test_data = np.concatenate([test_data, data[int(0.8 * data.shape[0]):, :, :, :]])
        label = np.int32(file['valence_labels'][0]) \
                + np.int32(file['arousal_labels'][0]) ** 2 \
                + np.int32(file['dominance_labels'][0]) ** 4
        label = one_hot(label, 8)
        label = label[random_indices, :]
        train_label = np.concatenate([train_label, label[:int(0.6 * data.shape[0]), :]])
        valid_label = np.concatenate([valid_label, label[int(0.6 * data.shape[0]):int(0.8 * data.shape[0]), :]])
        test_label = np.concatenate([test_label, label[int(0.8 * data.shape[0]):, :]])

x_train = torch.tensor(train_data, requires_grad=False, dtype=torch.float32, device=device)
vl_train = torch.tensor(train_label, requires_grad=False, dtype=torch.float32, device=device)
x_valid = torch.tensor(valid_data, requires_grad=False, dtype=torch.float32, device=device)
vl_valid = torch.tensor(valid_label, requires_grad=False, dtype=torch.float32, device=device)
x_test = torch.tensor(test_data, requires_grad=False, dtype=torch.float32, device=device)
vl_test = torch.tensor(test_label, requires_grad=False, dtype=torch.float32, device=device)

net = CNN(num_class=8).to(device)
num_train = x_train.shape[0]

batch_size = 100
num_batches = num_train // batch_size
epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

for epoch in tqdm(range(epochs)):
    train_loss = 0
    valid_loss = 0
    for k in range(num_batches):
        mini_batch_indices = list(range(k * batch_size, (k + 1) * batch_size))
        inputs = x_train[mini_batch_indices, :, :, :]
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
    print('epoch: %d training loss: %.3f validation loss: %.3f' % (
        epoch + 1, train_loss / num_batches, valid_loss / num_batches))

with torch.no_grad():
    vl_pre = net(x_valid)
    vl_pre = torch.argmax(vl_pre, dim=1)
    vl_valid = torch.argmax(vl_valid, dim=1)
print('Validation acc after training:')
print((vl_valid == vl_pre).float().mean())

with torch.no_grad():
    vl_pre = net(x_test)
    vl_pre = torch.argmax(vl_pre, dim=1)
    vl_test = torch.argmax(vl_test, dim=1)
print('Test acc after training:')
print((vl_test == vl_pre).float().mean())
