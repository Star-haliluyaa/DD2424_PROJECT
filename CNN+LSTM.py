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


class ConvLSTM(nn.Module):
    def __init__(self, embedding_size, batch_size, num_classes, hidden_size=8, num_directions=1, num_layers=1):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(4, 24, 5)
        self.conv2 = nn.Conv2d(24, 128, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 96)
        self.fc2 = nn.Linear(96, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.hidden = None
        self.init_hidden(batch_size, hidden_size, num_directions, num_layers)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def init_hidden(self, batch_size, hidden_size=8, num_directions=1, num_layers=1):
        self.hidden = (
            torch.autograd.Variable(torch.zeros(num_directions * num_layers, batch_size, hidden_size, device=device)),
            torch.autograd.Variable(torch.zeros(num_directions * num_layers, batch_size, hidden_size, device=device)))

    def forward(self, x):  # input:  9 X 9 after padding
        x_list = []
        for t in range(seq_length):
            x_t = F.relu(self.conv1(x[:, t, :, :]))  # (9-5+1) = 5: 5X5
            x_t = F.max_pool2d(F.relu(self.conv2(x_t)), (2, 2))  # (5-2+1)/2 = 2: 2X2
            x_t = x_t.view(-1, np.prod(x_t.shape[1:]))  # get the number of features in a batch
            x_t = F.relu(self.fc1(x_t))
            x_t = torch.unsqueeze(F.relu(self.fc2(x_t)), dim=1)
            x_list.append(x_t)
        x = torch.cat(x_list, dim=1)
        x, (hn, cn) = self.lstm(x, self.hidden)
        y = self.fc3(x[:, -1, :])
        return y, x, hn, cn


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

net = ConvLSTM(embedding_size=16, batch_size=batch_size, num_classes=num_classes).to(device)

num_train = x_train.shape[0]
num_batches = num_train // batch_size
epochs = 350
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
        net.init_hidden(batch_size=batch_size)
        outputs, _, _, _ = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            net.init_hidden(batch_size=x_valid.shape[0])
            valid_output, _, _, _ = net(x_valid)
            valid_loss += criterion(valid_output, vl_valid).item()
    print('epoch: %d training loss: %.3f validation loss: %.3f' % (epoch + 1, train_loss, valid_loss))

with torch.no_grad():
    net.init_hidden(batch_size=x_valid.shape[0])
    vl_pre, _, _, _ = net(x_valid)
    vl_pre = torch.argmax(vl_pre, dim=1)
    vl_valid = torch.argmax(vl_valid, dim=1)
print('Validation acc after training:')
print((vl_valid == vl_pre).float().mean())

with torch.no_grad():
    net.init_hidden(batch_size=x_test.shape[0])
    vl_pre, _, _, _ = net(x_test)
    vl_pre = torch.argmax(vl_pre, dim=1)
    vl_test = torch.argmax(vl_test, dim=1)
print('Test acc after training:')
print((vl_test == vl_pre).float().mean())
