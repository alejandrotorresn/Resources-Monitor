import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd


train_data = np.random.randn(5000, 3, 32, 32).astype(np.float32)
label_data = np.random.randint(0, 10, 5000).astype(np.int32)

trainset = torch.utils.data.TensorDataset(torch.tensor(train_data), 
                                          torch.tensor(label_data))
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=1)

times_layers = pd.DataFrame(dtype=object)
times_epochs = pd.DataFrame(dtype=object)

class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64) 
        self.conv2 = nn.Conv2d(64, 128, 3) 
        self.bn2 = nn.BatchNorm2d(128) 
        self.conv3 = nn.Conv2d(128, 256, 3) 
        self.bn3 = nn.BatchNorm2d(256) 
        self.fc = nn.Linear(26*26*256, 10)
        
    def forward(self, x):
        global times_layers
        time_layer = pd.DataFrame(dtype=object)
        time_l = []
        time_l.append(time.time())
        x = self.conv1(x)
        time_l.append(time.time())
        x = self.bn1(x)
        time_l.append(time.time())
        x = self.activation(x)
        time_l.append(time.time())
        x = self.conv2(x)
        time_l.append(time.time())
        x = self.bn2(x)
        time_l.append(time.time())
        x = self.activation(x)
        time_l.append(time.time())
        x = self.conv3(x)
        time_l.append(time.time())
        x = self.bn3(x)
        time_l.append(time.time())
        x = self.activation(x)
        time_l.append(time.time())
        x = x.reshape(-1, x.size(1)*x.size(2)*x.size(3))
        
        time_layer = pd.DataFrame(time_l)
        times_layers = times_layers.append(time_layer.T, ignore_index=True)

        return self.fc(x)

net = Net().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    start = time.time()
    times_epochs = times_epochs.append([start])
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        outputs = net(inputs.cuda())
        loss = criterion(outputs, labels.long().cuda())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print("time: ", time.time() - start)
    print("epoch: {:d} loss: {:.3f}".format(epoch + 1, running_loss / len(trainloader)))


times_layers.to_csv('reports/times_layers_report_pytorch.csv', index=False)
times_epochs.to_csv('reports/times_epochs_report_pytorch.csv', index=False)
