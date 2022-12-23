import os
import dataloader
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from loss_func import FocalLoss
from models.model_collects import *

from models import Transformer
import matplotlib.pyplot as plt
import torchinfo

log_path = 'logs/HDFS_2k.log_structured.csv'
label_path = 'logs/anomaly_label.csv'
template_path = 'logs/HDFS_2k.log_templates.csv'

x_train, y_train, x_test, y_test = dataloader.load_HDFS(
    log_file=log_path,
    label_file=label_path,
    template_file=template_path,
    train_ratio=0.7,
    save_csv=False)

num_val = x_train.shape[0] // 10
num_train = x_train.shape[0] - num_val

x_val = x_train[:num_val]
y_val = y_train[:num_val]
x_train = x_train[num_val:]
y_train = y_train[num_val:]

num_test = x_test.shape[0]
num_total = num_train + num_val + num_test

num_train_pos = sum(y_train)
num_val_pos = sum(y_val)
num_test_pos = sum(y_val)
num_pos = num_train_pos + num_val_pos + num_test_pos

print('Total: {} instances, {} anomaly, {} normal' \
      .format(num_total, num_pos, num_total - num_pos))
print('Train: {} instances, {} anomaly, {} normal' \
      .format(num_train, num_train_pos, num_train - num_train_pos))
print('Validation: {} instances, {} anomaly, {} normal' \
      .format(num_val, num_val_pos, num_val - num_val_pos))
print('Test: {} instances, {} anomaly, {} normal\n' \
      .format(num_test, num_test_pos, num_test - num_test_pos))

batch_size = 512
lr = 0.001
num_epochs = 300
max_length = x_train.shape[1]
input_shape = x_train.shape[1:]
val_interval = 1

x_train_tensor = torch.Tensor(x_train)
y_train_tensor = torch.Tensor(y_train).to(torch.int64)
y_train_tensor = F.one_hot(y_train_tensor, num_classes=2)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor.to(torch.float))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

x_val_tensor = torch.Tensor(x_val)
y_val_tensor = torch.Tensor(y_val).to(torch.int64)
y_val_tensor = F.one_hot(y_val_tensor, num_classes=2)

val_dataset = TensorDataset(x_val_tensor, y_val_tensor.to(torch.float))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model_path = "saved_models"
if not os.path.exists(model_path):
    os.mkdir(model_path)

train_loss_dict = dict()
val_loss_dict = dict()

# define model
device = torch.device("mps")
model = TransLog(max_length).to(device)
model_name = "TransLog"
best_model = model

# Loss and optimizer
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
loss_min = 1e10

train_loss_list = []
val_loss_list = []

for epoch in range(1, num_epochs + 1):
    train_loss = 0
    val_loss = 0
    # Training
    for step, (seq, label) in enumerate(train_dataloader):
        seq = seq.clone().detach().view(-1, max_length, 1).to(device)
        output = model(seq)
        loss = criterion(output, label.to(device))
        optimizer.zero_grad()
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

    ave_trainloss = train_loss / len(train_dataloader)
    train_loss_list.append(ave_trainloss)

    # Vaildating
    saved = False
    with torch.no_grad():
        for step, (seq, label) in enumerate(val_dataloader):
            seq = seq.clone().detach().view(-1, max_length, 1).to(device)
            output = model(seq)
            #loss = criterion(output, label.to(device))
            loss = nn.BCEWithLogitsLoss()(output, label.to(device))
            val_loss += loss.item()

    ave_valoss = val_loss / len(val_dataloader)
    val_loss_list.append(ave_valoss)

    if ave_valoss < loss_min:
        loss_min = ave_valoss
        save_path = os.path.join(model_path, 'best_{}.pth'.format(model_name))
        torch.save(model.state_dict(), save_path)
        best_model = model
        saved = True

    print('epoch [{}/{}], train_loss= {:.10f} val_loss= {:.10f} save= {}'.
          format(epoch, num_epochs, ave_trainloss, ave_valoss, saved))

train_loss_dict[model_name] = train_loss_list
val_loss_dict[model_name] = val_loss_list

torchinfo.summary(model,input_size=(batch_size,max_length,1))

xx = range(1, num_epochs+1)
for name, train_loss_list in train_loss_dict.items():
    # if name=="Mini_TransLog":
    #     continue
    plt.plot(xx, train_loss_list, label=name)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.show()
plt.savefig("train_loss.png")
plt.clf()

for name, val_loss_list in val_loss_dict.items():
    # if name=="Mini_TransLog":
    #     continue
    plt.plot(xx, val_loss_list, label=name)
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()
plt.savefig("valid_loss.png")
plt.clf()