import os, os.path
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import dataloader
import numpy as np
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F

from models import Transformer

log_path = 'logs/HDFS/structured/HDFS.log_structured.csv'
label_path = 'logs/HDFS/anomaly_label.csv'
template_path = 'logs/HDFS/structured/HDFS.log_templates.csv'

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

batch_size = 256
lr = 0.001
num_epochs = 300
max_length = x_train.shape[1]
val_interval = 10

x_train_tensor = torch.Tensor(x_train)
y_train_tensor = torch.Tensor(y_train).to(torch.int64)
y_train_tensor = F.one_hot(y_train_tensor, num_classes=2)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor.to(torch.float))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

x_val_tensor = torch.Tensor(x_val[:200])
y_val_tensor = torch.Tensor(y_val[:200]).to(torch.int64)
y_val_tensor = F.one_hot(y_val_tensor, num_classes=2)

val_dataset = TensorDataset(x_val_tensor, y_val_tensor.to(torch.float))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = nn.Sequential(
    Transformer(
        in_dim=1,
        embed_dim=64,
        depth=6,
        heads=8,
        dim_head=64,
        dim_ratio=2,
        dropout=0.1
    ),
    nn.Linear(max_length * 64, 100),
    nn.ReLU(),
    nn.Linear(100, 2),
    nn.Softmax()
)

model = nn.DataParallel(model)  # multi-GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
print('device: ', device)

# Loss and optimizer
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# Train the model
loss_min = 99999
model_name = 'best_model.pth'
model_path = "saved_models"

if not os.path.exists(model_path):
    os.mkdir(model_path)

save_path = os.path.join(model_path, model_name)
best_model = model
train_loss_list = []
val_loss_list = []

from tqdm import tqdm

print("Begin training ......")
for epoch in range(1, num_epochs + 1):  # Loop over the dataset multiple times
    train_loss = 0
    val_loss = 0

    # Training
    for step, (seq, label) in enumerate(tqdm(train_dataloader)):
        seq = seq.clone().detach().view(-1, max_length, 1).to(device)
        output = model(seq)
        loss = criterion(output, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    ave_trainloss = train_loss / len(train_dataloader)
    train_loss_list.append(ave_trainloss)

    if ave_valoss < loss_min:
        loss_min = ave_valoss
        torch.save(model.state_dict(), save_path)
        best_model = model
        print("Model saved")

    print('Epoch [{}/{}], train_loss: {:.14f}'.format(epoch, num_epochs, ave_trainloss))

    if epoch % val_interval == 0:
        # Vaildating
        with torch.no_grad():
            for step, (seq, label) in enumerate(val_dataloader):
                seq = seq.clone().detach().view(-1, max_length, 1).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))
                val_loss += loss.item()

        ave_valoss = val_loss / len(val_dataloader)
        val_loss_list.append(ave_valoss)
        print('Epoch [{}/{}] val loss: {:.14f}'.format(epoch, num_epochs, ave_valoss))

print(f"Finished training, model saved in: {save_path} ")

xx = range(len(train_loss_list))
plt.plot(xx, train_loss_list, label="Train")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("train_loss.png")

xx = range(len(val_loss_list))
plt.plot(xx, val_loss_list, label="Val")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("val_loss.png")