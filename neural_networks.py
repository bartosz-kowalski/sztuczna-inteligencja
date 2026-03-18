import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn 
from torch import optim 
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("./HDP.csv")
target = "Heart_Disease"
df[target] = df[target].map({
    "Presence":1,
    "Absence":0
})

df = df.fillna(df.median())
df = df.fillna(0)

X = df.drop(columns = [target])
Y = df[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_temp, y_train, y_temp = train_test_split(
    X,Y,
    test_size = 0.3,
    stratify=Y,
    random_state=42
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_val = x_val.astype(np.float32)

y_val = y_val.to_numpy(dtype = np.float32)
y_train = y_train.to_numpy(dtype = np.float32)
y_test = y_test.to_numpy(dtype = np.float32)

x_train_tensor = torch.from_numpy(x_train)
x_test_tensor = torch.from_numpy(x_test)
x_val_tensor = torch.from_numpy(x_val)

y_train_tensor = torch.from_numpy(y_train).unsqueeze(1)
y_test_tensor = torch.from_numpy(y_test).unsqueeze(1)
y_val_tensor = torch.from_numpy(y_val).unsqueeze(1)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout (0.3),
            nn.Linear(16,1)
        )
    
    def forward(self, x):
        return self.model(x)

input_dim = x_train.shape[1]

model = SimpleNN(input_dim)

num_pos = y_train_tensor.sum().item()
num_neg = y_train_tensor.shape[0] - num_pos

pos_weigth = torch.tensor(
    [num_neg/num_pos],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weigth)
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

n_epochs = 400

epochs = []
val_loss = []
train_loss = []
train_acc = []
val_acc =[]

for epoch in range(n_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*x_batch.size(0)
        probs = torch.sigmoid(y_pred)
        predicted = (probs>=0.5).float()
        correct += (predicted==y_batch).sum().item()
        total += y_batch.size(0)
    epoch_train_loss = running_loss/total
    epoch_train_acc = correct/total
    train_loss.append(epoch_train_loss)
    epochs.append(epoch)
    train_acc.append(epoch_train_acc)

    model.eval()
    running_loss_val = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            running_loss_val += loss.item()*x_batch.size(0)
            probs = torch.sigmoid(y_pred)
            predicted = (probs >=0.5).float()
            correct_val += (predicted==y_batch).sum().item()
            total_val += y_batch.size(0)
    
    epoch_val_loss = running_loss_val /total_val
    epoch_val_acc = correct_val /total_val

    val_loss.append(epoch_val_loss)
    val_acc.append(epoch_val_acc)

plt.figure(1)
plt.plot(train_loss)
plt.plot(val_loss)
plt.show()
plt.figure(2)
plt.plot(train_acc)
plt.plot(val_acc)
plt.show()
