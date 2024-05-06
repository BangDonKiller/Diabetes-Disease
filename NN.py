import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F

data = pd.read_csv('./dataset/diabetes.csv', sep = ',')
train_data = data.iloc[:600, :8].values
train_label = data['Outcome'].iloc[:600].values

test_data = data.iloc[600:, :8].values
test_label = data['Outcome'].iloc[600:].values

class NeuroNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(8, 20),
            nn.Sigmoid(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(20, 10),
            nn.Sigmoid(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x
        
model = NeuroNetwork()
loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

train_data = torch.tensor(train_data, dtype = torch.float32)
train_label = torch.tensor(train_label, dtype = torch.float32).view(-1, 1)        

for epoch in range(10000):
    optimizer.zero_grad()
    output = model(train_data)
    loss = loss_func(output, train_label)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        test_output = model(torch.tensor(test_data, dtype = torch.float32))
        correct = 0
        for i in range(test_output.size(0)):
            if abs(test_output[i] - 1) < abs(test_output[i] - 0) and test_label[i] == 1:
                correct += 1
            elif abs(test_output[i] - 1) > abs(test_output[i] - 0) and test_label[i] == 0:
                correct += 1
          
        print(f"Epoch: {epoch} |", "Loss value: ", loss.data, "| test accuracy: ", correct/test_output.size(0))