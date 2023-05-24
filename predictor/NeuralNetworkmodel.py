import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
import joblib


class Architecture(nn.Module):
    def __init__(self,input_shape,L):
        super(Architecture, self).__init__()
        P = 2*L
        M = 2*P
        self.fc1 = nn.Linear(input_shape, M)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(M,P)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(P,L)
        self.fc4 = nn.Linear(L,1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc4.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.prelu(x, torch.tensor([0.01]))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.prelu(x, torch.tensor([0.01]))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.prelu(x, torch.tensor([0.01]))
        x = self.fc4(x)
        return x


class NeuralNetwork():
    def __init__(self,input_shape ,L=128, epoch=10, batch=32,random_state=314):
        torch.manual_seed(314)
        self.model = Architecture(input_shape,L)
        self.epoch = epoch
        self.batch_size = batch
        self.criterion = F.mse_loss
    
    def fit(self, x_train:pd.DataFrame, y_train:pd.DataFrame):    
        X = torch.tensor(x_train.values, dtype=torch.float32)
        Y = torch.tensor(y_train.values, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X, Y)
        train_loader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        losslist = np.array([float('inf')])
        for ep in range(self.epoch):
    
            for batch in train_loader:
                x, t = batch
                optimizer.zero_grad()
                y = self.model(x)[:,0]
                loss = self.criterion(y, t)
                loss.backward()
                optimizer.step()
            if losslist[-1] < loss.item():
                print("Early Stopping")
                break
            losslist = np.append(losslist,loss.item())
            print(f"epoch:{ep+1}, loss:{loss.item()}")

                
            
    
    def predict(self, x_test:pd.DataFrame):
        self.model.eval()
        index = x_test.index
        x_test = torch.tensor(x_test.values, dtype=torch.float32)
        y_predict = self.model(x_test)
        y_predict = y_predict.detach().numpy()
        y_predict = pd.DataFrame(y_predict,index=index,columns=['prediction'])
        y_predict["prediction"] = y_predict["prediction"].map(lambda x: int(max(min(x, 3), 0)))
        return y_predict
    
    def dump(self,filename="NeuralNetwork"):
        joblib.dump(self, f"results/model/{filename}.model")