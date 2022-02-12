import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
from sklearn.metrics import accuracy_score, f1_score

from unet import UNet

import dill
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Final
class Model(nn.Module) :
    def __init__(self, device=device) :
        super(Model, self).__init__()
        
        self.device = device
        
        self.unet = UNet()
        
        self.fwd_lstm1 = nn.LSTM(128, 128)
        self.fwd_lstm2 = nn.LSTM(128, 128)
        self.fwd_rev_lstm1 = nn.LSTM(128, 128)
        self.fwd_rev_lstm2 = nn.LSTM(128, 128)
        
        self.back_lstm1 = nn.LSTM(128, 128)
        self.back_lstm2 = nn.LSTM(128, 128)
        self.back_rev_lstm1 = nn.LSTM(128, 128)
        self.back_rev_lstm2 = nn.LSTM(128, 128)
        
        self.conv1 = nn.Conv1d(300, 128, 3)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.conv3 = nn.Conv1d(128, 300, 3)
        
        self.bn = nn.BatchNorm1d(300)
        
        self.lin = nn.Linear(14, 5)
        
        self.loss_crit = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, x) :
        if len(x.shape) == 3:
            x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
            
        x = self.unet(x)
        x = x.reshape((x.shape[0], x.shape[2], x.shape[3]))
      
        x = x.transpose(1,2)
        
        za = self.fwd_lstm1(x)
        xa, _ = self.back_lstm1(za[0], za[1])
        
        
        zb = self.fwd_lstm2(x)
        xb, _ = self.back_rev_lstm1(zb[0].flip(2), zb[1])
        
        zc = self.fwd_rev_lstm1(x.flip(2))
        xc, _ = self.back_lstm2(zc[0], zc[1])
        
        zd = self.fwd_rev_lstm2(x.flip(2))
        xd, _ = self.back_rev_lstm2(zd[0].flip(2), zd[1])
        
        x = (xa+xb.flip(2)+xc.flip(2)+xd)/4
        
        x = torch.relu(F.max_pool1d((self.conv1(x)), 2))
        x = torch.relu(F.max_pool1d((self.conv2(x)), 2))
        x = torch.relu(F.max_pool1d(self.bn(self.conv3(x)), 2))
        x = torch.relu(self.lin(x))
        return x.transpose(1,2)
    
    def run_on_batch(self, x, y=[]) :
        x.to(self.device)
        if len(y) == 0 :
            return self(x).argmax(1)
        
        else :
            y.to(self.device)
            pred = self(x)
            loss = self.loss_crit(pred, y)
            output = pred.argmax(1)
            return {'output' : output, 'loss' : loss}
        
    def step(self, loss) :
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


train_X = dill.load(open('use_dataset/logmels_patch300_train', 'rb'))
train_Y = dill.load(open('use_dataset/frameset_patch300_train', 'rb')).long()

test_X = dill.load(open('use_dataset/logmels_patch300_test', 'rb'))
test_Y = dill.load(open('use_dataset/frameset_patch300_test', 'rb')).long()

class BeatBoxDataset(Dataset):
    def __init__(self, input_, output_):
        super().__init__()
        self.input_ = input_
        self.output_ = output_

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.input_[idx], self.output_[idx]

trainloader = DataLoader(BeatBoxDataset(train_X, train_Y), batch_size=64)
testloader = DataLoader(BeatBoxDataset(test_X, test_Y), batch_size=64)

EPOCHS = 200
model = Model(device)

train_acc = [0]
train_f1 = [0]

test_acc = [0]
test_f1 = [0]


for epoch in range(EPOCHS) :
    for phase in ['train', 'validation'] :
        run_acc = []
        run_f1 = []
        if phase == 'train' :
            model.train()
        else :
            model.eval()
            
        if phase == 'train' :
            for logs, frames in tqdm(trainloader) :
                out = model.run_on_batch(logs, frames)
                model.step(out['loss'])
                
                run_acc.extend([accuracy_score(frames[i].to('cpu'), out['output'][i].to('cpu')) for i in range(frames.shape[0])])
                run_f1.extend([f1_score(frames[i].to('cpu'), out['output'][i].to('cpu'), average='weighted') for i in range(frames.shape[0])])
            
            train_acc.append(sum(run_acc)/len(run_acc))
            train_f1.append(sum(run_f1)/len(run_f1))
            
        if phase == 'validation' :
            for logs, frames in tqdm(testloader) :
                out = model.run_on_batch(logs, frames)
                
                run_acc.extend([accuracy_score(frames[i].to('cpu'), out['output'][i].to('cpu')) for i in range(frames.shape[0])])
                run_f1.extend([f1_score(frames[i].to('cpu'), out['output'][i].to('cpu'), average='weighted') for i in range(frames.shape[0])])
            
            test_acc.append(sum(run_acc)/len(run_acc))
            test_f1.append(sum(run_f1)/len(run_f1))
            
    print(epoch+1, train_acc[-1], test_acc[-1], train_f1[-1], test_f1[-1])
    
    
    torch.save(model.state_dict(), f'saved_models/model_5/model_{epoch+1}_{train_acc[-1]}_{train_f1[-1]}_{test_acc[-1]}_{test_f1[-1]}.pth')
