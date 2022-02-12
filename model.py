import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
from sklearn.metrics import accuracy_score, f1_score
from unet import UNet
import numpy as np
import dill
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MiniModel(nn.Module) :
    def __init__(self, out, device=device) :
        super(MiniModel, self).__init__()
        
        self.device = device
        
        self.conv1 = nn.Conv1d(300, 128, 3)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.conv3 = nn.Conv1d(128, 300, 3)
        
        self.bn = nn.BatchNorm1d(300)
        
        self.lin = nn.Linear(14, out)
        
    def forward(self, x) :    
        x = torch.relu(F.max_pool1d((self.conv1(x)), 2))
        x = torch.relu(F.max_pool1d((self.conv2(x)), 2))
        x = torch.relu(F.max_pool1d(self.bn(self.conv3(x)), 2))
        x = torch.relu(self.lin(x))
        return x.transpose(1,2)
    

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
        
        self.onset_detector = MiniModel(2, device)
        self.frameset_detector = MiniModel(5, device)
        
        self.loss_crit = nn.CrossEntropyLoss() # (weight=torch.Tensor([0,1]))
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
        
        
        return self.onset_detector(x), self.frameset_detector(x)
    
    def run_on_batch(self, x, y_onset=[], y_frameset=[]) :
        x.to(self.device)
        if len(y_onset) == 0 or len(y_frameset) == 0:
            return self(x)
        
        else :
            y_onset.to(self.device)
            y_frameset.to(self.device)
            pred_onset, pred_frameset = self(x)
            loss = self.loss_crit(pred_onset, y_onset) + self.loss_crit(pred_frameset, y_frameset)
            return {'output' : (pred_onset.argmax(1), pred_frameset.argmax(1)), 'loss' : loss}
        
    def step(self, loss) :
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        

train_X = dill.load(open('use_dataset/logmels_patch300_train', 'rb'))
train_Y_onset = dill.load(open('use_dataset/onset_patch300_train', 'rb')).long()
train_Y_frameset = dill.load(open('use_dataset/frameset_patch300_train', 'rb')).long()
train_Y_v = dill.load(open('use_dataset/v_patch300_train', 'rb'))

test_X = dill.load(open('use_dataset/logmels_patch300_test', 'rb'))
test_Y_onset = dill.load(open('use_dataset/onset_patch300_test', 'rb')).long()
test_Y_frameset = dill.load(open('use_dataset/frameset_patch300_test', 'rb')).long()
test_Y_v = dill.load(open('use_dataset/v_patch300_test', 'rb'))

class BeatBoxDataset(Dataset):
    def __init__(self, input_, output_onset_, output_frameset_, vs):
        super().__init__()
        self.input_ = input_
        self.output_onset = output_onset_
        self.output_frameset = output_frameset_
        self.vs = vs

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.input_[idx], self.output_onset[idx], self.output_frameset[idx], self.vs[idx]

trainloader = DataLoader(BeatBoxDataset(train_X, train_Y_onset, train_Y_frameset, train_Y_v), batch_size=64)
testloader = DataLoader(BeatBoxDataset(test_X, test_Y_onset, test_Y_frameset, test_Y_v), batch_size=64)

EPOCHS = 5000
model = Model(device)

def convert_sets(set_, v) :
    import numpy as np
    time_sets_ = []
    
    for i in range(len(set_)) :
#         print(set_[i])
        if set_[i] != 0 :
            time_sets_.append(i/v)
        
    return np.array(time_sets_)

def evaluate(true_sets, pred_sets, vs) :
    
    from mir_eval.onset import evaluate as onset_evaluation 
    import numpy as np
    
    f1 = []
    
    for i in range(len(true_sets)) :
#         print(true_sets[/i])
        true_time_set = convert_sets(true_sets[i], vs[i])
        pred_time_set = convert_sets(np.round(pred_sets[i]), vs[i])
        
        f1.append(onset_evaluation(true_time_set, pred_time_set, window=0.02).pop('F-measure') )
        
    return np.mean(f1)

train_onset_f1 = [0]
train_offset_f1 = [0]

test_onset_f1 = [0]
test_offset_f1 = [0]

train_onset_f1 = [0]
train_frameset_f1 = [0]

test_onset_f1 = [0]
test_frameset_f1 = [0]

for epoch in range(EPOCHS) :
    for phase in ['train', 'validation'] :
        run_f1_ons = []        
        run_f1_frame = []
        
        if phase == 'train' :
            model.train()
        else :
            model.eval()
            
        if phase == 'train' :
            for logs, onset, frameset, vs in tqdm(trainloader) :
                out = model.run_on_batch(logs, onset, frameset)
                model.step(out['loss'])
                
                run_f1_ons.append(evaluate(onset, out['output'][0], vs))
                run_f1_frame.extend([f1_score(frameset[i], out['output'][1][i], average='weighted') for i in range(len(logs))])
    
            train_frameset_f1.append(np.mean(run_f1_frame))
            train_onset_f1.append(np.mean(run_f1_ons))
                
        if phase == 'validation' :
            for logs, onset, frameset, vs in tqdm(testloader) :
                out = model.run_on_batch(logs, onset, frameset)
                
                run_f1_ons.append(evaluate(onset, out['output'][0], vs))
                run_f1_frame.extend([f1_score(frameset[i], out['output'][1][i], average='weighted') for i in range(len(logs))])
            
            test_frameset_f1.append(np.mean(run_f1_frame))
            test_onset_f1.append(np.mean(run_f1_ons))

    print(epoch+1, train_onset_f1[-1], train_frameset_f1[-1], test_onset_f1[-1], test_frameset_f1[-1])
    
    torch.save(model.state_dict(), f'saved_models/model_15/model_{epoch+1}_{train_onset_f1[-1]}_{train_frameset_f1[-1]}_{test_onset_f1[-1]}_{test_frameset_f1[-1]}.pth')
        
