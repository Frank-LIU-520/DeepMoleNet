#!/usr/bin/env python
# encoding: utf-8
# File Name: DeepMoleNet.py
# Author: Ziteng Liu@ Nanjing University      
# E-mail: njuziteng@hotmail.com
# twitter: MarriotteNJU
# Create Time: 2020/7/08 8:55

import math
import torch_geometric.transforms as T
from Alchemy_dataset import TencentAlchemyDataset
from torch_geometric.nn import NNConv, Set2Set,global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops,softmax
from torch.optim import lr_scheduler
import os
import pandas as pd
import numpy as np
import json
from ase.units import Hartree, eV, Bohr, Ang
import torch
import torch.nn as nn
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device
        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index
        data.ele=data.x[:,46:186]
        data.x=data.x[:,0:38]
        return data


transform = T.Compose([Complete(), T.Distance(norm=False)])
all_dataset = TencentAlchemyDataset(root='data-bin', mode='dev', transform=transform)

c=Hartree / eV
all_dataset.data.y=all_dataset.data.y*torch.tensor([1,1,c,c,c,1,c,c,c,c,c,1])

all_dataset.data.y=torch.transpose(torch.transpose(all_dataset.data.y,1,0)[[0,1,2,3,4,5 ,6 , 11, 7, 8, 9, 10]],1,0)

atomref = nn.Embedding.from_pretrained(
            torch.from_numpy(np.load("data-bin/atom_ref.npy")[:,1:5]).float()    #atomref needed for calculation
            )
validation_split = .1
shuffle_dataset = True
random_seed= 666


dataset_size = len(all_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)


train_indices ,val_indices,test_indices = indices[:110000],indices[110000:120000],indices[120000:]

train_dataset ,valid_dataset,test_dataset= all_dataset[torch.LongTensor(train_indices)],all_dataset[torch.LongTensor(val_indices)],all_dataset[torch.LongTensor(test_indices)]

y_mean=torch.mean(train_dataset.data.y,dim=0)
y_std=torch.std(train_dataset.data.y,dim=0)

inten_mean=torch.mean(train_dataset.data.y[:,:8],dim=0)
inten_std=torch.std(train_dataset.data.y[:,:8],dim=0)

exten_mean=torch.Tensor([[-4.2446, -4.2704, -4.2945, -3.9515]]) 
exten_std= torch.tensor([[0.1894, 0.1887, 0.1885, 0.1877]]) 

train_dataset.data.y[:,:8]=(train_dataset.data.y[:,:8]-inten_mean)/inten_std
valid_dataset.data.y[:,:8]=(valid_dataset.data.y[:,:8]-inten_mean)/inten_std

valid_loader = DataLoader(valid_dataset, batch_size=16)  
test_loader= DataLoader(test_dataset, batch_size=16)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=12)


class ScaleShift(nn.Module):
    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
    def forward(self,x):
        y=x*self.stddev+self.mean
        return y

class MPNN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=38,
                 edge_input_dim=41,
                 output_inten_dim=8,
                 output_exten_dim=4,
                 node_hidden_dim=200,
                 edge_hidden_dim=200,
                 num_step_message_passing=6,
                 num_step_set2set=12):
        super(MPNN, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_network = nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim,edge_hidden_dim*2),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim*2, node_hidden_dim * node_hidden_dim)
                )
        self.conv = NNConv(node_hidden_dim, node_hidden_dim, edge_network, aggr='mean', root_weight=False)
        self.lstm = nn.LSTM(node_hidden_dim, node_hidden_dim)
        self.att=nn.Sequential(
                nn.Linear(node_hidden_dim, 2*node_hidden_dim),
                nn.ReLU(),
                nn.Linear(2*node_hidden_dim, 2*node_hidden_dim),
                nn.ReLU(),
                nn.Linear(2*node_hidden_dim, node_hidden_dim),
                )
                
        self.set2set = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        self.output_inten=nn.Sequential(
                nn.Linear(2 * node_hidden_dim, node_hidden_dim),
                nn.ReLU(),
                nn.Linear(node_hidden_dim, output_inten_dim))
        self.output_exten=nn.Sequential(
                nn.Linear(node_hidden_dim, 2*node_hidden_dim),
                nn.ReLU(),
                nn.Linear(2*node_hidden_dim, 2*node_hidden_dim),
                nn.ReLU(),
                nn.Linear(2*node_hidden_dim,output_exten_dim))
        self.ss=ScaleShift(exten_mean,exten_std)
        self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(np.load("data-bin/atom_ref.npy")[:,1:5]).float()
                )
        self.lin_regular=nn.Sequential(
                nn.Linear(node_hidden_dim, 2*node_hidden_dim),
                nn.ReLU(),
                nn.Linear(2*node_hidden_dim, 2*node_hidden_dim),
                nn.ReLU(),
                nn.Linear(2*node_hidden_dim,140),
                )
    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = (out.unsqueeze(0),out.unsqueeze(0))

        for i in range(self.num_step_message_passing):
            att=self.att(h[0].squeeze(0))
            att=(out*att).sum(dim=-1,keepdim=True)
            att=torch.sigmoid(att)
            out=out*att
            out =self.conv(out, data.edge_index, data.edge_attr)
            out,h=self.lstm(out.unsqueeze(0),h)
            out=out.squeeze(0)
        ele=self.lin_regular(out)
        out_inten = self.set2set(out, data.batch)
        out_inten=self.output_inten(out_inten)
        out_exten=self.output_exten(out)
        out_exten=self.ss(out_exten)
        out_exten=self.atomref(data.x[:,7:8].long().squeeze(1))+out_exten
        out_exten=global_add_pool(out_exten,data.batch)
        return out_inten,out_exten,ele

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MPNN(node_input_dim=train_dataset.num_features).to(device)
lr=1e-05
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler_cosine)

print_every=5000


def train(epoch):
    model.train()
    loss_all = 0
    t=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out_inten,out_exten,ele = model(data)
        loss_inten = F.l1_loss(out_inten, data.y[:,:8])
        loss_exten = F.l1_loss(out_exten, data.y[:,8:])
        loss_ele=F.l1_loss(ele,data.ele)
        loss=loss_inten+loss_exten*0.5+loss_ele*16
        if t%print_every==0:
            print("loss is{:.7f} in iter{:d}".format(loss.item(),t))
        t+=1
        loss.backward()      
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    scheduler_warmup.step()
    print('lr rate', optimizer.param_groups[0]['lr'])
    return loss_all / len(train_loader.dataset)
def valid(epoch):
    with torch.no_grad():
        model.eval()
        loss_all=0
        mae_inten,mae_exten,valid_in_ex=0,0,0
        for data in valid_loader:
            data = data.to(device)
            out_inten,out_exten,ele = model(data)
            loss_inten = F.l1_loss(out_inten, data.y[:,:8])
            loss_exten = F.l1_loss(out_exten, data.y[:,8:])
            loss_ele=F.l1_loss(ele,data.ele)
            loss=loss_inten+loss_exten*0.5+loss_ele*16
            mae_inten+=torch.sum(torch.abs(out_inten-data.y[:,:8]),dim=0)
            mae_exten+=torch.sum(torch.abs(out_exten-data.y[:,8:]),dim=0)
            mae_inten_exten=torch.cat((mae_inten,mae_exten),0)
            loss_all += loss.item() * data.num_graphs
            
            valid_y_pred=torch.cat((out_inten*inten_std.to(device)+inten_mean.to(device),out_exten),1)
            data.y=torch.cat((data.y[:,:8]*inten_std.to(device)+inten_mean.to(device),data.y[:,8:]),1)
            valid_in_ex+= torch.sum(torch.abs(valid_y_pred-data.y),dim=0)
        return loss_all / len(valid_loader.dataset),mae_inten_exten/len(valid_loader.dataset),valid_in_ex/len(valid_loader.dataset)
        
def test(epoch):
    with torch.no_grad():
        model.eval()
        mae=0
        for data in test_loader:
            data = data.to(device)
            out_inten,out_exten,ele = model(data)
            y_inten = (out_inten*inten_std.to(device))+inten_mean.to(device)
            y_pred=torch.cat((y_inten,out_exten),1)
            mae+=torch.sum(torch.abs(y_pred-data.y),dim=0)
        mae=mae/len(test_dataset)
        std_mae=(torch.sum(mae/y_std.to(device).to(device),dim=0)/12)
        ln_mae=torch.sum(torch.log(mae/y_std.to(device).to(device)),dim=0)/12
        return mae,std_mae,ln_mae

epoch = 400

print("DeepMoleNet training start...")
history_std_mae=[1]
history_loss=[10]
for epoch in range(epoch):
    loss = train(epoch)      #error
    print('Epoch: {:03d}, Loss on train: {:.7f}'.format(epoch, loss))
    loss,mae_inten,mae_exten = valid(epoch)
    history_loss.append(loss)
    print('Epoch: {:03d}, Loss on valid: {:.7f}'.format(epoch, loss))
    print('Epoch: {:03d}, Mae on valid: '.format(epoch),mae_inten)
    print('Epoch: {:03d}, Mae real valid: '.format(epoch),mae_exten)

    if epoch >=300 and loss <= min(history_loss[:-1]): 
     tmae,std_mae,ln_mae = test(epoch)
     history_std_mae.append(std_mae)
     print('hisotyr std mae',history_std_mae[-2],std_mae)
     print('Epoch: {:03d}, mae on test:'.format(epoch), tmae)
     print('Epoch: {:03d}, std Mae on test: '.format(epoch),std_mae)
     print('Epoch: {:03d}, ln std Mae on test: '.format(epoch),ln_mae)
     torch.save(model, 'gilmeracsf16regnet-temp240.pkl')     
     if  std_mae< min(history_std_mae[:-1]):
      torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'data-bin/models/gilmeracsf16regattmpnn-epoch{:03d}loss{:.7f}.pkl'.format(epoch,loss))

    torch.save(model.state_dict(), './data-bin/models/DeepMoleNet_state_dict.pkl')
    torch.save(model, 'DeepMoleNet.pkl')
       



