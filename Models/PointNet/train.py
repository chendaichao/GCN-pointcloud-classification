import torch
from torch import nn
import torch.utils.data as Data
import os, sys

from data import ModelNet40
from model import *
from utils import *

import argparse
parser = argparse.ArgumentParser(description='Training PointNet.')
parser.add_argument('-lr', '--learning_rate', type=float, default = 0.001,
                    help='Initial learning rate.')
parser.add_argument('-n', '--num_points', type=int, default = 2048,
                    help="The number of points sampled from the pointcloud.")
parser.add_argument('-m', '--model', type=str, default = "PointNet.pt",
                    help="The file in which the trained model will be saved.")
args = parser.parse_args()

save_name = args.model

########### loading data ###########

num_points = args.num_points
train_data = ModelNet40(num_points)
test_data = ModelNet40(num_points, 'test')

train_size = int(0.9 * len(train_data))
valid_size = len(train_data) - train_size
train_data, valid_data = Data.random_split(train_data, [train_size, valid_size])
valid_data.partition = 'valid'
train_data.partition = 'train'

print("train data size: ", len(train_data))
print("valid data size: ", len(valid_data))
print("test data size: ", len(test_data))

def collate_fn(batch):
    Xs = torch.stack([X for X, _ in batch])
    Ys = torch.tensor([Y for _, Y in batch], dtype = torch.long)
    return Xs, Ys

train_iter  = Data.DataLoader(train_data, shuffle = True, batch_size = 64, collate_fn = collate_fn)
valid_iter = Data.DataLoader(valid_data, batch_size = 64, collate_fn = collate_fn)
test_iter = Data.DataLoader(test_data, batch_size = 64, collate_fn = collate_fn)
    
############### loading model ####################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = PointNet(nfeat=3, nclass=40, dropout=0.3)
net.to(device)
print(net)

############### training #########################

lr  = args.learning_rate
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)
loss = nn.NLLLoss()

def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

retrain = True
if os.path.exists(save_name):
    print("Model parameters have already been trained before. Retrain ? (y/n)")
    ans = input()
    if not (ans == 'y'):
        checkpoint = torch.load(save_name, map_location = device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g['lr'] = lr
        
train_model(train_iter, valid_iter, net, loss, optimizer, device = device, max_epochs = 1000, 
            adjust_lr = adjust_lr, early_stop = EarlyStop(patience = 20, save_name = save_name))
    

############### testing ##########################

loss, acc = evaluate_model(test_iter, net, loss)
print('test acc = %.6f' % (acc))






