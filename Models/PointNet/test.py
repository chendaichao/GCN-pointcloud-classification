import torch
from torch import nn
import torch.utils.data as Data
import os, sys

from data import ModelNet40
from model import *
from utils import *

import argparse
parser = argparse.ArgumentParser(description='Training PointNet.')
parser.add_argument('-n', '--num_points', type=int, default = 2048,
                    help="The number of points sampled from the pointcloud.")
parser.add_argument('-m', '--model', type=str, default = "PointNet.pt",
                    help="The file in which the trained model will be saved.")
args = parser.parse_args()

save_name = args.model

########### loading data ###########

num_points = args.num_points
test_data = ModelNet40(num_points, 'test')

print("test data size: ", len(test_data))

def collate_fn(batch):
    Xs = torch.stack([X for X, _ in batch])
    Ys = torch.tensor([Y for _, Y in batch], dtype = torch.long)
    return Xs, Ys

test_iter = Data.DataLoader(test_data, batch_size = 64, collate_fn = collate_fn)
    
############### loading model ####################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = PointNet(nfeat=3, nclass=40, dropout=0.3)
net.to(device)
print(net)
    
checkpoint = torch.load(save_name, map_location = device)
net.load_state_dict(checkpoint["net"])
        
############### testing ##########################

loss_func = torch.nn.NLLLoss()
loss, acc = evaluate_model(test_iter, net, loss_func)
print('test loss = %.6f,  test acc = %.6f' % (loss, acc))
