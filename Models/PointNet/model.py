import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import GraphConvolution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, adjs = None):
        if (adjs is None):
            xs, adjs = xs
        
        num_points = xs.shape[1]
        
        xs = torch.cat(tuple(xs), dim=0)
        xs = xs.to(device)
        adjs = adjs.to(device)
        res = F.relu(self.gc1(xs, adjs))
        res = F.relu(self.gc2(res, adjs))
        res = self.dropout(res)
        ys = torch.stack(torch.split(res, num_points, dim=0)).to(device)
        return ys

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
    def forward(self, x, dim=1, keepdim = False):
        res, _ = torch.max(x, dim=dim, keepdim = keepdim)
        return res

class Permute(nn.Module):
    def __init__(self, param):
        super(Permute, self).__init__()
        self.param = param
    def forward(self, x):
        return x.permute(self.param)

class BatchNorm(nn.Module):
    '''
        Perform batch normalization.
        Input: A tensor of size (N, M, feature_dim), or (N, feature_dim, M) (available when feature_dim != M), 
                or (N, feature_dim)
        Output: A tensor of the same size as input.
    '''
    def __init__(self, feature_dim):
        super(BatchNorm, self).__init__()
        self.feature_dim = feature_dim
        self.batchnorm = nn.BatchNorm1d(feature_dim)
        self.permute = Permute((0, 2, 1))
    def forward(self, x):
        if (len(x.shape) == 3) and (x.shape[-1] == self.feature_dim):
            return self.permute(self.batchnorm(self.permute(x)))
        else:
            return self.batchnorm(x)
    
class MLP(nn.Module):
    def __init__(self, hidden_size, batchnorm = True, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i == len(hidden_size) - 2) and (last_activation):
                if (batchnorm):
                    q.append(("Batchnorm_%d" % i, BatchNorm(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)
    
class TNet(nn.Module):
    def __init__(self, nfeat):
        super(TNet, self).__init__()
        self.nfeat = nfeat
        self.tnet = nn.Sequential(MLP((nfeat, 64, 128, 1024)), MaxPooling(), 
                                  BatchNorm(1024), MLP((1024, 512, 256, nfeat*nfeat))
                                 )
        
    def forward(self, x):
        batch_size = x.shape[0]
        return self.tnet(x).view(batch_size, self.nfeat, self.nfeat)
    
class PointNet(nn.Module):
    def __init__(self, nfeat, nclass, dropout = 0):
        super(PointNet, self).__init__()

        self.input_transform = TNet(nfeat)
        self.mlp1 = nn.Sequential(BatchNorm(3), MLP((nfeat, 64, 64)))
        self.feature_transform = TNet(64)
        self.mlp2 = nn.Sequential(BatchNorm(64), MLP((64, 64, 128, 1024)))
        self.maxpooling = MaxPooling()
        self.mlp3 = nn.Sequential(BatchNorm(1024), MLP((1024, 512, 256)), nn.Dropout(dropout), nn.Linear(256, nclass))
        
        self.eye64 = torch.eye(64).to(device)

    def forward(self, xs):
        batch_size = xs.shape[0]
        
        transform = self.input_transform(xs)
        xs = torch.stack([torch.mm(xs[i],transform[i]) for i in range(batch_size)])
        xs = self.mlp1(xs)
        
        transform = self.feature_transform(xs)
        xs = torch.stack([torch.mm(xs[i],transform[i]) for i in range(batch_size)])
        xs = self.mlp2(xs)
        
        xs = self.mlp3(self.maxpooling(xs))
        
        if (self.training):
            transform_transpose = transform.transpose(1, 2)
            tmp = torch.stack([torch.mm(transform[i], transform_transpose[i]) for i in range(batch_size)])
            L_reg = ((tmp - self.eye64) ** 2).sum() / batch_size
            
        return (F.log_softmax(xs, dim=1), L_reg) if self.training else F.log_softmax(xs, dim=1)
