#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified from 
  https://github.com/WangYueFt/dgcnn/blob/master/pytorch/data.py
by
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""

import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import ortho_group

def download():
    BASE_DIR = os.path.dirname("../../")
    DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    download()
    BASE_DIR = os.path.dirname("../../")
    DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    # xyz_rotation = ortho_group.rvs(3)
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    #translated_pointcloud = np.add(np.dot(np.multiply(pointcloud, xyz1), xyz_rotation), xyz2).astype('float32')
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def pointcloud2graph(pointcloud, all_edges, threshold = 0.11):
    N = all_edges.shape[0]
    dist = pointcloud.repeat((N, 1, 1))
    dist = (dist - pointcloud.unsqueeze(1)).norm(dim = 2, keepdim = False)
    threshold *= dist.max().cpu()
    k = threshold**2/2
    mask = dist < threshold
    dist = torch.exp(-dist**2 / k) * (mask).float()
    edges = all_edges[mask]
    rowsum = dist.sum(dim = 1)
    factor = 1. / rowsum[edges[:, 0]].float()
    factor[factor>1] = 0.
    #adj = torch.sparse.FloatTensor(edges.t(), torch.ones(edges.shape[0], dtype = torch.float) * factor, torch.Size([N,N]))
    #return adj
    return (edges, dist[mask] * factor) #edges and their values in adjacency matrix

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition 
        
        tmp = torch.LongTensor(list(range(num_points)))
        idx = torch.zeros((num_points, num_points, 2), dtype = torch.long)
        idx[:, :, 0] = tmp.unsqueeze(1)
        idx[:, :, 1] = tmp.unsqueeze(0)
        self.all_edges = idx

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = jitter_pointcloud(translate_pointcloud(pointcloud))
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        pointcloud = torch.from_numpy(pointcloud)
        return pointcloud, pointcloud2graph(pointcloud, self.all_edges), torch.from_numpy(label)

    def __len__(self):
        return self.data.shape[0]