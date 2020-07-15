# Pointcloud Classification with PointNet and GCNs

This is a project using PointNet and GCNs for pointcloud classification. We implement the original PointNet<sup>[1]</sup> and make some attempts on incorporating graph convolutional networks<sup>[2]</sup> into PointNet. Both models are trained and tested on the ModelNet40<sup>[3]</sup> Dataset.

## Model Description

This is an illustration of PointNet, taken from Figure 2 in [1]. The max-pooling operation aggregates the local feature from each point and ensures permutation invariance. The T-Net are a mini version of PointNet, consisting of a shared *MLP(64, 128, 1024)*, a max-pooling operator and a *MLP(512, 256, k×k)*, which regress the *n×k* input to a *k×k* transform. 

![illustration for PointNet](figure.png)

To incorporate graph convolutional  networks, I modified the three *MLP(64, 128, 1024)* s (one in the main PointNet, two in TNets) by adding GCN layers, illustrated as follows. There are also dense connections between the GCN layers, combining features at different levels. To compare their performances, their output dimensions are kept identical (both are 1024). In addition, GCNs require graphs as an extra input, which are constructed by a thresholded Gaussian kernel weighting function. Hopefully the modification will endow the network with more capacity to capture structural information, and thus result in better performances.

![incorporating GCNs to PointNet](figure2_gcn.png)

## Implementation

We implement both models in *pytorch*. Codes are stored in `Models/PointNet/` and `Models/PointNet+GCN`, respectively. The dataset classes in `Models/*/data.py` are modified from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/data.py. The pytorch implementation for GCN ( `Models/PointNet+GCN/layers.py` and the class `GCN` in`Models/PointNet+GCN/models.py`) are borrowed from https://github.com/tkipf/pygcn/blob/master/pygcn/layer.py. 

Similar to [1], an regularization loss term (with weight 0.001) encouraging the 64-by-64 feature transform matrix to be close to an orthonormal matrix is added to the total loss. We use Adam optimizer, with learning rate initially set as 0.001 and decayed by 0.95 every epoch. We test both models for *#points*=256, 512, 1024.

**Results.** The results of both models are listed below.

|                | PointNet |     PointNet+GCN     |
| -------------: | :------: | :------------------: |
|  *#points*=256 | 87.2771% |     **87.4392%**     |
|  *#points*=512 | 87.3582% |     **87.4797%**     |
| *#points*=1024 | 89.5057% | *(still testing...)* |

## How to Use

First switch the working directory, by

```shell
cd Models/"PointNet" 
```

```shell
cd Models/"PointNet+GCN" 
```

### Training

Run

```shell
python train.py -lr=1e-3 -n=512 --model=PointNet.pt
```

```shell
python train.py -lr=1e-3 -n=512 --model=PointNetGCN.pt
```

for training. The ModelNet40 dataset will be automatically downloaded (you can also download it manually from https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip , and unzip the file to `./Datasets/` directory). 

The trained model will be stored in `PointNet.pt` or `PointNetGCN.pt`. 

### Testing

Run

```shell
python test.py -n=512 --model=PointNet.pt
```

```shell
python test.py -n=512 --model=PointNetGCN.pt
```

for testing the trained model (loaded from `PointNet.pt` or `PointNetGCN.pt`).

### Using pretrained models

If you want to use our pretrained models, please download them from the following links:

*I would upload them later.*

Please save the `*.pt`  files in `Models/"PointNet"` and `Models/"PointNet+GCN"`, rename them as `PointNet.pt` and `PointNetGCN.pt`, respectively, and run the previous testing command.

## References

[1] Charles, R. Q., Su, H., Kaichun, M., & Guibas, L. J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *computer vision and pattern recognition*.

[2] Kipf, T., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *international conference on learning representations*.

[3] Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., & Xiao, J. (2015). 3D ShapeNets: A deep representation for volumetric shapes. *computer vision and pattern recognition*.
