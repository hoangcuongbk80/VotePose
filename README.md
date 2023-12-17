
# Voting and Attention-based Pose Relation Learning for Object Pose Estimation from 3D Point Clouds

## [Video](https://youtu.be/VHo3vSEYjeM)

## Introduction
This repository is code release for our paper.

In this repository, we provide implementation of the proposed method (with Pytorch):
1. Backbone and feature extraction can be found in [models/backbone_module.py](https://github.com/votepose/votepose/blob/master/models/backbone.py)
2. Voting object part centers and learning part-to-part correlation module can be found in [models/votepose.py](https://github.com/votepose/votepose/blob/master/models/votepose.py) and [models/proposal_module.py](https://github.com/votegrasp/votegrasp/blob/master/models/proposal_module.py)
3. Voting object centers and learning object-to-object correlation module can be found in [models/votepose.py](https://github.com/votegrasp/votegrasp/blob/master/models/votepose.py) and [models/proposal_module.py](https://github.com/votegrasp/votegrasp/blob/master/models/proposal_module.py)
4. Multi-task loss function can found in [models/loss_helper.py](https://github.com/votegrasp/votegrasp/blob/master/models/loss_helper.py)

## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/). It is required that you have access to GPUs. The code is tested with Ubuntu 18.04, Pytorch v1.1, CUDA 10.0 and cuDNN v7.4.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'

## Training

#### Data Preparation

Prepare data by running `python dataset/data.py --gen_data`

#### Train

To train a new model:

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset dataset --log_dir log

You can use `CUDA_VISIBLE_DEVICES=0,1,2` to specify which GPU(s) to use. Without specifying CUDA devices, the training will use all the available GPUs and train with data parallel (Note that due to I/O load, training speedup is not linear to the nubmer of GPUs used). Run `python train.py -h` to see more training options.
While training you can check the `log/log_train.txt` file on its progress.

#### Run predict

    python predict.py

## Citations

```
@article{hoang2022voting,
  title={Voting and attention-based pose relation learning for object pose estimation from 3d point clouds},
  author={Hoang, Dinh-Cuong and Stork, Johannes A and Stoyanov, Todor},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={4},
  pages={8980--8987},
  year={2022},
  publisher={IEEE}
}

```

## License
Licensed under the [MIT License](LICENSE)
