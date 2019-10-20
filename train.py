from __future__ import print_function
from mxnet.gluon import nn, loss as gloss
from mxnet import autograd, nd, init
from utils.patch_config import patch_config_as_nothrow
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb


import numpy as np
import mxnet as mx
import argparse
import importlib
import pprint
import sys
import os


def train_net(config):
    General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, ModelParam, \
    OptimizeParam, TestParam, transform, data_name, label_name, metric_list = config.generate_config(is_train=True)
    pGen     = patch_config_as_nothrow(General)
    pKv      = patch_config_as_nothrow(KvstoreParam)
    pRpn     = patch_config_as_nothrow(RpnParam)
    pRoi     = patch_config_as_nothrow(RoiParam)
    pBbox    = patch_config_as_nothrow(BboxParam)
    pDataset = patch_config_as_nothrow(DatasetParam)
    pModel   = patch_config_as_nothrow(ModelParam)
    pOpt     = patch_config_as_nothrow(OptimizeParam)
    pTest    = patch_config_as_nothrow(TestParam)

    gpus = pKv.gpus
    if len(gpus) == 0:
        ctx = [mx.cpu()]
    else:
        ctx = [mx.gpu(i) for i in gpus]

    input_batch_size = pKv.batch_image * len(ctx)
    pretrain_prefix  = pModel.pretrain.prefix
    pretrain_epoch   = pModel.pretrain.epoch
    save_path        = os.path.join('experiments', pGen.name)
    model_prefix     = os.path.join(save_path, 'checkpoint')
    begin_epoch      = pOpt.schedule.begin_epoch
    end_epoch        = pOpt.schedule.end_epoch
    lr_steps         = pOpt.schedule.lr_steps


    ## load dataset
    if pDataset.Dataset == 'widerface':
        image_set = pDataset.image_set
        roidb = load_gt_roidb(pDataset.Dataset, image_set, root_path='data', dataset_path='data/widerface', flip=True)




    net = pModel.train_network

    if pOpt.schedule.begin_epoch != 0:
        net.load_model(model_prefix, pOpt.schedule.begin_epoch)
    else:
        net.load_model(pretrain_prefix)

    print('hello github!')













def parse_args():
    parser = argparse.ArgumentParser(description='Training RetinaFace')
    ## general
    parser.add_argument('--config', default='config/widerface_resnet50.py', type=str, help='config file path')
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config



if __name__ == '__main__':
    config = parse_args()
    train_net(config)

# parse_args()



