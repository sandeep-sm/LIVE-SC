"""
DDP training for Contrastive Learning
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.multiprocessing as mp

from options.train_options import TrainOptions
from learning.contrast_trainer import ContrastTrainer
from networks.build_backbone import build_model
from datasets.util import build_contrast_loader, build_contrast_loader_testing
from memory.build_memory import build_mem

import csv
import os
from scipy import io
import numpy as np

import subprocess

def main():
    args = TrainOptions().parse()
    print("in main")

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError('Currently only DDP training')


def main_worker(gpu, ngpus_per_node, args):

    print("in main worker")
    # initialize trainer and ddp environment
    trainer = ContrastTrainer(args)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    print("init_ddp_environment-complete")

    # build model
    model, model_ema = build_model(args)

    print("build_model-complete")

    # build dataset
    train_dataset, train_loader, train_sampler = \
        build_contrast_loader(args, ngpus_per_node)

    print("dataloader-complete") 

    # build memory
    contrast = build_mem(args, len(train_dataset))
    contrast.cuda()

    # build criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # wrap up models
    model, model_ema, optimizer = trainer.wrap_up(model, model_ema, optimizer)

    # checkpoint = torch.load('/work/08804/smishra/ls6/PyContrast/pycontrast/train_2_save/MoCov2_resnet50_RGB_Jig_False_moco_aug_B_mlp_0.2_cosine_warm/ckpt_epoch_54.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['model'])

    # optional step: synchronize memory
    trainer.broadcast_memory(contrast)

    # check and resume a model
    start_epoch = trainer.resume_model(model, model_ema, contrast, optimizer)

    # init tensorboard logger
    trainer.init_tensorboard_logger()

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        trainer.adjust_learning_rate(optimizer, epoch)

        outs = trainer.train(epoch, train_loader, model, model_ema,
                             contrast, criterion, optimizer)

        # log to tensorbard
        trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'])

        # save model
        trainer.save(model, model_ema, contrast, optimizer, epoch)


def tester():
    
    args = TrainOptions().parse()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    # root_folder = '/work/08804/smishra/ls6/PNG_training/train/'
    root_folder = '/work/08804/smishra/ls6/DS_test_data/'

    feature_mat_file = 'ShareChat_Contrast_features.mat'
    feature_mat = []

    with open('ShareChat_metadata.csv', newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        index = 0
        for row in reader:
            # dest_folder = '/work/08804/smishra/ls6/DS_test_data/'
            data_folder = os.path.join(root_folder, row['video_ID'][:-4])
            # dest_folder = os.path.join(dest_folder, row['video_ID'][:-4])
            # cmd = 'mkdir ' + dest_folder
            # subprocess.run(cmd, shell=True)
            # cmd = 'cp -r ' + data_folder + ' ' + dest_folder + '/.'
            # subprocess.run(cmd, shell=True)
            if args.multiprocessing_distributed:
                args.world_size = ngpus_per_node * args.world_size
                # mp.spawn(test_model, nprocs=1, args=(ngpus_per_node, args, data_folder))
                output = test_model(args, data_folder)
                feature_mat.append(output)
            else:
                raise NotImplementedError('Currently only DDP training')
            print(data_folder)
            index+=1
            print('Video '+str(index)+' complete')

    io.savemat(feature_mat_file, {"data": np.array(feature_mat) })

def test_model(args, data_folder):

    # initialize trainer and ddp environment
    # trainer = ContrastTrainer(args)
    # trainer.init_ddp_environment(gpu, ngpus_per_node)

    # build model
    model, _ = build_model(args)
    model = torch.nn.DataParallel(model)

    # build dataset
    train_dataset, train_loader = \
        build_contrast_loader_testing(args, data_folder)

    # build memory
    # contrast = build_mem(args, len(train_dataset))
    # contrast.cuda()

    # build criterion and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(),
                                # lr=args.learning_rate,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay)

    # wrap up models
    # model, model_ema, optimizer = trainer.wrap_up(model, model_ema, optimizer)

    # optional step: synchronize memory
    # trainer.broadcast_memory(contrast)

    # check and resume a model
    # start_epoch = trainer.resume_model(model, model_ema, contrast, optimizer)
    # checkpoint = torch.load('/work/08804/smishra/ls6/PyContrast/pycontrast/train_2_save/MoCov2_resnet50_RGB_Jig_False_moco_aug_B_mlp_0.2_cosine_warm/ckpt_epoch_54.pth', map_location='cpu')
    # checkpoint = torch.load('/work/08804/smishra/ls6/PyContrast/pycontrast/train_5_normal_save/MoCov2_resnet50_RGB_Jig_False_moco_aug_B_mlp_0.2_cosine/ckpt_epoch_46.pth', map_location='cpu')
    # checkpoint = torch.load('/work/08804/smishra/ls6/PyContrast/pycontrast/train_4_frame_batch_save/MoCov2_resnet50_RGB_Jig_False_moco_aug_B_mlp_0.2_cosine/ckpt_epoch_5.pth', map_location='cpu')
    ckpt_path = '/work/08804/smishra/ls6/PyContrast/pycontrast/train_4_dist_nopretrain_2048_frame_batch7_save/MoCov2_resnet50_RGB_Jig_False_moco_aug_B_mlp_0.2_cosine/noderank_0_ckpt_epoch_3.pth'
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.cuda()
    model.eval()

    # init tensorboard logger
    # trainer.init_tensorboard_logger()
    final_feature_vector = torch.zeros((2048))
    total_length=0
    h, w = 1, 1 
    with torch.no_grad():
        for idx, data in enumerate(train_loader):
            output = model(data[0].cuda()).sum(axis=0).detach().cpu()
            output2 = output
            # h,w = output.shape[1], output.shape[2]
            # output2 = output.sum(axis=-1).sum(axis=-1)
            final_feature_vector += output2
            # print(len(data[0]))
            total_length += len(data[0])

    del model

    return (final_feature_vector.numpy())/(total_length*h*w)      

if __name__ == '__main__':
    main()
    # tester()
