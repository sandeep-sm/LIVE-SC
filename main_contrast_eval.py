"""
DDP training for Contrastive Learning
"""
from __future__ import print_function

import torch, torchvision
import torch.nn as nn
import torch.utils.data.distributed
import torch.multiprocessing as mp

from options.train_options import TrainOptions
from learning.contrast_trainer import ContrastTrainer
from networks.build_backbone import build_model
from datasets.util import build_contrast_loader
from memory.build_memory import build_mem
from datasets.dataset import test_datasetSH
from torch.utils.data import DataLoader

import csv
import os
from scipy import io
import numpy as np
import time
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

def tester():
    
    args = TrainOptions().parse()

    root_folder = '/work/08804/smishra/ls6/scratch_shortcut/DS_test_data/'
    # root_folder = '/work/08804/smishra/ls6/LIVE-ShareChat_MOS_videos/test_evaluate'
    dest_folder = 'feature_bank'

    # best
    # exp_name = 'exp4_IQA_20kvid_lr_0.08_epoch500'
    # ckpt_num = 500
    exp_name = 'exp4_IQA_20kvid_lr_0.1_epoch800'
    ckpt_num = 800

    exp_name = 'exp4_IQA_20kvid_512dim_lr_0.1_epoch800'
    ckpt_num = 250
    predicted_feature_mat_file = 'ShareChat_features_'+str(exp_name)+'_ckpt_'+str(ckpt_num)+'.mat'
    predicted_feature_mat = []
    predicted_feature_mat_file_path = os.path.join(dest_folder, predicted_feature_mat_file)

    head_feature_mat_file = 'ShareChat_features_head_'+str(exp_name)+'_ckpt_'+str(ckpt_num)+'.mat'
    head_feature_mat = []
    head_feature_mat_file_path = os.path.join(dest_folder, head_feature_mat_file)

    # build model
    model, _ = build_model(args)
    model = torch.nn.DataParallel(model)

    # check and resume a model
    ckpt_path = '/work/08804/smishra/ls6/scratch_shortcut/MOCO_VQA_experiments/'+exp_name+'/MoCov2_resnet50_RGB_Jig_False_moco_aug_B_mlp_0.2_cosine/ckpt_epoch_'+str(ckpt_num)+'.pth'
    
    # class R50(nn.Module):
    #     def __init__(self):
    #         super(R50, self).__init__()
    #         encoder = torchvision.models.resnet50(pretrained=False)
    #         self.encoder = nn.Sequential(*list(encoder.children())[:-2])

    #         self.n_features = 2048
    #         self.projector = nn.Sequential(
    #             nn.Linear(self.n_features, self.n_features, bias=False),
    #             nn.BatchNorm1d(self.n_features),
    #             nn.ReLU(),
    #             nn.Linear(self.n_features, 128, bias=False),
    #             nn.BatchNorm1d(128),
    #         )
    #         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    #     def forward(self, x):
    #         out = self.avgpool(self.encoder(x))
    #         return out

    # model = R50()
    # ckpt_path = 'CONTRIQUE_checkpoint25.tar'
    # print(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # model.load_state_dict(checkpoint)
    # model = torch.nn.DataParallel(model)

    model.cuda()
    model.eval()

    running_loss = 0

    with open('ShareChat_metadata.csv', newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        index = 0
        for row in reader:
            data_folder = os.path.join(root_folder, row['video_ID'][:-4])

            # set init time
            start_time = time.time()

            # run model
            encoder_feats = test_model(args, data_folder, model)
            # encoder_feats, head_feats = test_model_dual_output(args, data_folder, model)

            # measure runtime
            print("--- %s seconds ---" % (time.time() - start_time))
            predicted_feature_mat.append(encoder_feats)
            # head_feature_mat.append(head_feats)
            print(data_folder)
            index+=1
            print('Video '+str(index)+' complete')

            if index%100 == 0:
                io.savemat(predicted_feature_mat_file_path, {"data": np.array(predicted_feature_mat) })

    io.savemat(predicted_feature_mat_file_path, {"data": np.array(predicted_feature_mat) })
    # io.savemat(head_feature_mat_file_path, {"data": np.array(head_feature_mat) })

def test_model(args, data_folder, model):

    # build dataset
    test_data = test_datasetSH(data_folder)
    test_loader= DataLoader(test_data, batch_size=15, shuffle=False, num_workers=32, pin_memory=True)

    # final_feature_vector = torch.zeros((2048))
    total_length=0
    h, w = 1, 1 
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # output = model(data[0].cuda()).sum(axis=0).detach().cpu()
            if idx == 0: 
                output = model.module.encoder(data.cuda())
                # output = model(data.cuda())
            else:
                output = torch.cat((output, model.module.encoder(data.cuda())),dim=0)    
                # output = torch.cat((output, model(data.cuda())),dim=0)    

            total_length += len(data[0])

        output = output.mean(dim=0)
    return output.cpu().numpy()

def test_model_dual_output(args, data_folder, model):

    # build dataset
    test_data = test_datasetSH(data_folder)
    test_loader= DataLoader(test_data, batch_size=15, shuffle=False, num_workers=32, pin_memory=True)

    # final_feature_vector = torch.zeros((2048))
    total_length=0
    h, w = 1, 1 
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # output = model(data[0].cuda()).sum(axis=0).detach().cpu()
            if idx == 0: 
                output = model.module.encoder(data.cuda())
                head = model.module.head[0](output)
            else:
                temp_output = model.module.encoder(data.cuda())
                output = torch.cat((output, temp_output),dim=0)       
                head = torch.cat((head, model.module.head[0](temp_output)),dim=0) 

            total_length += len(data[0])

        output = output.mean(dim=0)
        head = head.mean(dim=0)
    return output.cpu().numpy(), head.cpu().numpy()

def test_modelMS(args, data_folder, model):

    # build dataset
    test_data = test_datasetSH(data_folder, use_scale=0)
    test_loader= DataLoader(test_data, batch_size=15, shuffle=False, num_workers=32, pin_memory=True)

    # final_feature_vector = torch.zeros((2048))
    total_length=0
    h, w = 1, 1 
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # output = model(data[0].cuda()).sum(axis=0).detach().cpu()
            if idx == 0: 
                output = model.module.encoder(data.cuda())
            else:
                output = torch.cat((output, model.module.encoder(data.cuda())),dim=0)        

            total_length += len(data[0])

        output1 = output.mean(dim=0)
    
    # build dataset
    test_data = test_datasetSH(data_folder, use_scale=1)
    test_loader= DataLoader(test_data, batch_size=15, shuffle=False, num_workers=32, pin_memory=True)

    # final_feature_vector = torch.zeros((2048))
    total_length=0
    h, w = 1, 1 
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # output = model(data[0].cuda()).sum(axis=0).detach().cpu()
            if idx == 0: 
                output = model.module.encoder(data.cuda())
            else:
                output = torch.cat((output, model.module.encoder(data.cuda())),dim=0)        

            total_length += len(data[0])

        output2 = output.mean(dim=0)

    output = torch.cat((output1, output2), dim=0)

    return output.cpu().numpy()

if __name__ == '__main__':
    # main()
    tester()
