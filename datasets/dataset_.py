from __future__ import print_function

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils import data
import pandas as pd
import datasets.iqa_distortions as iqa
import random
import os
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.imgs[index]
        image = self.loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, index, jigsaw_image
        else:
            return img, index


class IQAImageClass(data.Dataset):

    def __init__(self, root = './', n_aug = 5):

        super().__init__()

        self.root = os.path.join(root, 'train')
        self.data = os.listdir(self.root)
        # print('root:' ,self.root)
        self.root_extra_data = os.path.join(root, 'train_under30')
        # print('extra_root:' ,self.root_extra_data)
        self.extra_data = os.listdir(self.root_extra_data)
        self.n_aug = n_aug
        self.crop_transform()

    def __len__(self):

        return len(self.data) + len(self.extra_data)

    def iqa_transformations(self, choice, im):

        level = random.randint(0,4)

        if choice == 1:

            im = iqa.imblurgauss(im, level)
 
        elif choice == 2 :
        
            im = iqa.imblurlens(im,level)
        
        elif choice == 3 :

            im = iqa.imblurlens(im,level)

        elif choice == 4 :
            # size = im.size
            # amount = 2
            # w = size[0]
            # h = size[1]

            # scaled_w = int(w/amount)
            # scaled_h = int(h/amount)
            # resized_image = im.resize((scaled_w,scaled_h))
            im = iqa.imcolordiffuse(im,level)
            # im = im.resize((w,h))

        elif choice == 5 :

            im = iqa.imcolorshift(im,level)

        elif choice == 6 :
            # size = im.size
            # amount = 2
            # w = size[0]
            # h = size[1]

            # scaled_w = int(w/amount)
            # scaled_h = int(h/amount)
            # resized_image = im.resize((scaled_w,scaled_h))
            im = iqa.imcolorsaturate(im,level)
            # im = im.resize((w,h))

        elif choice == 7 :
            # size = im.size
            # amount = 2
            # w = size[0]
            # h = size[1]

            # scaled_w = int(w/amount)
            # scaled_h = int(h/amount)
            # resized_image = im.resize((scaled_w,scaled_h))
            im = iqa.imsaturate(im,level)
            # im = im.resize((w,h))

        elif choice == 8 :

            im = iqa.imcompressjpeg(im,level)

        elif choice == 9 :

            im = iqa.imnoisegauss(im,level)

        elif choice == 10 :

            im = iqa.imnoisecolormap(im,level)

        elif choice == 11 :

            im = iqa.imnoiseimpulse(im,level)

        elif choice == 12 :

            im = iqa.imnoisemultiplicative(im,level)

        elif choice == 13 :

            im = iqa.imdenoise(im,level)

        elif choice == 14 :
            # size = im.size
            # amount = 2
            # w = size[0]
            # h = size[1]

            # scaled_w = int(w/amount)
            # scaled_h = int(h/amount)
            # resized_image = im.resize((scaled_w,scaled_h))
            im = iqa.imbrighten(im,level)
            # im = im.resize((w,h))

        elif choice == 15 :

            im = iqa.imdarken(im, level)

        elif choice == 16 :

            im = iqa.immeanshift(im,level)

        elif choice == 17 :

            im = iqa.imresizedist(im,level)

        elif choice == 18 :

            im = iqa.imsharpenHi(im,level)

        elif choice == 19 :

            im = iqa.imcontrastc(im,level)

        elif choice == 20 :

            im = iqa.imcolorblock(im,level)

        elif choice == 21 :

            im = iqa.impixelate(im,level)

        elif choice == 22 :

            im = iqa.imnoneccentricity(im,level)

        elif choice == 23 :
            # size = im.size
            # amount = 2
            # w = size[0]
            # h = size[1]

            # scaled_w = int(w/amount)
            # scaled_h = int(h/amount)
            # resized_image = im.resize((scaled_w,scaled_h))
            im = iqa.imjitter(im,level)
            # im = im.resize((w,h))
        else :
            
            pass

        return im

    def crop_transform(self, crop_size=300, crop_type='random'):
        if crop_type == 'center':
            self.transform_crop = transforms.transforms.CenterCrop(crop_size)
        elif crop_type == 'random':
            self.transform_crop = transforms.transforms.RandomCrop(crop_size)

    def __getitem__(self, idx) :

        if idx >= len(self.data):
            idx = idx-len(self.data)
            path = self.extra_data[idx]
            video_path = os.path.join(self.root_extra_data, path)
        else:
            path = self.data[idx]
            video_path = os.path.join(self.root, path)

        filelist = os.listdir(video_path)
        filelist.sort()

        ## choose single image from 1 video
        indx = random.randint(0,len(filelist)-1)
        
        ## load image
        image_path = os.path.join(video_path, filelist[indx])
        image = Image.open(image_path).convert('RGB')

        ## create  positive pair
        img_pair1 = transforms.ToTensor()(image)  # 1, 3, H, W
        chunk1 = img_pair1.unsqueeze(0)

        img_pair2 = transforms.ToTensor()(image)  # 1, 3, H, W
        chunk2 = img_pair2.unsqueeze(0)

        choices = list(range(1, 24))
        random.shuffle(choices)

        for i in range(0,self.n_aug):
            ## generate self.aug distortion-augmentations
            img_aug_i = transforms.ToTensor()(self.iqa_transformations(choices[i], image))
            img_aug_i = img_aug_i.unsqueeze(0)
            chunk1 = torch.cat([chunk1, img_aug_i], dim=0)
            chunk2 = torch.cat([chunk2, img_aug_i], dim=0)

        # chunk1, chunk2  -> self.n_aug+1 , 3, H, W

        # generate two random crops
        chunk1 = self.transform_crop(chunk1)
        chunk2 = self.transform_crop(chunk2)

        #chunk1, chunk2  -> self.n_aug+1 , 3, 256 , 256

        temp = chunk1[0]
        chunk1[0] = chunk2[0]
        chunk2[0] = temp

        return torch.cat((chunk1, chunk2), dim=1)
    
    def __time__(self):
        image = Image.open('../../../codebase_python_image_operations/out-001.png').convert('RGB')
        import time
        end = time.time()
        for choice in range(1,24):
            for reps in range(0,5):
                im_out = self.iqa_transformations(choice, image)
            print("Time for choice " + str(choice) + ":" + str(time.time()-end))
            end = time.time()

# object = IQAImageClass()
# object.__time__()