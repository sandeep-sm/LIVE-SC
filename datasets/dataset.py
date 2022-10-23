from __future__ import print_function

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import random as RND
from random import *
import os
import csv
from torch.utils.data import Dataset
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
import time
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage
import warnings
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

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

def compute_MS_transform(image, window, extend_mode='reflect'):
    h,w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image

def MS_transform(image):
#   MS Transform
    image = np.array(image).astype(np.float32)
    window = gen_gauss_window(3, 7/6)
    image[:,:,0] = compute_MS_transform(image[:,:,0], window)
    image[:,:,0] = (image[:,:,0] - np.min(image[:,:,0]))/(np.ptp(image[:,:,0])+1e-3)
    image[:,:,1] = compute_MS_transform(image[:,:,1], window)
    image[:,:,1] = (image[:,:,1] - np.min(image[:,:,1]))/(np.ptp(image[:,:,1])+1e-3)
    image[:,:,2] = compute_MS_transform(image[:,:,2], window)
    image[:,:,2] = (image[:,:,2] - np.min(image[:,:,2]))/(np.ptp(image[:,:,2])+1e-3)
    
    image = Image.fromarray((image*255).astype(np.uint8))
    return image

def gray_transform(image, rand2):
    if rand2>0.75:
        image = transforms.RandomGrayscale(p=1)(image)
    return image

class VideoFolderInstance(Dataset):
    def __init__(self, root):
        super().__init__()

        self.root = root
        self.data = os.listdir(root)
        self.extra_data = os.listdir('/work/08804/smishra/ls6/PNG_training_backup/train/')
        self.n_aug = 12    ## chunk size
        self.crop_transform()
        loader: Callable[[str], Any] = default_loader
        self.loader = loader
        self.min_side = 512.0
        self.resnet_transform = torchvision.transforms.Normalize(
                                                        mean = [0.5204, 0.4527, 0.4395],
                                                        std = [0.2828, 0.2745, 0.2687])
    
    def crop_transform(self, crop_size=400, crop_type='random'):
        if crop_type == 'center':
            self.transform_crop = transforms.transforms.CenterCrop(crop_size)
        elif crop_type == 'random':
            self.transform_crop = transforms.transforms.RandomCrop(crop_size)

    def __len__(self):
        return len(self.data) + len(self.extra_data)  

    def apply_distortion_transform(self, choice, image):
        ### POSTERIZE
        if choice == 1:
            rand_bits = randint(4,6)
            posterize = torchvision.transforms.functional.posterize
            image = posterize(image, rand_bits)

        ### SOLARIZE
        elif choice == 2:
            threshold = 255*RND.uniform(0, 1)
            solarize = torchvision.transforms.functional.solarize
            image = solarize(image, threshold)

        ### increase SHARPness
        elif choice == 3:
            sharpness_factor = 3*RND.uniform(0.4, 1)   
            sharpen = torchvision.transforms.functional.adjust_sharpness
            image = sharpen(image, sharpness_factor)

        ### reduce SHARPness
        elif choice == 4:
            unsharpness_factor = RND.uniform(0, 0.8)
            unsharpen = torchvision.transforms.functional.adjust_sharpness
            image = unsharpen(image, unsharpness_factor)

        ### increase CONTRAST
        elif choice == 5:
            contrast_factor = 3*RND.uniform(0.4, 1)   
            contrast = torchvision.transforms.functional.adjust_contrast
            image = contrast(image, contrast_factor)

        ### reduce CONTRAST
        elif choice == 6:
            contrast_factor = RND.uniform(0.2, 0.8)
            contrast = torchvision.transforms.functional.adjust_contrast
            image = contrast(image, contrast_factor)

        ### change HUE
        elif choice == 7:
            hue_factor = RND.uniform(0, 1)-0.5   ## range [-0.5, 0.5]
            hue = torchvision.transforms.functional.adjust_hue
            image = hue(image, hue_factor)

        ### increase SATURATION
        elif choice == 8:
            saturation_factor = 3*RND.uniform(0.4, 1)     
            saturate = torchvision.transforms.functional.adjust_saturation
            image = saturate(image, saturation_factor)

        ### reduce SATURATION
        elif choice == 9:
            saturation_factor = RND.uniform(0.2, 0.8)   
            saturate = torchvision.transforms.functional.adjust_saturation
            image = saturate(image, saturation_factor) 

        ### increase BRIGHTNESS
        elif choice == 10:
            brightness_factor = 3*RND.uniform(0.4, 1)   
            adjust_brightness = torchvision.transforms.functional.adjust_brightness
            image = adjust_brightness(image, brightness_factor)

        ### reduce BRIGHTNESS
        elif choice == 11:
            brightness_factor = RND.uniform(0.2, 0.8)   
            adjust_brightness = torchvision.transforms.functional.adjust_brightness
            image = adjust_brightness(image, brightness_factor)                      

        ### increase GAMMA
        elif choice == 12:
            gamma = 3*RND.uniform(0.4, 1)   
            adjust_gamma = torchvision.transforms.functional.adjust_gamma
            image = adjust_gamma(image, gamma)

        ### reduce GAMMA
        elif choice == 13:
            gamma = RND.uniform(0.2, 0.8)  
            adjust_gamma = torchvision.transforms.functional.adjust_gamma
            image = adjust_gamma(image, gamma)

        ### apply GaussianBLUR
        elif choice == 14:
            kernel_size = 1+2*(randint(1,4))     ## range : [3, 9]
            sigma = RND.uniform(0.5, 2.0)  
            gaussian_blur = torchvision.transforms.functional.gaussian_blur
            image = gaussian_blur(image, kernel_size, sigma)
    
        return image

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        if index >= len(self.data):
            index = index-len(self.data)
            path = self.extra_data[index]
            video_path = os.path.join('/work/08804/smishra/ls6/PNG_training_backup/train/', path)
        else:
            path = self.data[index]
            video_path = os.path.join(self.root, path)

        # print("This batch index = ", index)
        # video_path = os.path.join(os.path.join(self.root, path), path)
        
        filelist = os.listdir(video_path)
        filelist.sort()

        ## choose single image from 1 video
        indx = randint(0,len(filelist)-1)
        
        ## load image
        image_path = os.path.join(video_path, filelist[indx])
        image = self.loader(image_path)

        ## transform to tensor
        # image = transforms.ToTensor()(image)

        ## resize min_side to self.min_side
        this_frame_min_side = min(image.size[0], image.size[1])
        ratio = self.min_side/this_frame_min_side
        new_1st_dim = int(ratio*image.size[0])
        new_2nd_dim = int(ratio*image.size[1])
        resize_transform = transforms.Resize([new_2nd_dim, new_1st_dim])
        image = resize_transform(image)

        ## concate positive pair
        current_img_pair1 = transforms.ToTensor()(image)  # 1, 6, H, W
        chunk1 = current_img_pair1.unsqueeze(0)

        current_img_pair2 = transforms.ToTensor()(image)  # 1, 6, H, W
        chunk2 = current_img_pair2.unsqueeze(0)

        choices = list(range(3, 15))
        RND.shuffle(choices)

        for i in range(self.n_aug):
            ## generate distortion-augmentations
            img_aug_i = transforms.ToTensor()(self.apply_distortion_transform(choices[i], image))
            img_aug_i = img_aug_i.unsqueeze(0)
            chunk1 = torch.cat([chunk1, img_aug_i], dim=0)
            chunk2 = torch.cat([chunk2, img_aug_i], dim=0)

        ## generate two random crops
        chunk1 = self.transform_crop(self.resnet_transform(chunk1))
        chunk2 = self.transform_crop(self.resnet_transform(chunk2))

        temp = chunk1[0]
        chunk1[0] = chunk2[0]
        chunk2[0] = temp

        final_image = torch.cat((chunk1, chunk2), dim=1)

        # from torchvision.utils import save_image
        # for i in range(15):
        #     img = chunk1[i]
        #     save_image(img, './temp/img'+str(i)+'.png')

        # import pdb;pdb.set_trace()

        ## reshape final_image : 15, 6, H, W -> 6*15, H, W
        # shape = final_image.shape
        # final_image = final_image.reshape(shape[0]*shape[1], shape[2], shape[3])
        return final_image

def test_dataloader_VideoFolderInstance():
    loader = VideoFolderInstance(root='/scratch/08804/smishra/PNG_training_key_Frames/train')
    # loader = VideoFolderInstance2(root='/work/08804/smishra/ls6/LIVE-ShareChat_Data/train', transform=train_transform, crop_size=512, crop_type='center', two_crop=False, target_transform=None)
    train_loader = torch.utils.data.DataLoader(
        loader, batch_size=3, shuffle=True,
        num_workers=0, pin_memory=True)

    end_time = time.time()
    for idx, data in enumerate(train_loader):
        current_time = time.time()
        import pdb;pdb.set_trace()

        print("This batch time: " + str(current_time-end_time))
        # if idx == 0:
        #     out = data[0]
        # else:
        #     out = torch.cat((out, data[0]), dim=1)
        end_time = time.time()

    
    
# test_dataloader_VideoFolderInstance()

class test_datasetSH(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, use_scale=0):
        super(test_datasetSH, self).__init__(root)
        self.num = self.__len__()
        self.min_side=512
        self.crop_transform(crop_size=512, crop_type='center')
        self.use_scale = use_scale
        self.resnet_transform = torchvision.transforms.Normalize(
                                                        mean = [0.5204, 0.4527, 0.4395],
                                                        std = [0.2828, 0.2745, 0.2687])
    

    def crop_transform(self, crop_size='224', crop_type='center'):
        if crop_type == 'center':
            self.transform_crop = transforms.transforms.CenterCrop(crop_size)
        elif crop_type == 'random':
            self.transform_crop = transforms.transforms.RandomCrop(crop_size)

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = torchvision.transforms.ToTensor()(img)

        if self.use_scale == 1:
            this_frame_min_side = min(img.shape[1], img.shape[2])
            ratio = self.min_side/this_frame_min_side
            new_1st_dim = int(ratio*img.shape[1])
            new_2nd_dim = int(ratio*img.shape[2])
            resize_transform = transforms.Resize([new_1st_dim, new_2nd_dim], interpolation=Image.BICUBIC)
            img = resize_transform(img)

        # img = self.resnet_transform(img)

        return img