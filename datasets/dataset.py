from __future__ import print_function

import numpy as np
import torch
from torchvision import datasets
from random import *


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


    def find_new_index(self, index, path):
        class_idx = self.targets[index]

        if (index+6)<len(self.targets) and (index-6)>=0:
            if self.targets[index+6] == class_idx:
                new_index = index+6
            elif self.targets[index-6] == class_idx:
                new_index = index-6
            elif self.targets[index+4] == class_idx:
                new_index = index+4
            elif self.targets[index-4] == class_idx:
                new_index = index-4
            elif self.targets[index+2] == class_idx:
                new_index = index+2
            elif self.targets[index-2] == class_idx:
                new_index = index-2  
            elif self.targets[index+1] == class_idx:
                new_index = index+1
            elif self.targets[index-1] == class_idx:
                new_index = index-1       
            else:
                print("couldn't find a match for " + path)
                new_index = index
        elif (index+4)<len(self.targets) and (index-4)>=0:
            if self.targets[index+4] == class_idx:
                new_index = index+4
            elif self.targets[index-4] == class_idx:
                new_index = index-4
            elif self.targets[index+2] == class_idx:
                new_index = index+2
            elif self.targets[index-2] == class_idx:
                new_index = index-2  
            elif self.targets[index+1] == class_idx:
                new_index = index+1
            elif self.targets[index-1] == class_idx:
                new_index = index-1       
            else:
                print("couldn't find a match for " + path)
                new_index = index   
        elif (index+2)<len(self.targets) and (index-2)>=0:
            if self.targets[index+2] == class_idx:
                new_index = index+2
            elif self.targets[index-2] == class_idx:
                new_index = index-2  
            elif self.targets[index+1] == class_idx:
                new_index = index+1
            elif self.targets[index-1] == class_idx:
                new_index = index-1       
            else:
                print("couldn't find a match for " + path)
                new_index = index  
        elif (index+1)<len(self.targets) or (index-1)>=0: 
            try:
                if self.targets[index+1] == class_idx:
                    new_index = index+1
            except:
                if self.targets[index-1] == class_idx:
                    new_index = index-1       
        else:
            print("couldn't find a match for " + path)
            new_index = index

        return new_index        

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """

        # print(index)
        
        path, target = self.imgs[index]
        # print(path)
        image = self.loader(path)
    
        new_index = self.find_new_index(index, path)                         

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                # print('two_crop')
                path2, target2 = self.imgs[new_index]
                # print(path2)
                image2 = self.loader(path2)
                img2 = self.transform(image2)
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

class BatchImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(BatchImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__() 
        self.class_to_idx_dict = {}  
        self.find_class_to_index()

    def find_class_to_index(self):
        for i in range(len(self.classes)):
            self.class_to_idx_dict[i] = []
        for index in range(len(self.targets)):    
            self.class_to_idx_dict[self.targets[index]].append(index)
        min_length = 10000    
        for key in self.class_to_idx_dict:
            # print(len(self.class_to_idx_dict[key]))
            if min_length>len(self.class_to_idx_dict[key]):
                min_length = len(self.class_to_idx_dict[key])
        # print("min_length: ", str(min_length))       

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """

        # print(index)
        


        # class_idx = index//30

        # temp_idx = randint(0,5)
        # first_segment_idx = true_idx*30 + temp_idx
        # temp_idx = randint(min(temp_idx+5,20), 20)
        # second_segment_idx = true_idx*30 + temp_idx

        path, target = self.imgs[index]

        fb_size = 30
        
        ## check if size of class < 20
        while len(self.class_to_idx_dict[target])<=2*fb_size:
            target = target+1
            if target>=len(self.classes):
                target = 0

        class_idx = target
        temp_idx1 = randint(0,len(self.class_to_idx_dict[class_idx])-fb_size-fb_size//2)
        temp_idx2 = randint(min(temp_idx1+fb_size//2,len(self.class_to_idx_dict[class_idx])-fb_size), len(self.class_to_idx_dict[class_idx])-fb_size)

        first_segment = self.class_to_idx_dict[class_idx][temp_idx1:temp_idx1+fb_size]
        second_segment = self.class_to_idx_dict[class_idx][temp_idx2:temp_idx2+fb_size]

        for i in range(fb_size):
            path, target = self.imgs[first_segment[i]]
            # print(path)
            image = self.loader(path)
            if self.transform is not None:
                img = self.transform(image)
                if self.two_crop:
                    # print('two_crop')
                    path2, target2 = self.imgs[second_segment[i]]
                    # print(path2)
                    image2 = self.loader(path2)
                    img2 = self.transform(image2)
                    current_img = torch.cat([img, img2], dim=0).unsqueeze(0)
                    if i==0:
                        final_image = current_img
                    else:
                        final_image = torch.cat((final_image, current_img),0)
                    # print(final_image.shape)

        img = final_image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, index, jigsaw_image
        else:
            return img, first_segment[0]


class VideoFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(VideoFolderInstance, self).__init__(root, transform, target_transform)
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
                # print('two_crop')
                if np.random.uniform()<0.5:
                    # img2 = self.transform(image)
                    # print('i flipped')
                    img2 = torch.fliplr(img)
                else:
                    img2 = img
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
