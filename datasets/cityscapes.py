import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np


class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('background', 0, 0, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('road', 1, 1, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('potholes', 2, 2, 'flat', 1, True, False, (165, 42, 42)),
        CityscapesClass('shoulder', 3, 255, 'flat', 1, False, True, (244, 35, 232)),
        CityscapesClass('vegetation', 4, 255, 'nature', 2, False, True, (0, 128, 0)),
        CityscapesClass('building', 5, 255, 'construction', 3, True, True, (255, 255, 0)),
        CityscapesClass('sky', 6, 255, 'sky', 4, False, True, (0, 0, 255)),
        CityscapesClass('animal', 7, 255, 'animal', 5, True, True, (220, 20, 60)),
        CityscapesClass('person', 8, 255, 'human', 6, True, True, (220, 20, 60)),
        CityscapesClass('vehicle', 9, 255, 'vehicle', 7, True, True, (255, 128, 64)),
        CityscapesClass('water body', 10, 255, 'void', 0, False, True, (51, 102, 255)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        # target[target == 255] = 19
        target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]


    def __getitem__(self, index):
        try:
            image = Image.open(self.images[index]).convert('RGB')
        except (Image.UnidentifiedImageError, OSError):
            # Handle UnidentifiedImageError or OSError (skip the problematic image)
            print(f"Skipping UnidentifiedImageError at index {index}")
            return self.__getitem__((index + 1) % len(self))

        target_folder = os.path.dirname(self.images[index])
        
        # Assign class labels based on folder structure
        target_class = 0 if 'city0' in target_folder else 1

        # Resize the image to a consistent size
        resize_transform = transforms.Resize((512, 384))
        image = resize_transform(image)

        # Convert image to tensor
        image = transforms.ToTensor()(image)

        # Convert target to tensor (0 or 1)
        target = torch.tensor([target_class], dtype=torch.float32)
        
        # Print image name and label assigned
        # print("Image Name:", os.path.basename(self.images[index]))
        # print("Assigned Label:", target_class)
        
        return image, target


    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds_binarylabelid.png'.format(mode)  # Assuming binary labels are saved as binarylabelid.png
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)