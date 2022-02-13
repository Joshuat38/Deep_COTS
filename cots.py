# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import random

# These imports are primarily so that we can run an iteration.
import argparse
import sys
import re
import yaml
import time

class COTSDataLoader:
    def __init__(self, args, cfg, mode='train', verbose=0):
        
        if mode == 'train':
            self.samples = COTSPreprocess(args=args, cfg=cfg, mode=mode, 
                                          verbose=verbose)

            self.train_sampler = None
    
            self.data = DataLoader(self.samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True, sampler=self.train_sampler)

        elif mode == 'valid':
            self.samples = COTSPreprocess(args=args, cfg=cfg, mode=mode, 
                                          verbose=verbose)

            self.eval_sampler = None
                
            self.data = DataLoader(self.samples, args.valid_batch_size,
                                   shuffle=False, num_workers=args.num_threads,
                                   pin_memory=True, sampler=self.eval_sampler)
        
        elif mode == 'test' or mode == 'analyse':
            self.samples = COTSPreprocess(args=args, cfg=cfg, mode=mode, 
                                          verbose=verbose)
            
            self.data = DataLoader(self.samples, args.batch_size, 
                                   shuffle=False, num_workers=args.num_threads, 
                                   pin_memory=True)

        else:
            print('mode should be one of \'train, valid, test\'. Got {}'.format(mode))

class COTSPreprocess(Dataset):

    def __init__(self, args, cfg, mode='train', verbose=0):

        self.cfg = cfg
        self.mode = mode
        self.verbose = verbose
        
        self.max_boxes = cfg['model']['max_boxes']
        self.strides = cfg['model']['strides']
        self.num_classes = cfg['model']['num_classes']

        self.image_size = cfg['train']['base_size'] if mode == 'train' else cfg['valid']['base_size']
        self.process_size = cfg['train']['process_size'] if mode == 'train' else cfg['valid']['process_size']
        self.batch_size = cfg['train']['batch_size'] if mode == 'train' else cfg['valid']['batch_size']

        self.mosaic = cfg['train']['mosaic'] if mode == 'train' else False
        self.mixup = cfg['train']['mixup'] if mode == 'train' else False
        self.label_smoothing = cfg['train']['label_smoothing'] if mode == 'train' else False
        self.degrees = cfg['train']['degrees'] # rotate angle
        self.translate = cfg['train']['translate']
        self.scale = cfg['train']['scale']
        self.shear = cfg['train']['shear']
        self.perspective = cfg['train']['perspective']
        self.mixup_scale = (0.5, 1.5)
        
        self.inputs_path = cfg['train']['inputs_path'] if mode == 'train' else cfg['valid']['inputs_path']
        self.labels_path = cfg['train']['labels_path'] if mode == 'train' else cfg['valid']['labels_path']
        self.data_file = cfg['train']['data_file'] if mode == 'train' else cfg['valid']['data_file'] 
        
        self.current_epoch = 0
        self.disable_mosaic_epoch = args.num_epochs - cfg['train']['disable_mosaic_epochs']
        
        if self.verbose == 1 and mode == 'train':
            print(f'\nMosaic and mixup agumentations will be disabled after {self.disable_mosaic_epoch} epochs.')

        # Load the file lines so that we can load the data in the __getitem__ method.
        filenames = []
        with open(self.data_file, 'r') as f: # Open the data file and store as lines for reading.
            filenames.extend(f.readlines())
        self.filenames = filenames
        
        # Get the appropriate transforms function so that if can be called in the train/valid/test loops.
        if mode == 'train':
            self.transforms = self.transform_training()
            self.mosaic_transforms = self.transform_validating()
        elif mode == 'valid':
            self.transforms = self.transform_validating()
        elif mode == 'test' or mode == 'analyse':
            self.transforms = self.transform_testing()
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
            
        if self.mosaic and random.random() < 0.5 and self.current_epoch < self.disable_mosaic_epoch:
            mosaic_labels = []
            input_h, input_w = self.process_size

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices to make a 4 image mosaic.
            indices = [idx] + [random.randint(0, len(self.filenames)-1) for _ in range(3)]

            for i, index in enumerate(indices):
                image, targets, image_path, annotations_path = self.get_data(index)
                h0, w0 = image.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                image = cv2.resize(image, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                
                # Generate output mosaic image
                (h, w, c) = image.shape[:3]
                if i == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 127, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), \
                (s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coordinate(mosaic_img, i, xc, yc, 
                                                                      w, h, input_h, input_w)

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = image[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = targets.copy()
                # Normalized xywh to pixel xyxy format
                if targets.size > 0:
                    labels[:, 0] = scale * targets[:, 0] + padw
                    labels[:, 1] = scale * targets[:, 1] + padh
                    labels[:, 2] = scale * targets[:, 2] + padw
                    labels[:, 3] = scale * targets[:, 3] + padh
                    mosaic_labels.append(labels)
                 
            if len(mosaic_labels) > 0:
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
                
            mosaic_labels = np.asarray(mosaic_labels).astype(np.float32) # Convert into a numpy array so that we can have consistancy.
            mosaic_img, mosaic_labels = random_perspective(mosaic_img, mosaic_labels,
                                                           degrees=self.degrees,
                                                           translate=self.translate,
                                                           scale=self.scale,
                                                           shear=self.shear,
                                                           perspective=self.perspective,
                                                           border=[-input_h // 2, -input_w // 2])  # border to remove

            if self.mixup and len(mosaic_labels) > 0 and random.random() < 0.5:
                mosaic_img, mosaic_labels = self.mixup_augment(mosaic_img, mosaic_labels, self.process_size)
            # img_info = (mosaic_img.shape[1], mosaic_img.shape[0])
            # mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.samples_shapes[idx])
            # return mix_img, padded_labels, img_info, -1
        
            sample = self.mosaic_transforms({'image': mosaic_img, 'annotations': mosaic_labels})
        
        else:
            image, targets, image_path, annotations_path = self.get_data(idx)

            sample = self.transforms({'image': image, 'annotations': targets})
        
        # Separate the inputs and outputs so that we can use the my universal training loop.
        if self.mode == 'analyse':
            inputs = {'image': sample['image'], 'image_path' : image_path}
            outputs = {'annotations': sample['annotations'], 
                       'annotations_path' : annotations_path} 
        else:
            inputs = {'image': sample['image']}
            outputs = {'annotations': sample['annotations']} 
        
        return inputs, outputs
    
    # def random_resize(self, data_loader, epoch, rank, is_distributed):
    #      tensor = torch.LongTensor(2).cuda()
    
    #      if rank == 0:
    #          size_factor = self.input_size[1] * 1.0 / self.input_size[0]
    #          if not hasattr(self, 'random_size'):
    #              min_size = int(self.input_size[0] / 32) - self.multiscale_range
    #              max_size = int(self.input_size[0] / 32) + self.multiscale_range
    #              self.random_size = (min_size, max_size)
    #          size = random.randint(*self.random_size)
    #          size = (int(32 * size), 32 * int(size * size_factor))
    #          tensor[0] = size[0]
    #          tensor[1] = size[1]
    
    #      if is_distributed:
    #          dist.barrier()
    #          dist.broadcast(tensor, 0)
    
    #      input_size = (tensor[0].item(), tensor[1].item())
    #      return input_size
    
    def update_epoch(self, current_epoch):
        self.current_epoch = current_epoch
        
        if self.verbose == 1:
            print(f'\nCOTS dataset registered current epoch as: {self.current_epoch+1}.')
            
            if self.current_epoch >= self.disable_mosaic_epoch:
                print('Mosaic and mixup augmentations have been disabled!')
    
    def get_data(self, idx):
        sample_path = self.filenames[idx]
        
        image_path = self.inputs_path+sample_path.split()[0]
        annotations_path = self.labels_path+sample_path.split()[1]

        # Load the image in RGB format so that we can feed it into the model.
        image = cv2.cvtColor(cv2.imread(image_path, 1), cv2.COLOR_BGR2RGB) # Must be in RGB form.
        
        # Open the text file holding the annotations for this particular image.
        annotations = []
        with open(annotations_path, 'r') as f: # Open the file for the annotations and read in the lines.
            annotations.extend(f.readlines())
            
        # Process all the annotations so that they are all in 
        targets = []
        for annotation in annotations:
            annotation_data = [float(ann) for ann in annotation.split()] # Expected form is x1, y1, x2, y2, class_label
            targets.append(annotation_data)

        # Turn these into numpy arrays for processing.
        if len(targets) != 0:
            targets = np.array(targets)
        else:
            targets = np.zeros((0, 5))
            
        return image, targets, image_path, annotations_path
    
    def transform_training(self):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4),
            # RandomDistort(alpha=1, beta=0),
            # RandomRotate(degrees=self.cfg['train']['degrees']),
            # RandomCropZoom(size=self.process_size, jitter=0.1),
            Preprocess(size=self.process_size, max_boxes=self.max_boxes),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                      rescale=255.0),
            ToTensor()])

        return composed_transforms

    def transform_validating(self):

        composed_transforms = transforms.Compose([
            Preprocess(size=self.process_size, max_boxes=self.max_boxes),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                      rescale=255.0),
            ToTensor()])

        return composed_transforms

    def transform_testing(self):

        composed_transforms = transforms.Compose([
            Preprocess(size=self.process_size, max_boxes=self.max_boxes),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                      rescale=255.0),
            ToTensor()])

        return composed_transforms
    
    def get_mosaic_coordinate(self, mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
        # TODO update doc
        # index0 to top left part of image
        if mosaic_index == 0:
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coord = w - (x2 - x1), h - (y2 - y1), w, h
        # index1 to top right part of image
        elif mosaic_index == 1:
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
            small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
            small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
            small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
        return (x1, y1, x2, y2), small_coord
    
    def mixup_augment(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        flip = random.uniform(0, 1) > 0.5
        cp_index = random.randint(0, self.__len__() - 1)
        # img, cp_labels, _, _ = self.pull_item(cp_index)
        img, cp_labels, _, _ = self.get_data(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * cp_scale_ratio), 
                                       int(img.shape[0] * cp_scale_ratio)), 
                                 interpolation=cv2.INTER_LINEAR)
        cp_img[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)] = resized_img
        cp_img = cv2.resize(cp_img, (int(cp_img.shape[1] * jit_factor), 
                                     int(cp_img.shape[0] * jit_factor)))
        cp_scale_ratio *= jit_factor
        if flip:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), 
                               max(origin_w, target_w), 3), dtype=np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset: y_offset + target_h, 
                                        x_offset: x_offset + target_w]

        if cp_labels.size == 0: # If there are no labels, blend the image anyway.
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
        else:
            cp_bboxes_origin_np = self.adjust_box_anns(cp_labels[:, :4].copy(), 
                                                       cp_scale_ratio, 0, 0, 
                                                       origin_w, origin_h)
            
            if flip:
                cp_bboxes_origin_np[:, 0::2] = (origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1])
                
            cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
            cp_bboxes_transformed_np[:, 0::2] = np.clip(cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w)
            cp_bboxes_transformed_np[:, 1::2] = np.clip(cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h)
            # print('Original bboxes: ', cp_bboxes_origin_np.T, '\n\nNew bboxes: ', cp_bboxes_transformed_np.T)
            keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)
            # print('\nThis is the keep list: ', keep_list)
    
            if keep_list.sum() >= 1.0:
                cls_labels = cp_labels[keep_list, 4:5].copy()
                box_labels = cp_bboxes_transformed_np[keep_list]
                labels = np.hstack((box_labels, cls_labels))
                origin_labels = np.vstack((origin_labels, labels))
                origin_img = origin_img.astype(np.float32)
                origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
    
    def adjust_box_anns(self, bbox, scale_ratio, padw, padh, w_max, h_max):
        bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
        return bbox
    
class RandomHorizontalFlip(object):
    
    def __call__(self, sample):
        
        image = sample['image']
        targets = sample['annotations']
        
        if random.random() < 0.5:
            _, width, _ = image.shape
            image = np.fliplr(image) # Or image = image[:, ::-1, :]
            if targets.size > 0:
                targets = targets.copy()
                targets[:, [0, 2]] = width - targets[:, [2, 0]] # Switch the positions and subtract the width.

        return {'image': image, 'annotations': targets}

class RandomHSV(object):
    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4):
        
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
    
    def __call__(self, sample):
        
        image = sample['image']
        targets = sample['annotations']
        
        if random.random() < 0.5:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
            dtype = image.dtype  # uint8
    
            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    
            image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), 
                                   cv2.LUT(val, lut_val))).astype(dtype)
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)  # no return needed

        return {'image': image, 'annotations': targets}  
 
class RandomDistort(object):
    def __init__(self, alpha=1, beta=0):
        
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, sample):
        
        image = sample['image']
        targets = sample['annotations']
        
        if random.random() < 0.5:
        
            image = image.copy()
    
            # All colour channels shift augmentation.
            if random.random() < 0.5:
                image = self.apply(image, beta=random.uniform(-32, 32))
    
            # Brightness augmentation
            if random.random() < 0.5:
                image = self.apply(image, alpha=random.uniform(0.5, 1.5))
    
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
            # Hue shift augmentation (see HSV format for colour changes).
            if random.random() < 0.5:
                tmp = image[:, :, 0].astype(np.int64) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp
    
            # Saturation shift augmentation (see HSV format for colour changes).
            if random.random() < 0.5:
                image[:, :, 1] = self.apply(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
    
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        
        return {'image': image, 'annotations': targets} 
        
    def apply(self, image, alpha=1, beta=0):
        tmp = image.astype(np.float32) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp
        
        return image
    
class RandomRotate(object):
    def __init__(self, degrees=7):
        
        self.degrees = degrees
    
    def __call__(self, sample):
        
        image = sample['image']
        targets = sample['annotations']
        
        if random.random() < 0.5:
            angle = np.random.uniform(-self.degrees, self.degrees)

            h, w, _ = image.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, m, (w, h), borderValue=(127, 127, 127))

            if targets.size > 0:
                top_left = targets[..., [0, 1]] # x1, y1
                top_right = targets[..., [2, 1]] # x2, y1
                bottom_left = targets[..., [0, 3]] # x1, y2
                bottom_right = targets[..., [2, 3]] # x2, y2

                # N, 4, 2
                points = np.stack([top_left, top_right, bottom_left, bottom_right], axis=-2)
                points_3d = np.ones(points.shape[:-1] + (3,), np.float32)
                points_3d[..., :2] = points

                # points = m @ points_3d[0].T
                points = map(lambda x: m @ x.T, points_3d)
                points = np.array(list(points))
                points = np.transpose(points, [0, 2, 1])

                targets[..., 0] = np.min(points[..., 0], axis=-1)
                targets[..., 1] = np.min(points[..., 1], axis=-1)
                targets[..., 2] = np.max(points[..., 0], axis=-1)
                targets[..., 3] = np.max(points[..., 1], axis=-1)

                targets[:, [0, 2]] = np.clip(targets[:, [0, 2]], 0, w)
                targets[:, [1, 3]] = np.clip(targets[:, [1, 3]], 0, h)

        return {'image': image, 'annotations': targets} 
    
class RandomCropZoom(object):
    def __init__(self, size, jitter=0.3):
        
        self.size = size
        self.jitter = jitter
    
    def __call__(self, sample):
        
        image = sample['image']
        targets = sample['annotations']
        
        if random.random() < 0.5:
            net_h, net_w = self.size
            h, w, _ = image.shape
            dw = w * self.jitter
            dh = h * self.jitter

            rate = (w + np.random.uniform(-dw, dw)) / (h + np.random.uniform(-dh, dh))
            scale = np.random.uniform(1/1.5, 1.5)

            if (rate < 1):
                new_h = int(scale * net_h)
                new_w = int(new_h * rate)
            else:
                new_w = int(scale * net_w)
                new_h = int(new_w / rate)

            dx = int(np.random.uniform(0, net_w - new_w))
            dy = int(np.random.uniform(0, net_h - new_h))

            M = np.array([[new_w / w, 0., dx],
                          [0., new_h / h, dy]], dtype=np.float32)
            image = cv2.warpAffine(image, M, tuple(self.size), 
                                   borderValue=(127, 127, 127))

            if targets.size > 0:
                targets[:, [0, 2]] = targets[:, [0, 2]] * new_w / w + dx # Scale x1 and x2 by new shape for bbox consistancy.
                targets[:, [1, 3]] = targets[:, [1, 3]] * new_h / h + dy # Scale y1 and y2 by new shape for bbox consistancy.
    
                targets[:, [0, 2]] = np.clip(targets[:, [0, 2]], 0, net_w)
                targets[:, [1, 3]] = np.clip(targets[:, [1, 3]], 0, net_h)
    
                filter_b = np.logical_or(targets[:, 0] >= targets[:, 2], targets[:, 1] >= targets[:, 3])
                targets = targets[~filter_b]

        return {'image': image, 'annotations': targets} 

class Normalize(object):
    
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.), 
                 rescale=255.0):
        self.mean = mean
        self.std = std
        self.rescale = rescale

    def __call__(self, sample):
        image = sample['image']
        targets = sample['annotations']
        
        image = image.astype(np.float32)
        targets = targets.astype(np.float32)
        image /= self.rescale
        image -= self.mean
        image /= self.std
        
        return {'image': image, 'annotations': targets} 
    
class Preprocess(object):
    
    def __init__(self, size, max_boxes):
        
        self.size = size
        self.max_boxes = max_boxes
    
    def __call__(self, sample):
        """
        :param image: RGB, uint8
        :param size:
        :param bboxes:
        :return: RGB, uint8
        """
        
        image = sample['image']
        targets = sample['annotations']
        
        iw, ih = self.size
        h, w, _ = image.shape
    
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
    
        image_padded = np.full(shape=[ih, iw, 3], dtype=np.uint8, fill_value=127)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
    
        # Convert all the targets to the desired format and padd them with zeros where approapriate.
        if targets.size > 0:
            targets = np.asarray(targets).astype(np.float32)
            # This assumes x1y1x2y2 format. Mine are in xywh meaning I do not need to add dw or dh
            targets[:, [0, 2]] = targets[:, [0, 2]] * scale + dw
            targets[:, [1, 3]] = targets[:, [1, 3]] * scale + dh
            
            targets_padded = np.zeros((self.max_boxes, 5))
            targets_padded[:len(targets), :] = targets
            targets_padded = targets_padded.astype(np.float32)
        else:
            targets_padded = np.zeros((self.max_boxes, 5), dtype=np.float32)
            
        targets_padded = self.xyxy2cxcywh(targets_padded) # Convert boxes to YOLOX format.
    
        return {'image': image_padded, 'annotations': targets_padded}
    

    def xyxy2cxcywh(self, targets):
        bboxes = targets[:, 0:-1]
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] # w
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] # h
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5 # xc
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5 # yc
        targets[:, 0:-1] = bboxes
        return targets
    
class ToTensor:
    
    """ This converts the image data to a Pytorch tensor for better 
    processing. Seems to be a common fixture in pytorch applocations. """
    
    def __call__(self, sample):
        
        image = sample['image']
        targets = sample['annotations']
        
        image = self.to_tensor(image)
        targets = self.to_tensor(targets)

        return {'image': image, 'annotations': targets} 
    
    def to_tensor(self, pic):
        
        if not (self._is_pil_image(pic) or self._is_numpy_image(pic)):
            return pic
            # raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            if len(pic.shape) == 3:
                img = torch.from_numpy(pic.transpose((2, 0, 1)))
                
            else:
                img = torch.from_numpy(pic)
                
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
    
    def _is_pil_image(self, img):
        return isinstance(img, Image.Image)
    
    def _is_numpy_image(self, img):
        return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
    

def random_perspective(img, targets=(), degrees=10, translate=0.1, scale=0.1, 
                       shear=10, perspective=0.0, border=(0, 0)):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = np.tan(random.uniform(-shear, shear) * np.pi / 180)  # x shear (deg)
    S[1, 0] = np.tan(random.uniform(-shear, shear) * np.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
    T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n > 0:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return ((w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr))  # candidates

    
def vis_inputs(image, annotations, mean=(0.485, 0.456, 0.406), 
               std=(0.229, 0.224, 0.225)):

    img = (((image * std) + mean) * 255).astype(np.uint8).copy() # Strangely, it needs the .copy() for everything to work!

    for annotation in annotations:
        if annotation[0:-1].sum() > 0:
            
            xc, yc, w, h, _ = [int(ann) for ann in annotation]
            
            x1 = xc - w//2
            y1 = yc - h//2
            x2 = xc + w//2
            y2 = yc + h//2

            bbox = [x1, y1, x2, y2]

            # Draw the bbox.
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
            # Add the label and the confidence.
            txt = 'COTS'
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            img = cv2.rectangle(img, (bbox[0], bbox[1]-txt_size[1]-2), 
                                (bbox[0]+txt_size[0], bbox[1]-2), (0, 0, 255), -1)
            img = cv2.putText(img, txt, (bbox[0], bbox[1]-2), font, 0.5, 
                              (255, 255, 255), thickness=1, 
                              lineType=cv2.LINE_AA)

    plt.imshow(img)
    plt.show()
    
###############################################################################
    
if __name__ == '__main__':
    def convert_arg_line_to_args(arg_line):
        """ A useful override of this for argparse to treat each space-separated 
        word as an argument"""
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg
            
    # To enable yaml loader to determine the 1e-3 -s 0.0001 and not a string, we
    # need to give it an alternate resolver to fix this problem.
    custom_loader = yaml.SafeLoader
    custom_loader.add_implicit_resolver(u'tag:yaml.org,2002:float',
                                        re.compile(u'''^(?:
                                                   [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                                                   |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                                                   |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                                                   |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                                                   |[-+]?\\.(?:inf|Inf|INF)
                                                   |\\.(?:nan|NaN|NAN))$''', re.X),
                                       list(u'-+0123456789.'))

    parser = argparse.ArgumentParser(description='YoloX Pytorch 1.10 Implementation.', fromfile_prefix_chars='@') # This allows for a command-line interface.
    parser.convert_arg_line_to_args = convert_arg_line_to_args # Override the argparse reader when reading from a .txt file.

    # Model operation args
    parser.add_argument('--mode',             type=str,   help='train or test', default='train')
    parser.add_argument('--config',           type=str,   help='path and filename of config file to use', default='yolox_config.yaml')
    parser.add_argument('--batch_size',       type=int,   help='batch size', default=1)
    parser.add_argument('--valid_batch_size', type=int,   help='validation batch size', default=1)
    parser.add_argument('--num_epochs',       type=int,   help='number of epochs', default=400)
    parser.add_argument('--num_gpus',         type=int,   help='number of GPUs to use for training', default=1)
    parser.add_argument('--num_threads',      type=int,   help='number of threads to use for data loading', default=0)
    parser.add_argument('--save_directory',   type=str,   help='directory to save checkpoints and summaries', default='./models')
    parser.add_argument('--pretrained_model', type=str,   help='path to a pretrained model checkpoint to load', default='None')
    parser.add_argument('--initial_epoch',    type=int,   help='if used with pretrained_model, will start from this epoch', default=0)
    parser.add_argument('--gpu_id',           type=str,   help='specifies the gpu to use', default='0')

    if sys.argv.__len__() == 2: # This handls prefixes.
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
        
    with open(args.config, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=custom_loader)
        cfg['mode'] = args.mode
        cfg['train']['batch_size'] = args.batch_size
        cfg['train']['num_epochs'] = args.num_epochs
        
    # Create the dataloader objects.
    train_dataset = COTSDataLoader(args, cfg, mode='train', verbose=0)
    
    if cfg['valid']['data_file'] is not None:
        valid_dataset = COTSDataLoader(args, cfg, mode='valid', verbose=0)
        
    for i in range(100):
        x_batch_train, y_batch_train = next(iter(train_dataset.data)) # Get some samples.
        
        image = x_batch_train['image'].detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        annotations = y_batch_train['annotations'].detach().cpu().numpy()[0]
        vis_inputs(image, annotations, mean=(0.485, 0.456, 0.406), 
                   std=(0.229, 0.224, 0.225))
        
        time.sleep(1)
        
    # for step, (x_batch_val, y_batch_val) in enumerate(valid_dataset.data):