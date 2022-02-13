# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torchvision

from PytorchTrainer.callbacks import VisualizerCallback

def yolox_post_process(outputs, down_strides, num_classes, conf_thre, nms_thre, label_name, img_shape): # img_ratios, 
    hw = [i.shape[-2:] for i in outputs]
    grids, strides = [], []
    for (hsize, wsize), stride in zip(hw, down_strides):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)  # bs, all_anchor, 85(+128)
    grids = torch.cat(grids, dim=1).type(outputs.dtype).to(outputs.device)
    strides = torch.cat(strides, dim=1).type(outputs.dtype).to(outputs.device)

    # x, y
    outputs[..., 0:2] = (outputs[..., 0:2] + grids) * strides
    # w, h
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    # obj
    outputs[..., 4:5] = torch.sigmoid(outputs[..., 4:5])
    # 80 class
    outputs[..., 5:5 + num_classes] = torch.sigmoid(outputs[..., 5:5 + num_classes])
    # reid
    reid_dim = outputs.shape[2] - num_classes - 5
    if reid_dim > 0:
        outputs[..., 5 + num_classes:] = F.normalize(outputs[..., 5 + num_classes:], dim=2)

    box_corner = outputs.new(outputs.shape)
    box_corner[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2  # x1
    box_corner[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2  # y1
    box_corner[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2] / 2  # x2
    box_corner[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3] / 2  # y2
    outputs[:, :, :4] = box_corner[:, :, :4]

    output = [[] for _ in range(len(outputs))]
    # output = []
    for i, image_pred in enumerate(outputs):
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        if reid_dim > 0:
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5 + num_classes:]), 1)
        else:
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            if reid_dim > 0:
                output[i].append([*[0.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0])
            else:
                output[i].append([*[0.0, 0.0, 0.0, 0.0], 0.0, 0.0])
            # continue
        else:
            nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5],
                                                        detections[:, 6], nms_thre)
            detections = detections[nms_out_index]
    
            # detections[:, :4] = detections[:, :4] / img_ratios[i]
    
            img_h, img_w = img_shape[i]
            for det in detections:
                x1, y1, x2, y2, obj_conf, class_conf, class_pred = det[0:7]
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                conf = float(obj_conf * class_conf)
                # label = label_name[int(class_pred)]
                label = int(class_pred)
                # clip bbox
                bbox[0] = max(0, min(img_w, bbox[0]))
                bbox[1] = max(0, min(img_h, bbox[1]))
                bbox[2] = max(0, min(img_w, bbox[2]))
                bbox[3] = max(0, min(img_h, bbox[3]))
    
                if reid_dim > 0:
                    reid_feat = det[7:].cpu().numpy().tolist()
                    output[i].append([*bbox, label, conf, reid_feat])
                else:
                    output[i].append([*bbox, label, conf])
    # print(output)
    return output

class Visualize(VisualizerCallback):
    
    def __init__(self, classes, vis_keys=['image', 'annotations'],
                 train_figure_size=[720, 1280], test_figure_size=[720, 1280],
                 show_results=True, save_dir=None, save_test_images=False, 
                 save_train_images=False,  img_mean=[0.0, 0.0, 0.0], 
                 img_std=[1.0, 1.0, 1.0], img_rescale=255.0, num_steps=10, 
                 show_on_train=False, **post_process_kwargs):
        super(Visualize, self).__init__(vis_keys=vis_keys)
        
        self.classes = classes
        self.train_figure_size = train_figure_size
        self.test_figure_size = test_figure_size
        self.show_results = show_results
        self.save_dir = save_dir
        self.save_test_images = save_test_images
        self.save_train_images = save_train_images
        self.img_mean = img_mean
        self.img_std = img_std
        self.img_rescale = img_rescale
        self.num_steps = num_steps
        self.show_on_train = show_on_train
        self.post_process_kwargs = post_process_kwargs
        
    def on_train_end(self, vis_data, step):
        if step > 0 and (step < self.num_steps or self.num_steps == -1) and self.show_on_train == True:
            self.visualize(vis_data, step, save_images=self.save_train_images,
                           figure_size=self.train_figure_size, mode='train')
        
    def on_test_end(self, vis_data, step):
        if step >= 0 and (step < self.num_steps or self.num_steps == -1):
            self.visualize(vis_data, step, save_images=self.save_test_images,
                           figure_size=self.test_figure_size, mode='test')
            
    def cxcywh2xyxy(self, y_true):
        bboxes = y_true[:, 0:-1]
        bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5 # x1
        bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5 # y1
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] # x2
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] # y2
        y_true[:, 0:-1] = bboxes
        return y_true
            
    def postprocess(self, image, y_true, y_pred, size, mode='test'):
        """
        :param image: RGB, uint8
        :param size:
        :param bboxes:
        :return: RGB, uint8
        """
        
        # Convert the image back to the expected uint8 format.    
        image = np.add(np.multiply(image, np.array(self.img_std)), 
                       np.array(self.img_mean)) * self.img_rescale
        image = image.astype(np.uint8)
        
        # Convert y_true to xyxy format.
        y_true = self.cxcywh2xyxy(y_true)
        
        if mode == 'test':
            ih, iw = image.shape[:2]
            h, w = size
        
            scale = min(iw / w, ih / h)
            nw, nh = int(scale * w), int(scale * h)
            dw, dh = (iw - nw) // 2, (ih - nh) // 2
        
            image = image[dh:nh + dh, dw:nw + dw, :]
            image = cv2.resize(image, (w, h))
            
            # Resize to the original size and shape.
            y_true[:, [0, 2]] = np.clip((y_true[:, [0, 2]] - dw) / scale, 0., w)
            y_true[:, [1, 3]] = np.clip((y_true[:, [1, 3]] - dh) / scale, 0., h)
            
            # Adjust to the new input shape.
            y_pred[:, [0, 2]] = np.clip((y_pred[:, [0, 2]] - dw) / scale, 0., w)
            y_pred[:, [1, 3]] = np.clip((y_pred[:, [1, 3]] - dh) / scale, 0., h)

        return image, y_true, y_pred
    
    def draw_annotations(self, img, y_true, y_pred):
        
        img = img.copy()
        
        for y_true_ann in y_true:
            if y_true_ann[0:4].sum() > 0:

                bbox = y_true_ann[0:4].astype(np.int64)

                # Draw the bbox.
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Add the label and the confidence.
                txt = 'COTS'
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
                img = cv2.rectangle(img, (bbox[0], bbox[1]-txt_size[1]-2), 
                                    (bbox[0]+txt_size[0], bbox[1]-2), (0, 255, 0), -1)
                img = cv2.putText(img, txt, (bbox[0], bbox[1]-2), font, 0.5, 
                                  (255, 255, 255), thickness=1, 
                                  lineType=cv2.LINE_AA)
                
        for y_pred_ann in y_pred:
            if y_pred_ann[0:4].sum() > 0:

                bbox = y_pred_ann[0:4].astype(np.int64)
                conf = y_pred_ann[-1]

                # Draw the bbox.
                img = cv2.rectangle(img, (bbox[0], bbox[1]), 
                                    (bbox[2], bbox[3]), (0, 0, 255), 2)
                
                # Add the label and the confidence.
                txt = f'COTS-{conf:.2f}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
                img = cv2.rectangle(img, (bbox[0], bbox[1]-txt_size[1]-2), 
                                    (bbox[0]+txt_size[0], bbox[1]-2), (0, 0, 255), -1)
                img = cv2.putText(img, txt, (bbox[0], bbox[1]-2), font, 0.5, 
                                  (255, 255, 255), thickness=1, 
                                  lineType=cv2.LINE_AA)
                
        return img
        
    def visualize(self, vis_data, step, save_images=False, 
                  figure_size=[480, 640], mode='test'):
        imgs = vis_data['Inputs']['image'].permute(0, 2, 3, 1).numpy()
        y_true = vis_data['Targets']['annotations'].numpy()
        
        # y_pred = vis_data['Logits']['annotations'].numpy()[0]
        
        # img_ratio = [float(min(opt.test_size[0] / img_info[0][i], opt.test_size[1] / img_info[1][i])) for i in
        #                  range(inps.shape[0])]
        img_shape = [[int(imgs[i].shape[0]), int(imgs[i].shape[1])] for i in range(imgs.shape[0])]
        
        y_pred = yolox_post_process(outputs=vis_data['Logits']['annotations'], 
                                    img_shape=img_shape, **self.post_process_kwargs)
        
        img = imgs[0]
        y_true = y_true[0]
        y_pred = np.array(y_pred[0]).astype(np.float32)
        
        img, y_true, y_pred = self.postprocess(img, y_true, y_pred, 
                                               size=figure_size, mode=mode)


        dir_names = ['image', 'image_annotated', 'pred_annotations', 
                     'gt_annotations']
        
        if save_images == True:
            if self.show_raw == True:
                image_fig_dir = self.save_dir+'/'+dir_names[0]
                if not os.path.exists(image_fig_dir):
                    os.makedirs(image_fig_dir)
                
            annotated_fig_dir = self.save_dir+'/'+dir_names[1]
            if not os.path.exists(annotated_fig_dir):
                os.makedirs(annotated_fig_dir)
                    
            pred_annotations_list_dir = self.save_dir+'/'+dir_names[2]
            if not os.path.exists(pred_annotations_list_dir):
                os.makedirs(pred_annotations_list_dir)
                
            gt_annotations_list_dir = self.save_dir+'/'+dir_names[3]
            if not os.path.exists(gt_annotations_list_dir):
                os.makedirs(gt_annotations_list_dir)
                
        #######################################################################
          
        # Make the RGB images for display.    
        im_fig = plt.figure(frameon=False)
        im_fig.set_size_inches(figure_size[1]/float(im_fig.get_dpi()), 
                               figure_size[0]/float(im_fig.get_dpi()))
        ax = im_fig.add_subplot(111)
        ax.imshow(img, aspect='auto')
        ax.set_axis_off() # Turns off the axis of the plot for simple picture.
        if save_images == True:
            im_fig.savefig(image_fig_dir+'/'+dir_names[0]+'_'+str(step+1)+'.png', 
                           bbox_inches='tight', pad_inches=0)
        if self.show_results == True:
            plt.show()
        plt.close(im_fig)
        
        #######################################################################
        
        # Make annotated RG images for visual inspection.
        im_fig = plt.figure(frameon=False)
        im_fig.set_size_inches(figure_size[1]/float(im_fig.get_dpi()), 
                               figure_size[0]/float(im_fig.get_dpi()))
        ax = im_fig.add_subplot(111)

        annotated_img = self.draw_annotations(img, y_true, y_pred)
        
        ax.imshow(annotated_img, aspect='auto')
        ax.set_axis_off() # Turns off the axis of the plot for simple picture.
        if save_images == True:
            im_fig.savefig(annotated_fig_dir+'/'+dir_names[1]+'_'+str(step+1)+'.png', 
                           bbox_inches='tight', pad_inches=0)
        if self.show_results == True:
            plt.show()
        plt.close(im_fig)
        
        #######################################################################
        
        plt.close('all')

class WarmupCosLR(_LRScheduler):
    """ YoloX Warmup Cos LR """
    
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr_ratio=0.05, 
                 warmup_init_lr=0, no_aug_steps=0, verbose=False):
        if total_steps <= 1.:
            raise ValueError('total_steps should be greater than 1.')
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.warmup_init_lr = warmup_init_lr
        self.no_aug_steps = no_aug_steps
        self.verbose = verbose
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_last_lr(self):
        return [self.warnup_cos_step(step=self.last_step, lr=base_lr) for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1

        new_lrs = [self.warnup_cos_step(step=step, lr=base_lr) for base_lr in self.base_lrs]
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
                
        if self.verbose == True:
            print(f'Learning rate changed to: {self.get_lr:.8f}')
            
    def warnup_cos_step(self, step, lr):
        """Cosine learning rate with warm up."""
        min_lr = lr * self.min_lr_ratio
        if step <= self.warmup_steps:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - self.warmup_init_lr) * pow(step / float(self.warmup_steps), 2) + self.warmup_init_lr
        elif step >= self.total_steps - self.no_aug_steps:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + np.cos(np.pi * (step - self.warmup_steps) 
                                                              / (self.total_steps - self.warmup_steps - self.no_aug_steps)))
        return lr