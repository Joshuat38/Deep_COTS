# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

import copy
import numpy as np
import torch

from PytorchTrainer.metrics import BaseMetric
from utils import yolox_post_process


def get_iou(bbox1, bbox2):
    '''
        adapted from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    '''
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = bbox1[2] * bbox1[3]
    bb2_area = bbox2[2] * bbox2[3]
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_tp_fp_fn(bboxes, gt_bboxes, iou_thresh):
    TP = []
    FP = []
    FN = 0
    
    gt = copy.deepcopy(gt_bboxes)
    bb = copy.deepcopy(bboxes)
    
    if len(bboxes) == 0:
        # all gt are false negative
        FN += len(gt)
    else:
        # for idx, b in enumerate(bb):
        #     b.append(scores[idx])
        bb.sort(key = lambda x: x[4], reverse = True) # Sort based on confidence scores.

        if len(gt) == 0: # Check if all our gt are invalid.
            # all bboxes are false positives
            for b in bb:
                FP.append(b[4])
        else:
            # match bbox with gt
            for b in bb:
                matched = False
                for g in gt:
                    # check whether gt box is already matched to an inference bb
                    if len(g) == 5:
                        # g bbox is unmatched
                        if get_iou(b, g) >= iou_thresh:
                            g.append(b[4]) # assign confidence values to g; marks g as matched
                            matched = True
                            TP.append(b[4])
                            break
                if not matched:
                    FP.append(b[4])
            for g in gt:
                if len(g) == 5: # Our gt includes labels so we need to check if we have a length of 5.
                    FN += 1
    return TP, FP, FN

class BboxIOU(BaseMetric):
    def __init__(self, ignore_thresh, down_strides=[8, 16, 32], num_classes=1, 
                 conf_thresh=0.05, nms_thresh=0.2, label_name=['COTS'], 
                 img_shape=[800, 800], name='BboxIOU'):
        super(BboxIOU, self).__init__(name=name)
        
        self.ignore_thresh = ignore_thresh
        self.down_strides = down_strides
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.label_name = label_name
        self.img_shape = img_shape
        
    def call(self, y_pred, y_true):
        
        y_true = y_true.clone() # Must clone the tensor to preserve the original.
        y_pred = [output.detach().clone() for output in y_pred] # Must detach and clone the tensor to preserve the original.
        
        y_pred_list = yolox_post_process(outputs=y_pred, 
                                         down_strides=self.down_strides, 
                                         num_classes=self.num_classes, 
                                         conf_thre=self.conf_thresh, 
                                         nms_thre=self.nms_thresh, 
                                         label_name=self.label_name, 
                                         img_shape=[self.img_shape for i in range(y_true.shape[0])]) # Converts to the desired format.
        
        # Convert y_true to xyxy format.
        bboxes = y_true[:, :, 0:-1]
        bboxes[:, :, 0] = bboxes[:, :, 0] - bboxes[:, :, 2] * 0.5 # x1
        bboxes[:, :, 1] = bboxes[:, :, 1] - bboxes[:, :, 3] * 0.5 # y1
        bboxes[:, :, 2] = bboxes[:, :, 0] + bboxes[:, :, 2] # x2
        bboxes[:, :, 3] = bboxes[:, :, 1] + bboxes[:, :, 3] # y2
        y_true[:, :, 0:-1] = bboxes
          
        y_true_list = y_true.cpu().numpy().tolist()

        batch_ious = []
        for i, y_pred_batch in enumerate(y_pred_list): # Loop through the batches of boxes.
            y_true_batch = y_true_list[i]
            
            if len(y_pred_batch) > 0: # If there are predictions in this batch, perform the computation.
                
                bbox_ious = []
                for pred_bbox in y_pred_batch: # Loop through the predictions in this batch.
                    largest_iou = 0.0
                    for gt_bbox in y_true_batch: # Loop through all gt bboxes.
                        if sum(gt_bbox) > 0: # Only check valid gt bboxes.
                            iou = get_iou(pred_bbox[0:4], gt_bbox[0:4])
                            if iou > largest_iou and iou > self.ignore_thresh:
                                largest_iou = iou
                                
                    bbox_ious.append(largest_iou)
                    
                bbox_ious = sum(bbox_ious)/len(bbox_ious)
                batch_ious.append(bbox_ious)
                
            elif len(y_pred_batch) == 0 and sum([sum(gt_bbox) for gt_bbox in y_true_batch]) > 0:
                bbox_ious = 0.0
                batch_ious.append(bbox_ious)
                
        if len(batch_ious) == 0:
            return None
        else:
            return sum(batch_ious)/len(batch_ious)
    
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true)
        if res is not None: # We only want to measure bounding box accuracy.
            if self.value is None:
                self.value = res
                if self.eval == True:
                    self.n += 1
            else:
                if self.eval == False:
                    self.value = self.value + (res-self.value)/self.accumulate_limit
                else:
                    self.value = self.value + (res-self.value)/(self.n)
                    self.n += 1
                    
    def result(self):
        return self.value # Remove the .item() since this metric is evaluated on the cpu.
    
class F2_Score(BaseMetric):
    def __init__(self, iou_thresholds=np.linspace(0.3, 0.8, 11),
                 down_strides=[8, 16, 32], num_classes=1, conf_thresh=0.05, 
                 nms_thresh=0.2, label_name=['COTS'], img_shape=[800, 800], 
                 name='F2_Score'):
        super(F2_Score, self).__init__(name=name)
        
        self.iou_thresholds = iou_thresholds
        self.down_strides = down_strides
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.label_name = label_name
        self.img_shape = img_shape
        
    def call(self, y_pred, y_true):
        
        y_true = y_true.clone() # Must clone the tensor to preserve the original.
        y_pred = [output.detach().clone() for output in y_pred] # Must detach and clone the tensor to preserve the original.
        
        y_pred_list = yolox_post_process(outputs=y_pred, 
                                         down_strides=self.down_strides, 
                                         num_classes=self.num_classes, 
                                         conf_thre=self.conf_thresh, 
                                         nms_thre=self.nms_thresh, 
                                         label_name=self.label_name, 
                                         img_shape=[self.img_shape for i in range(y_true.shape[0])]) # Converts to the desired format.
        
        # Convert y_true to xyxy format.
        bboxes = y_true[:, :, 0:-1]
        bboxes[:, :, 0] = bboxes[:, :, 0] - bboxes[:, :, 2] * 0.5 # x1
        bboxes[:, :, 1] = bboxes[:, :, 1] - bboxes[:, :, 3] * 0.5 # y1
        bboxes[:, :, 2] = bboxes[:, :, 0] + bboxes[:, :, 2] # x2
        bboxes[:, :, 3] = bboxes[:, :, 1] + bboxes[:, :, 3] # y2
        y_true[:, :, 0:-1] = bboxes
          
        y_true_list = y_true.cpu().numpy().tolist()
        
        # Remove all the invalid gt and replace with []
        tmp = []
        for y_true_batch in y_true_list:
            y_list = []
            for y_bbox in y_true_batch:
                if sum(y_bbox[0:4]) > 0:
                    y_list.append(y_bbox)
            tmp.append(y_list)
        y_true_list = tmp
        
        f2_mean = []
        f2_conf_mean = []
        for i, y_pred_batch in enumerate(y_pred_list): # Loop through the batches of boxes.
            y_true_batch = y_true_list[i]
            
            TP_base = [[] for r in self.iou_thresholds] # Confidence scores of true positives
            FP_base = [[] for r in self.iou_thresholds] # Confidence scores of true positives
            FN_base = [0 for r in self.iou_thresholds]  # Count of false negative boxes
            
            # For each of the IOU thresholds, compute the TP, FP, FN values.
            for j, iou_threshold in enumerate(self.iou_thresholds):
                
                    
                TP, FP, FN = get_tp_fp_fn(bboxes=y_pred_batch, 
                                          gt_bboxes=y_true_batch, 
                                          iou_thresh=iou_threshold)
                
                if len(TP):
                    TP_base[j].append(TP)
                if len(FP):
                    FP_base[j].append(FP)
                FN_base[j] += FN
               
            # Flatten the TP and FP nested lists.
            for j, iou_threshold in enumerate(self.iou_thresholds):
                TP_base[j] = [item for sublist in TP_base[j] for item in sublist]
                FP_base[j] = [item for sublist in FP_base[j] for item in sublist]
            
            F2list = []
            F2max = 0.0
            F2maxat = 0.0
            F2mean = 0.0
            
            # For all possible confidence ranges, compute the F2 max score and add to the mean list. 
            # This is a bad approach however will give me an idea of how well things are going.
            for c in np.arange(0.0, 1.0, 0.01):
                F2temp = []
                for j, iou_threshold in enumerate(self.iou_thresholds):
                    FNcount = FN_base[j] + sum(1 for tp in TP_base[j] if tp < c)
                    TPcount = sum(1 for tp in TP_base[j] if tp >= c)
                    FPcount = sum(1 for fp in FP_base[j] if fp >= c)
                    R = TPcount / (TPcount + FNcount + 0.0001)
                    P = TPcount / (TPcount + FPcount + 0.0001)
                    F2 = (5 * P * R) / (4 * P + R + 0.0001)
                    F2temp.append(F2)
                
                F2mean = np.mean(F2temp)
                F2list.append((c, F2mean))
                if F2max < F2mean:
                    F2max = F2mean
                    F2maxat = c
            
            f2_conf_mean.append(F2maxat)
            f2_mean.append(F2max)
        return {'F2_Score' : sum(f2_mean)/len(f2_mean), 'Best_Conf' : sum(f2_conf_mean)/len(f2_conf_mean)}
    
    def accumulate_values(self, res, value, n):
        new_value = {}
        for key, val in value.items():
            new_value[key] = val + (res[key]-val)/n
            
        return new_value
    
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true)
        if res is not None: # We only want to measure bounding box accuracy.
            if self.value is None:
                self.value = res
                if self.eval == True:
                    self.n += 1
            else:
                if self.eval == False:
                    # self.value = self.value + (res-self.value)/self.accumulate_limit
                    self.value = self.accumulate_values(res, self.value, self.accumulate_limit)
                else:
                    # self.value = self.value + (res-self.value)/(self.n)
                    self.value = self.accumulate_values(res, self.value, self.n)
                    self.n += 1
                    
    def result(self):
        return self.value # Remove the .item() since this metric is evaluated on the cpu.

###############################################################################
# Depth estimation metrics.

class Threshold(BaseMetric):
    def __init__(self, thresh, clip_min=None, clip_max=None, mask_fn=None,
                 name='Threshold'):
        super(Threshold, self).__init__(name=name)
        
        self.thresh = thresh
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_pred, y_true):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        thresh = torch.maximum((y_true / y_pred), (y_pred / y_true))
        return torch.mean((thresh < self.thresh).float())
    
class AbsRelativeError(BaseMetric):
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='AbsRelativeError'):
        super(AbsRelativeError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_pred, y_true):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        return torch.mean(torch.abs(y_true - y_pred) / y_true)

class SquRelativeError(BaseMetric):
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='SquRelativeError'):
        super(SquRelativeError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_pred, y_true):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        return torch.mean(((y_true - y_pred) ** 2) / y_true)
    
class RootMeanSquareError(BaseMetric):
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='RootMeanSquareError'):
        super(RootMeanSquareError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_pred, y_true):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        rms = (y_true - y_pred) ** 2
        return torch.sqrt(torch.mean(rms))
    
class LogRootMeanSquareError(BaseMetric):
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='LogRootMeanSquareError'):
        super(LogRootMeanSquareError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_pred, y_true):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        log_rms = (torch.log(y_true) - torch.log(y_pred)) ** 2
        return torch.sqrt(torch.mean(log_rms))
    
class SilogError(BaseMetric):
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='SilogError'):
        super(SilogError, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_pred, y_true):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        err = torch.log(y_pred) - torch.log(y_true)
        return torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
    
class Log10Error(BaseMetric):
    def __init__(self, clip_min=None, clip_max=None, mask_fn=None,
                 name='Log10Error'):
        super(Log10Error, self).__init__(name=name)

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mask_fn = mask_fn
        
    def call(self, y_pred, y_true):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mask = None
        if self.clip_min is not None or self.clip_max is not None:
            y_pred = torch.clamp(y_pred, self.clip_min, self.clip_max) # Restrict to range.
            mask = torch.logical_and(y_true > self.clip_min, 
                                     y_true < self.clip_max) # Create a mask for values in the ground truth outside to clip limits.

        if self.mask_fn is not None: # Perform eigen cropping or other various trickery.
            mask = self.mask_fn(mask, shape=y_true.shape)
            
        if mask is not None:   
            y_true = y_true[mask] # Mask out bad values.
            y_pred = y_pred[mask] # Replicate this masking on the predictions so the shapes match.
        
        err = torch.abs(torch.log10(y_pred) - torch.log10(y_true))
        return torch.mean(err)
    
###############################################################################
# Segmentation metrics.

""" 
Note that for large numbers of classes ( > 20 ) the torch.bincount method
slows down very rapidly. It would be better not to evaluate segmentation
metrics during training time and evaluate only during evaluation.

This is because the counting function cannot be multi-treaded. 

When evaluating more that 20 classes, set compute_metrics_on_train to False.
It is also possible that the metrics may run faster on GPU. In that case,
it would be best to design an approach that only computes the confusion matrix
once. That will significantly increase testing speed.

These options should be explored in more detail in future.
"""

class Pixelwise_Accuracy(BaseMetric):
    def __init__(self, num_classes, ignore_index=None, name='Pixelwise_Accuracy'):
        super(Pixelwise_Accuracy, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
        
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true) 
        if self.value is None:
            self.value = res
        else:
            if self.eval == False:
                if self.n > self.accumulate_limit:
                    self.reset_confusion_matrix()
                    self.n = self.accumulate_limit//2
            self.value = res
        self.n += 1
                
    def reset(self):
        self.value = None
        self.reset_confusion_matrix()
        
    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
    
    def get_confusion_matrix(self, y_pred, y_true):
        # Make the confusion matrix.
        mask = (y_true >= 0) & (y_true < self.num_classes)# & (y_true != self.ignore_index)
        label = self.num_classes * y_true[mask].int() + y_pred[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        self.confusion_matrix += self.get_confusion_matrix(y_pred, y_true)
        
    def call(self, y_pred, y_true):
        
        # Get our outputs and labels in index form.
        y_pred = y_pred.argmax(dim=1)

        # Make the confusion matrix.
        self.update_confusion_matrix(y_pred, y_true)
          
        # Find the pixelwise accuracy.
        return torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
    
class Class_Accuracy(BaseMetric):
    def __init__(self, num_classes, ignore_index=None, name='Class_Accuracy'):
        super(Class_Accuracy, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
        
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true)
        if self.value is None:
            self.value = res
        else:
            if self.eval == False:
                if self.n > self.accumulate_limit:
                    self.reset_confusion_matrix()
                    self.n = self.accumulate_limit//2
            self.value = res
        self.n += 1
                
    def reset(self):
        self.value = None
        self.reset_confusion_matrix()
        
    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
    
    def get_confusion_matrix(self, y_pred, y_true):
        # Make the confusion matrix.
        mask = (y_true >= 0) & (y_true < self.num_classes)# & (y_true != self.ignore_index)
        label = self.num_classes * y_true[mask].int() + y_pred[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, y_pred, y_true):
        self.confusion_matrix += self.get_confusion_matrix(y_pred, y_true)
        
    def call(self, y_pred, y_true):
        
        # Get our outputs and labels in index form.
        y_pred = y_pred.argmax(dim=1)

        # Make the confusion matrix.
        self.update_confusion_matrix(y_pred, y_true)
          
        # Find the pixelwise accuracy.
        class_acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1)
        return class_acc[~torch.isnan(class_acc)].mean()
    
class Mean_IoU(BaseMetric):
    def __init__(self, num_classes, ignore_index=None, name='Mean_IoU'):
        super(Mean_IoU, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
        
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true)
        if self.value is None:
            self.value = res
        else:
            if self.eval == False:
                if self.n > self.accumulate_limit:
                    self.reset_confusion_matrix()
                    self.n = self.accumulate_limit//2
            self.value = res
        self.n += 1
                
    def reset(self):
        self.value = None
        self.reset_confusion_matrix()
        
    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
    
    def get_confusion_matrix(self, y_pred, y_true):
        # Make the confusion matrix.
        mask = (y_true >= 0) & (y_true < self.num_classes)# & (y_true != self.ignore_index)
        label = self.num_classes * y_true[mask].int() + y_pred[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, y_pred, y_true):
        self.confusion_matrix += self.get_confusion_matrix(y_pred, y_true)
        
    def call(self, y_pred, y_true):

        # Get our outputs and labels in index form.
        y_pred = y_pred.argmax(dim=1)

        # Make the confusion matrix.
        self.update_confusion_matrix(y_pred, y_true)
        
        # Find the mean intersection-over-union. mIoU = TP/(TP + FN + FP)
        mIoU = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=1) + \
                                                    self.confusion_matrix.sum(dim=0) - \
                                                        torch.diag(self.confusion_matrix))

        return mIoU[~torch.isnan(mIoU)].mean()
    
class Class_IoU(BaseMetric):
    def __init__(self, num_classes, ignore_index=None, class_names=None,
                 replace_nan=True, name='Class_IoU'):
        super(Class_IoU, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.replace_nan = replace_nan
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
        
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true)
        if self.value is None: 
            self.value = res
        else:
            if self.eval == False:
                if self.n > self.accumulate_limit:
                    self.reset_confusion_matrix()
                    self.n = self.accumulate_limit//2
            self.value = res
        self.n += 1
                
    def reset(self):
        self.value = None
        self.reset_confusion_matrix()
        
    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
    
    def get_confusion_matrix(self, y_pred, y_true):
        # Make the confusion matrix.
        mask = (y_true >= 0) & (y_true < self.num_classes)# & (y_true != self.ignore_index)
        label = self.num_classes * y_true[mask].int() + y_pred[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        self.confusion_matrix += self.get_confusion_matrix(y_pred, y_true)
                
    def result(self):
        result = {}
        if self.class_names is None:
            result = self.value
        else:
            for key, value in self.value.items():
                result[self.class_names[key]] = value
                
        return result
        
    def call(self, y_pred, y_true):

        # Get our outputs and labels in index form.
        y_pred = y_pred.argmax(dim=1)

        # Make the confusion matrix.
        self.update_confusion_matrix(y_pred, y_true)
        
        # Find the mean intersection-over-union. mIoU = TP/(TP + FN + FP)
        class_IoU = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=1) + \
                                                         self.confusion_matrix.sum(dim=0) - \
                                                         torch.diag(self.confusion_matrix))
        if self.replace_nan == True:
            class_IoU[torch.isnan(class_IoU)] = 0.0
        class_IoU = class_IoU.cpu().numpy().tolist()
        return {str(c):cls_iou for c, cls_iou in enumerate(class_IoU)}
    
class Frequency_Weighted_IoU(BaseMetric):
    def __init__(self, num_classes, ignore_index=None, name='Frequency_Weighted_IoU'):
        super(Frequency_Weighted_IoU, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
        
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true) 
        if self.value is None:
            self.value = res
        else:
            if self.eval == False:
                if self.n > self.accumulate_limit:
                    self.reset_confusion_matrix()
                    self.n = self.accumulate_limit//2
            self.value = res
        self.n += 1
                
    def reset(self):
        self.value = None
        self.reset_confusion_matrix()
        
    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
    
    def get_confusion_matrix(self, y_pred, y_true):
        # Make the confusion matrix.
        mask = (y_true >= 0) & (y_true < self.num_classes)# & (y_true != self.ignore_index)
        label = self.num_classes * y_true[mask].int() + y_pred[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        self.confusion_matrix += self.get_confusion_matrix(y_pred, y_true)
        
    def call(self, y_pred, y_true):
        
        # Get our outputs and labels in index form.
        y_pred = y_pred.argmax(dim=1)

        # Make the confusion matrix.
        self.update_confusion_matrix(y_pred, y_true)
        
        # Find the frequency weighted intersection-over-union.
        freq = self.confusion_matrix.sum(dim=1) / self.confusion_matrix.sum()
        iu = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=1) + \
                                                  self.confusion_matrix.sum(dim=0) - \
                                                  torch.diag(self.confusion_matrix))
        return (freq[freq > 0] * iu[freq > 0]).sum()
    
class Weighted_F1_Score(BaseMetric):
    def __init__(self, num_classes, ignore_index=None, beta=1, 
                 name='Weighted_F1_Score'):
        super(Weighted_F1_Score, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.beta = beta
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
        
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true)
        if self.value is None:
            self.value = res
        else:
            if self.eval == False:
                if self.n > self.accumulate_limit:
                    self.reset_confusion_matrix()
                    self.n = self.accumulate_limit//2
            self.value = res
        self.n += 1
                
    def reset(self):
        self.value = None
        self.reset_confusion_matrix()
        
    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
    
    def get_confusion_matrix(self, y_pred, y_true):
        # Make the confusion matrix.
        mask = (y_true >= 0) & (y_true < self.num_classes)# & (y_true != self.ignore_index)
        label = self.num_classes * y_true[mask].int() + y_pred[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        self.confusion_matrix += self.get_confusion_matrix(y_pred, y_true)
        
        
    def call(self, y_pred, y_true):
        
        # Get our outputs and labels in index form.
        y_pred = y_pred.argmax(dim=1)

        # Make the confusion matrix.
        self.update_confusion_matrix(y_pred, y_true)
        
        precision = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=0)
        recall = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1)
        
        score = (1 + self.beta ** 2) * (precision * recall) / ((self.beta ** 2 * precision) + recall)
        return score[~torch.isnan(score)].mean()
    
class Precision(BaseMetric):
    def __init__(self, num_classes, ignore_index=None, name='Precision'):
        super(Precision, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
        
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true)
        if self.value is None:
            self.value = res
        else:
            if self.eval == False:
                if self.n > self.accumulate_limit:
                    self.reset_confusion_matrix()
                    self.n = self.accumulate_limit//2
            self.value = res
        self.n += 1
                
    def reset(self):
        self.value = None
        self.reset_confusion_matrix()
        
    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
    
    def get_confusion_matrix(self, y_pred, y_true):
        # Make the confusion matrix.
        mask = (y_true >= 0) & (y_true < self.num_classes)# & (y_true != self.ignore_index)
        label = self.num_classes * y_true[mask].int() + y_pred[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        self.confusion_matrix += self.get_confusion_matrix(y_pred, y_true)
        
        
    def call(self, y_pred, y_true):
        
        # Get our outputs and labels in index form.
        y_pred = y_pred.argmax(dim=1)

        # Make the confusion matrix.
        self.update_confusion_matrix(y_pred, y_true)
        
        # Find the precision. P = TP/(TP + FP)
        precision = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=0)
        return precision[~torch.isnan(precision)].mean()
    
class Recall(BaseMetric):
    def __init__(self, num_classes, ignore_index=None, name='Recall'):
        super(Recall, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
        
    def compute(self, y_pred, y_true):
        
        res = self.call(y_pred, y_true)
        if self.value is None:
            self.value = res
        else:
            if self.eval == False:
                if self.n > self.accumulate_limit:
                    self.reset_confusion_matrix()
                    self.n = self.accumulate_limit//2
            self.value = res
        self.n += 1
                
    def reset(self):
        self.value = None
        self.reset_confusion_matrix()
        
    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).cuda()
    
    def get_confusion_matrix(self, y_pred, y_true):
        # Make the confusion matrix.
        mask = (y_true >= 0) & (y_true < self.num_classes)# & (y_true != self.ignore_index)
        label = self.num_classes * y_true[mask].int() + y_pred[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape((self.num_classes, self.num_classes))

    def update_confusion_matrix(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        self.confusion_matrix += self.get_confusion_matrix(y_pred, y_true)
        
        
    def call(self, y_pred, y_true):
        
        # Get our outputs and labels in index form.
        y_pred = y_pred.argmax(dim=1)

        # Make the confusion matrix.
        self.update_confusion_matrix(y_pred, y_true)
        
        # Find the recall. P = TP/(TP + FN)
        recall = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1)
        return recall[~torch.isnan(recall)].mean()