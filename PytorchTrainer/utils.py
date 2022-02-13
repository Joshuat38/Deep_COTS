# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

import numpy as np
import time
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import Sampler
import torch.distributed as dist

class DistributedSamplerNoEvenlyDivisible(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        num_samples = int(np.floor(len(self.dataset) * 1.0 / self.num_replicas))
        rest = len(self.dataset) - num_samples * self.num_replicas
        if self.rank < rest:
            num_samples += 1
        self.num_samples = num_samples
        self.total_size = len(dataset)
        # self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        # assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9998, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model.module if self.is_parallel(model) else model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - np.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            decay_value = self.decay(self.updates)

            state_dict = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay_value
                    v += (1.0 - decay_value) * state_dict[k].detach()
                    
    def is_parallel(self, model):
        """check if model is in parallel mode."""
    
        parallel_type = (nn.parallel.DataParallel, 
                         nn.parallel.DistributedDataParallel)
        return isinstance(model, parallel_type)

class Progress_Bar:
    
    def __init__(self, length=-1, default_status=None, format_dhms=True, 
                 end_msg='Progress Complete!'):
        
        self.length = length
        self.default_status = default_status
        self.total_time = 0
        self.t1 = 0
        self.t2 = 0
        self.average_time = 0
        self.num_terms = 0
        self.default_display_data = {'ETA' : 'Pending...',
                                     'Iteration Time' : '0.0'}
        self.longest_len = 0
        self.format_dhms = format_dhms
        self.end_msg = end_msg
        self.start_count = False
        
    def start_progress(self):
        self.t1 = time.time()
        self.default_display_data['Iteration Time'] = '0.0s'
        
        self.show_progress(step=0)
        
    def update_progress(self, step, new_display_data={}, new_status=None):
        
        def convert_time_format(x):

            secs = x
            if self.format_dhms == True: # Convert to days hours minutes seconds format for easier reading.
                if x >= 60: # Longer than a minute.
                    if x >= 3600: # Longer than an hour.
                        if x >= 86400: # Longer than a day.
                            days = int(x//86400)
                            hrs = x%86400 # Use modulus to get the hours.
                            mins = hrs%3600 # Use modulus to get the minutes.
                            hrs = int(hrs//3600) # Get the hours.
                            secs = int(mins%60) # Get the seconds without the minutes.
                            mins = int(mins)//60 # Get the minutes without the seconds.
                            msg = '{0}d {1}h {2}min {3}s'.format(days, hrs, mins, secs)
                        else:
                            hrs = int(x//3600) # Get the hours.
                            mins = x%3600 # Use modulus to get the minutes.
                            secs = int(mins%60) # Get the seconds without the minutes.
                            mins = int(mins)//60 # Get the minutes without the seconds.
                            msg = '{0}h {1}min {2}s'.format(hrs, mins, secs)
                    else:
                        secs = int(x%60) # Get the seconds without the minutes.
                        mins = int(x//60) # Get the minutes without the seconds.
                        msg = '{0}min {1}s'.format(mins, secs)
                else:
                    msg = '{:.2f}s'.format(secs)
            else:
                msg = '{:.2f}s'.format(secs)
                
            return msg
        
        self.t2 = time.time()
        time_diff = self.t2-self.t1
        
        if self.start_count == True:
            self.num_terms += 1
            self.average_time += (time_diff - self.average_time)/self.num_terms # This method ensures a running average is kept such that the time doesn't fluctuate too much.
            self.total_time += time_diff # This computes the total time. 
        else:
            self.start_count = True
        eta_time = (self.length - step) * self.average_time

        self.default_display_data['ETA'] = convert_time_format(eta_time) # Estimate the time remaing based on the time taken for each iteration averaged out.
        self.default_display_data['Iteration Time'] = convert_time_format(time_diff) # Find the time taken to process this iteration.
        
        self.show_progress(step=step, new_display_data=new_display_data,
                           new_status=new_status)
              
        self.t1 = time.time()
        
    def finish_progress(self, new_display_data={}, new_status=None, 
                        verbose=True, get_duration=False):
        
        self.t2 = time.time()
        self.total_time += self.t2-self.t1 # This computes the total time.
        self.average_time = round(self.total_time/self.length, 2)
        
        if verbose == True:
            text = "\n\n"+self.end_msg+" \nTotal Time Elapsed: {0} - Average time per iteration: {1} - Status: {2}".format(round(self.total_time, 3), 
                                                                                                                           self.average_time, 
                                                                                                                           'Done.' if new_status is None else new_status)
            print(text+'\n')
        else:
            print('\n')
            
        if get_duration == True:
            return self.total_time
            
        
    def report_times(self): # Only call if process is finished.
        return self.total_time, self.average_time
        
    def reset_progress(self, new_length=None, new_default_status=None):
        if new_length is not None:
            self.length = new_length
        if new_default_status is not None:
            self.default_status = new_default_status
            
        self.total_time = 0
        self.average_time = 0
        self.num_terms = 0
        self.t1 = 0
        self.t2 = 0
        
        self.longest_len = 0
        self.start_count = False
        
    def show_progress(self, step, new_display_data={}, new_status=None):
        '''
        update_progress() : Displays or updates a console progress bar
        Accepts a float between 0 and 1. Any int will be converted to a float.
        A value under 0 represents a 'halt'.
        A value at 1 or bigger represents 100%
        '''
        
        progress = step/self.length
        barLength = 30 # Modify this to change the length of the progress bar
        if new_status is None:
            status = self.default_status
        else:
            status = new_status
            
        if isinstance(progress, int):
            progress = float(progress)
            
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float"
            
        if progress < 0:
            progress = 0
            status = "Halt..."
            
        if progress >= 1:
            progress = 1
            status = "Done..."
            
        block = int(round(barLength*progress))
        if (block >= 1 and block < barLength) or (block >= 1 and progress < 1):
            arrow = ">"
            block -= 1
        else:
            arrow = ""
            block += 1
          
        data_msg = ""
        display_data = {**self.default_display_data, **new_display_data}
        for key, value in display_data.items():
            if isinstance(value, str):
                data_msg += key + ": " + value + " - "
            elif isinstance(value, dict):
                data_msg += key + ": " + str([k+": {:.4f}".format(v) for k, v in value.items()])  + " - "
            else:
                data_msg += key + ": {:.4f} - ".format(value)
            
        text = "\r{0}/{1}: [{2}] {3}Status: {4}".format(step, self.length, 
                                                         "="*(block-1) + arrow + \
                                                         "."*(barLength-block), 
                                                         data_msg, status)
        if len(text) < self.longest_len:
            diff = self.longest_len - len(text)
            text += ' '*diff
        elif len(text) > self.longest_len:
            self.longest_len = len(text)
            
        print(text, end='') # The \x1b[2K] is the escape sequence tellign the print to erase the previous line.