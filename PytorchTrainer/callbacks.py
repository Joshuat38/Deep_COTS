# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

""" This file contains all the custom callbacks to be used in this code. """

import sys
from torch import save as save_checkpoint

class PytorchCallback(object):
    def __init__(self, name='PytorchCallback'):
        self.name = name
        
    def on_batch_end(self, *args, **kwargs):
        pass
    
    def on_epoch_end(self, *args, **kwargs):
        pass
    
    def on_train_end(self, *args, **kwargs):
        pass
    
    def on_test_end(self, *args, **kwargs):
        pass
    
class VisualizerCallback(object):
    def __init__(self, vis_keys={}, name='VisualizerCallback'):
        self.name = name
        self.vis_keys = vis_keys
        
    def get_keys(self):
        # Returns the keys of the data to grab.
        return self.vis_keys
    
    def on_train_end(self, vis_data, step):
        pass
    
    def on_test_end(self, vis_data, step):
        pass

class TerminateOnNanOrInf(PytorchCallback):
    def __init__(self, nan=True, inf=True, name='Terminate_On_Nan_Or_Inf'):
        super(TerminateOnNanOrInf, self).__init__(name=name)
        self.nan = nan # Check for NaNs
        self.inf = inf # Check for Infs
      
    def on_batch_end(self, *args, **kwargs):  
        if kwargs['loss'].isnan().any() and self.inf == True: # Check if the loss has gone to NaN.
            sys.exit('Loss has become NaN. Aborting training.')
        elif kwargs['loss'].isinf().any() and self.nan == True: # Check if the loss has exploded.
            sys.exit('Loss has become Inf. Aborting training.')
        
class ModelCheckpoint(PytorchCallback):
    def __init__(self, save_path, monitor, verbose=1, save_best_only=False, 
                 mode='auto', save_freq='epoch', save_history=False,
                 name='Model_Checkpoint'):
        super(ModelCheckpoint, self).__init__(name=name)
        
        self.save_path = save_path
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.save_history = save_history
        
        self.value = None
        
    def on_epoch_end(self, model, history, **kwargs):
        
        if self.save_freq == 'epoch':
            new_value = history[self.monitor][-1]
            if self.value is None:
                if self.verbose == 1:
                    msg = f'\n{self.monitor} improved from Inf to {new_value:.8f}. Saving checkpoint to: {self.save_path}'
                    print(msg)
                self.save(new_value, model, history, verbose=0, **kwargs)
            else:
                if self.mode == 'max':
                    if new_value > self.value:
                        self.save(new_value, model, history, **kwargs)
                    else:
                        print(f'{self.monitor} did not improve from {self.value:.8f}.')
                            
                elif self.mode == 'min':
                    if new_value < self.value:
                        self.save(new_value, model, history, **kwargs)
                    else:
                        print(f'{self.monitor} did not improve from {self.value:.8f}.')
                else:
                    if 'loss' in self.monitor.lower():
                        if new_value < self.value:
                            self.save(new_value, model, history, **kwargs)
                        else:
                            print(f'{self.monitor} did not improve from {self.value:.8f}.')
                    else:
                        if new_value > self.value:
                            self.save(new_value, model, history, **kwargs)
                        else:
                            print(f'{self.monitor} did not improve from {self.value:.8f}.')
                
        else:
            pass
        
    def on_train_end(self, model, history, **kwargs):
        last_save_path = '/'.join(self.save_path.split('/')[0:-1]) + '/last_model_checkpoint'
        msg = f'\nSaving final checkpoint to: {last_save_path}'
        print(msg)
        
        self.save(self.value, model, history, verbose=0, 
                  custom_save_path=last_save_path, **kwargs)
        
        
    def save(self, new_value, model, history, verbose=1, custom_save_path=None, 
             **kwargs):
        if verbose == 1:
            msg = f'\n{self.monitor} improved from {self.value:.8f} to {new_value:.8f}. Saving checkpoint to: {self.save_path}'
            print(msg)
            
        state = {'model': model.state_dict(),
                 'epoch': history['Epoch'][-1]}
        if 'optimizer' in kwargs:
            state.update({'optimizer': kwargs['optimizer'].state_dict()})
        if 'lr_schedule' in kwargs:
            if kwargs['lr_schedule'] is not None:
                state.update({'lr_schedule': kwargs['lr_schedule'].state_dict()})
        if self.save_history == True:
            state.update({'history' : history})
            
        save_checkpoint(state, self.save_path if custom_save_path is None else custom_save_path)
        self.value = new_value

