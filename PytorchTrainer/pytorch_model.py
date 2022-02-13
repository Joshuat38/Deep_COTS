# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

import numpy as np
import torch

from PytorchTrainer.callbacks import TerminateOnNanOrInf
from PytorchTrainer.utils import Progress_Bar, ModelEMA

class PytorchModel:
    def __init__(self, model, optimizer=None, loss=None, loss_weights=None, 
                 loss_grouped=False, metrics=None, lr_schedule=None, 
                 history=None, model_name=None, use_ema=False):
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.loss = loss
        self.loss_weights = loss_weights
        self.loss_grouped = loss_grouped
        self.metrics = metrics
        self.model_name = model_name
        
        self.ema = ModelEMA(self.model) if use_ema == True else None
        
        self.total_loss = 0   
        if history is None:
            self.history = {'Epoch' : [], 'Epoch_Duration' : []}
        else:
            self.history = history
            
    def print_summary(self, show_model=False):
    
        if show_model == True:
            print(self.model)
        num_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        print("Total number of parameters: {}".format(num_params))
    
        num_params_update = sum([np.prod(p.shape) for p in self.model.parameters() if p.requires_grad])
        print("Total number of trainable parameters: {}".format(num_params_update))
        
        print("Total number of non-trainable parameters: {}".format(num_params-num_params_update))
    
    def fit(self, epochs, batch_size, train_dataset, train_steps, 
            initial_epoch=0, valid_batch_size=None, valid_dataset=None, 
            valid_steps=None, callbacks_list=[TerminateOnNanOrInf], 
            visualizer=None, loss_acc_accumulate=500, apply_fn=None, 
            compute_metrics_on_train=True):
        
        def train_step(x, y, step, progress_data={}):
            
            # This line of code places the data onto the gpu. This must be call inside the training step otherwise
            # the memory allocation remains meaning gradually GPU resources are eaten up.
            x = {key : torch.autograd.Variable(val.cuda()) for key, val in x.items()} # This sends the data to the GPU. We must make a new variable so that python can move it.
            y = {key : torch.autograd.Variable(val.cuda()) for key, val in y.items()} # This sends the data to the GPU. We must make a new variable so that python can move it.
            
            self.optimizer.zero_grad()
            
            total_loss = 0
            losses = {}
            
            with torch.cuda.amp.autocast():
                logits = self.model(x)  # Logits for this minibatch (May have multiple outputs).

            # Compute the loss value for this minibatch.
            if self.loss_grouped == True:
                total_loss, losses = self.loss.forward(logits, y, total_loss, losses)
            elif type(self.loss) == dict:
                for loss_key, loss_obj in self.loss.items():        
                    loss = loss_obj.forward(logits[loss_key], y[loss_key]) # Update the loss for a given output.
                    loss *= 1 if self.loss_weights is None else self.loss_weights[loss_key] # Apply the loss weights where appropriate.
                    losses[loss_key+'_Loss'] = loss.item() # Update the loss values where expected.
                    total_loss += loss # Add this to the total loss.
            else:
                for k in y:
                    total_loss = self.loss.forward(logits[k], y[k])
            losses['Loss'] = total_loss.item()
                
            grad_scalar.scale(total_loss).backward()
            grad_scalar.step(self.optimizer)
            grad_scalar.update()
            
            if self.ema is not None: # Apply exponential moving average to improve model performance.
                self.ema.update(self.model)
            
            if self.lr_schedule is not None:
                self.lr_schedule.step()
            
            if compute_metrics_on_train == True:
                for metric_key, metric_values in self.metrics.items(): # Make the metrics as a class that has update_state as a function.
                    if type(metric_values) == dict:
                        for metric in metric_values.values():
                            metric.compute(logits[metric_key].detach() if torch.is_tensor(logits[metric_key]) else logits[metric_key], 
                                           y[metric_key].detach() if torch.is_tensor(y[metric_key]) else y[metric_key])
                    else:
                        for k in y:
                            metric_values.compute(logits[k].detach() if torch.is_tensor(logits[k]) else logits[k], 
                                                  y[k].detach() if torch.is_tensor(y[k]) else y[k])
                            
            if visualizer is not None:
                data_keys = visualizer.get_keys() # Get the keys of the data to grab.
                vis_data = {'Inputs' : {}, 'Logits' : {}, 'Targets' : {}}
                for key in data_keys:
                    if key in x.keys():
                        vis_data['Inputs'][key] = x[key].detach().cpu() if torch.is_tensor(x[key]) else x[key]
                    if key in logits.keys():
                        vis_data['Logits'][key] = logits[key].detach().cpu() if torch.is_tensor(logits[key]) else logits[key]
                    if key in y.keys():
                        vis_data['Targets'][key] = y[key].detach().cpu() if torch.is_tensor(y[key]) else y[key]   
                visualizer.on_train_end(vis_data, step)
                    
            for loss_key, loss_value in losses.items(): # Collect the loss values for display.
                progress_data[loss_key] = self.incremental_average(progress_data[loss_key] if step > 0 else 0, 
                                                                   loss_value, step+1 if step+1 < loss_acc_accumulate else loss_acc_accumulate)
            
            # Apply all on batch end callbacks here.
            for callback in callbacks_list:
                callback.on_batch_end(self.model, history=self.history, 
                                      optimizer=self.optimizer, 
                                      lr_schedule=self.lr_schedule,
                                      loss=total_loss, training=True)
            
            return progress_data
        
        def valid_step(x, y, step, progress_data={}):

            # This line of code places the data onto the gpu. This must be call inside the training step otherwise
            # the memory allocation remains meaning gradually GPU resources are eaten up.
            x = {key : torch.autograd.Variable(val.cuda()) for key, val in x.items()} # This sends the data to the GPU. We must make a new variable so that python can move it.
            y = {key : torch.autograd.Variable(val.cuda()) for key, val in y.items()} # This sends the data to the GPU. We must make a new variable so that python can move it.
        
            total_loss = 0
            losses = {}
            
            with torch.cuda.amp.autocast():
                logits = self.model(x)  # Logits for this minibatch (May have multiple outputs).
  
            # Compute the loss value for this minibatch.
            if self.loss_grouped == True:
                total_loss, losses = self.loss.forward(logits, y, total_loss, losses)
            elif type(self.loss) == dict:
                for loss_key, loss_obj in self.loss.items():
                    loss = loss_obj.forward(logits[loss_key], y[loss_key]) # Update the loss for a given output.
                    loss *= 1 if self.loss_weights is None else self.loss_weights[loss_key] # Apply the loss weights where appropriate.
                    losses[loss_key+'_Val_Loss'] = loss.item() # Update the loss values where expected.
                    total_loss += loss # Add this to the total loss.
            else:
                for k in y:
                    total_loss = self.loss.forward(logits[k], y[k])
            losses['Val_Loss'] = total_loss.item()
            
            for metric_key, metric_values in self.metrics.items(): # Make the metrics as a class that has update_state as a function.
                if type(metric_values) == dict:
                    for metric in metric_values.values():
                        metric.compute(logits[metric_key].detach() if torch.is_tensor(logits[metric_key]) else logits[metric_key], 
                                       y[metric_key].detach() if torch.is_tensor(y[metric_key]) else y[metric_key])
                else:
                    for k in y:
                        metric_values.compute(logits[k].detach() if torch.is_tensor(logits[k]) else logits[k], 
                                              y[k].detach() if torch.is_tensor(y[k]) else y[k])
                        
            if visualizer is not None:
                data_keys = visualizer.get_keys() # Get the keys of the data to grab.
                vis_data = {'Inputs' : {}, 'Logits' : {}, 'Targets' : {}}
                for key in data_keys:
                    if key in x.keys():
                        vis_data['Inputs'][key] = x[key].detach().cpu() if torch.is_tensor(x[key]) else x[key]
                    if key in logits.keys():
                        vis_data['Logits'][key] = logits[key].detach().cpu() if torch.is_tensor(logits[key]) else logits[key]
                    if key in y.keys():
                        vis_data['Targets'][key] = y[key].detach().cpu() if torch.is_tensor(y[key]) else y[key]   
                visualizer.on_test_end(vis_data, step)
                    
            for loss_key, loss_value in losses.items(): # Collect the loss values for display.
                progress_data[loss_key] = self.incremental_average(progress_data[loss_key] if step > 0 else 0, 
                                                                   loss_value, step+1)
                
            # Apply all on batch end callbacks here.
            for callback in callbacks_list:
                callback.on_batch_end(self.model, history=self.history, 
                                      optimizer=self.optimizer, 
                                      lr_schedule=self.lr_schedule,
                                      loss=total_loss, training=False)
                    
            return progress_data
        
        if self.ema is not None:
            self.ema.updates = train_steps * initial_epoch
        
        # Throw everything into training mode.
        self.model.train()
        for metric_key, metric_value in self.metrics.items():
            if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                for metric in metric_value.values():
                    metric.train()
            else: # If it is a metric object, simply apply the metric.
                metric_value.train()
           
        # Set the learning rate_schedule to the correct step.
        if self.lr_schedule is not None:
            self.history['Learning_Rate'] = [self.lr_schedule.get_last_lr()[-1]]
            print(f"\nLearning rate is initialized to: {self.history['Learning_Rate'][-1]:.8f}\n")
            
        if apply_fn is not None:
            apply_fn(self.model, verbose=1)
            
        train_bar = Progress_Bar(int(np.ceil(train_steps/batch_size)), "Training",
                                 end_msg="Training epoch complete!")
        
        if valid_dataset is not None:
            valid_bar = Progress_Bar(int(np.ceil(valid_steps/valid_batch_size)), "Validating",
                                     end_msg="Validation epoch complete!")
        
        grad_scalar = torch.cuda.amp.GradScaler()
         
        for metric_key, metric_value in self.metrics.items():
            if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                for metric in metric_value.values():
                    metric.accumulate_limit = loss_acc_accumulate
            else: # If it is a metric object, simply apply the metric.
                metric_value.accumulate_limit = loss_acc_accumulate
        
        if valid_dataset is None:
            print("\nTraining for {0} steps with batch size {1}.".format(int(np.ceil(train_steps/batch_size)), batch_size))
        else:
            print("\nTraining for {0} steps with batch size {1}.\nValidating for {2} steps with valid batch size {3}.".format(int(np.ceil(train_steps/batch_size)), 
                                                                                                                              batch_size, 
                                                                                                                              int(np.ceil(valid_steps/valid_batch_size)), 
                                                                                                                              valid_batch_size))
          
        for epoch in range(initial_epoch, epochs):
            
            # !!! This is only done for this particular problem.
            train_dataset.samples.update_epoch(epoch) # Update the dataloader's registered epoch so we can shut off mosaic and mixup in later iterations.
            self.loss.update_epoch(epoch) # This will only work for this kaggle competition.
    
            print("\nEpoch %d/%d" % (epoch+1, epochs))

            train_progress_data = {}
            train_bar.reset_progress()
            train_bar.start_progress()
            
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset.data):
 
                if step < train_steps:
                    train_progress_data = train_step(x_batch_train, 
                                                     y_batch_train, step,
                                                     train_progress_data)
                else:
                    break
                
                if compute_metrics_on_train == True: # Turing this off can speed up training time. Especially for models with large number of classes.
                    for metric_key, metric_value in self.metrics.items():
                        if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                            for metric in metric_value.values():
                                train_progress_data[metric.name] = metric.result()
                        else: # If it is a metric object, simply apply the metric.
                            train_progress_data[metric_value.name] = metric_value.result()
    
                train_bar.update_progress(step+1, 
                                          new_display_data=train_progress_data)

            train_time = train_bar.finish_progress(verbose=False, 
                                                   get_duration=True)
                
            # Reset training metrics at the end of each epoch
            for metric_key, metric_value in self.metrics.items():
                if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                    for metric in metric_value.values():
                        metric.reset()
                else: # If it is a metric object, simply apply the metric.
                    metric_value.reset()
        
            if valid_dataset is not None:
                valid_progress_data = {}
                valid_bar.reset_progress()    
                valid_bar.start_progress()
                
                self.model.eval() # Throw model into eval and test mode.
                for metric_key, metric_value in self.metrics.items():
                    if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                        for metric in metric_value.values():
                            metric.test()
                    else: # If it is a metric object, simply apply the metric.
                        metric_value.test()
            
                with torch.no_grad(): # Disable gradient calculations for test performance.
                    # Run a validation loop at the end of each epoch.
                    for step, (x_batch_val, y_batch_val) in enumerate(valid_dataset.data):
                        
                        if step < valid_steps:
                            valid_progress_data = valid_step(x_batch_val, 
                                                             y_batch_val, step,
                                                             valid_progress_data)
    
                        else:
                            break
                        
                        # Update the metric results and reset states.
                        for metric_key, metric_value in self.metrics.items():
                            if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                                for metric in metric_value.values():
                                    valid_progress_data['Val_'+metric.name] = metric.result()
                            else: # If it is a metric object, simply apply the metric.
                                valid_progress_data['Val_'+metric_value.name] = metric_value.result()
                            
                        valid_bar.update_progress(step+1, 
                                                  new_display_data=valid_progress_data)
            
                valid_bar.finish_progress() # We don't want to print anymore. 
                
                for metric_key, metric_value in self.metrics.items():
                    if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                        for metric in metric_value.values():
                            metric.reset()
                    else: # If it is a metric object, simply apply the metric.
                        metric_value.reset()
                    
                # Throw everything into training mode.
                self.model.train()
                for metric_key, metric_value in self.metrics.items():
                    if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                        for metric in metric_value.values():
                            metric.train()
                    else: # If it is a metric object, simply apply the metric.
                        metric_value.train()
                        
                # Apply the initalisation function. Eg, freeze initial layers.
                if apply_fn is not None:
                    apply_fn(self.model)
            
            # Update the history.
            self.history['Epoch'].append(epoch)
            self.history['Epoch_Duration'].append(train_time)
            
            for key, value in train_progress_data.items():
                if key in self.history:
                    self.history[key].append(value)
                else:
                    self.history[key] = [value]
            for key, value in valid_progress_data.items():
                if key in self.history:
                    self.history[key].append(value)
                else:
                    self.history[key] = [value]
            if self.lr_schedule is not None:
                self.history['Learning_Rate'].append(self.lr_schedule.get_last_lr()[-1])
                print(f"Learning rate has been adjusted to: {self.history['Learning_Rate'][-1]:.8f}\n")
                    
            # Run the epoch end callbacks.
            for callback in callbacks_list:
                callback.on_epoch_end(self.model, optimizer=self.optimizer, 
                                      lr_schedule=self.lr_schedule,
                                      history=self.history)
                
        for callback in callbacks_list:
            callback.on_train_end(self.model, optimizer=self.optimizer, 
                                  lr_schedule=self.lr_schedule,
                                  history=self.history)
                
    def test(self, test_dataset, test_steps, batch_size=1, callbacks_list=[],
             visualizer=None):
        
        def test_step(x, y, step, progress_data={}):

            # This line of code places the data onto the gpu. This must be call inside the training step otherwise
            # the memory allocation remains meaning gradually GPU resources are eaten up.
            x = {key : torch.autograd.Variable(val.cuda()) for key, val in x.items()} # This sends the data to the GPU. We must make a new variable so that python can move it.
            y = {key : torch.autograd.Variable(val.cuda()) for key, val in y.items()} # This sends the data to the GPU. We must make a new variable so that python can move it.
        
            total_loss = 0
            losses = {}
            
            with torch.cuda.amp.autocast():
                logits = self.model(x)  # Logits for this minibatch (May have multiple outputs).
  
            # Compute the loss value for this minibatch.
            if self.loss_grouped == True:
                total_loss, losses = self.loss.forward(logits, y, total_loss, losses)
            elif type(self.loss) == dict:
                for loss_key, loss_obj in self.loss.items():
                    loss = loss_obj.forward(logits[loss_key], y[loss_key]) # Update the loss for a given output.
                    loss *= 1 if self.loss_weights is None else self.loss_weights[loss_key] # Apply the loss weights where appropriate.
                    losses[loss_key+'_Loss'] = loss.item() # Update the loss values where expected.
                    total_loss += loss # Add this to the total loss.
            else:
                for k in y:
                    total_loss = self.loss.forward(logits[k], y[k])
            losses['Loss'] = total_loss.item()
            
            for metric_key, metric_values in self.metrics.items(): # Make the metrics as a class that has update_state as a function.
                if type(metric_values) == dict:
                    for metric in metric_values.values():
                        metric.compute(logits[metric_key].detach() if torch.is_tensor(logits[metric_key]) else logits[metric_key], 
                                       y[metric_key].detach() if torch.is_tensor(y[metric_key]) else y[metric_key])
                else:
                    for k in y:
                        metric_values.compute(logits[k].detach() if torch.is_tensor(logits[k]) else logits[k], 
                                              y[k].detach() if torch.is_tensor(y[k]) else y[k])
                        
            if visualizer is not None:
                data_keys = visualizer.get_keys() # Get the keys of the data to grab.
                vis_data = {'Inputs' : {}, 'Logits' : {}, 'Targets' : {}}
                for key in data_keys:
                    if key in x.keys():
                        vis_data['Inputs'][key] = x[key].detach().cpu() if torch.is_tensor(x[key]) else x[key]
                    if key in logits.keys():
                        vis_data['Logits'][key] = logits[key].detach().cpu() if torch.is_tensor(logits[key]) else logits[key]
                    if key in y.keys():
                        vis_data['Targets'][key] = y[key].detach().cpu() if torch.is_tensor(y[key]) else y[key]   
                visualizer.on_test_end(vis_data, step)
                    
            for loss_key, loss_value in losses.items(): # Collect the loss values for display.
                progress_data[loss_key] = self.incremental_average(progress_data[loss_key] if step > 0 else 0, 
                                                                   loss_value, step+1)
                
            # Apply all on batch end callbacks here.
            for callback in callbacks_list:
                callback.on_batch_end(self.model, history=self.history)
                    
            return progress_data
        
        test_bar = Progress_Bar(int(np.ceil(test_steps/batch_size)), "Testing",
                                end_msg="Testing complete!")
        
        print("\nTesting for {0} steps with batch size {1}.".format(int(np.ceil(test_steps/batch_size)), batch_size))
        
        self.model.eval() # Throw model into eval and test mode.
        for metric_key, metric_value in self.metrics.items():
            if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                for metric in metric_value.values():
                    metric.test()
            else: # If it is a metric object, simply apply the metric.
                metric_value.test()
            
        progress_data = {}
        
        test_bar.reset_progress()
        test_bar.start_progress()
    
        with torch.no_grad(): # Disable gradient calculations for test performance.
            # Run a validation loop at the end of each epoch.
            for step, (x_batch_test, y_batch_test) in enumerate(test_dataset.data):
                
                if step < test_steps:
                    progress_data = test_step(x_batch_test, y_batch_test, 
                                              step, progress_data)

                else:
                    break
                
                # Update the metric results and reset states.
                for metric_key, metric_value in self.metrics.items():
                    if type(metric_value) == dict: # Check if we have multi-task or keyed metrics.
                        for metric in metric_value.values():
                            progress_data[metric.name] = metric.result()
                    else: # If it is a metric object, simply apply the metric.
                        progress_data[metric_value.name] = metric_value.result()
                    
                test_bar.update_progress(step+1, new_display_data=progress_data)
    
        test_bar.finish_progress() # We don't want to print anymore. 
        
        # for metric in self.metrics.values():
        #     metric.reset()
            
        # self.model.train() # Throw model into training mode.
        # for metric in self.metrics.values():
        #     metric.train()
            
        # Apply all on epoch end callbacks here.
        for callback in callbacks_list:
            callback.on_epoch_end(self.model, history=self.history)
                
    def incremental_average(self, x1, x2, n):

        return x1 + (x2-x1)/n