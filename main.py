# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

# Built-in imports
import os
import argparse
import sys
import re
import yaml

# Third party imports
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# Custom imports
from models import YOLOX
from losses import YOLOXLoss
from metrics import BboxIOU, F2_Score

import cots
from utils import Visualize, WarmupCosLR
from PytorchTrainer import callbacks as cb
from PytorchTrainer.lr_schedulers import PolynomialLRDecay
from PytorchTrainer.pytorch_model import PytorchModel

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
parser.add_argument('--mode',             type=str,   help='train or test', default='test')
parser.add_argument('--config',           type=str,   help='path and filename of config file to use', default='yolox_config.yaml')
parser.add_argument('--batch_size',       type=int,   help='batch size', default=1)
parser.add_argument('--valid_batch_size', type=int,   help='validation batch size', default=1)
parser.add_argument('--num_epochs',       type=int,   help='number of epochs', default=50)
parser.add_argument('--num_gpus',         type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',      type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--save_directory',   type=str,   help='directory to save checkpoints and summaries', default='./models')
parser.add_argument('--pretrained_model', type=str,   help='path to a pretrained model checkpoint to load', default='None')
parser.add_argument('--initial_epoch',    type=int,   help='if used with pretrained_model, will start from this epoch', default=0)
parser.add_argument('--gpu_id',           type=str,   help='specifies the gpu to use', default='0')

if sys.argv.__len__() == 2: # This handls prefixes.
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()
     
if args.num_gpus == 1:
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id # Use the specified gpu and ignore all others.
    
with open(args.config, 'r') as yaml_file:
    cfg = yaml.load(yaml_file, Loader=custom_loader)
    cfg['mode'] = args.mode
    cfg['train']['batch_size'] = args.batch_size
    cfg['train']['num_epochs'] = args.num_epochs
    
def train(): # This is the training function.
    
    print("\nRunning Training...\n")
    
    fig_dir = args.save_directory+'/'+cfg['model']['encoder']+'/results'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Create the dataloader objects.
    train_data = cots.COTSDataLoader(args, cfg, mode='train', verbose=1)
    
    if cfg['valid']['data_file'] is not None:
        valid_data = cots.COTSDataLoader(args, cfg, mode='valid', verbose=1)
        
    # Create model
    model = YOLOX(args, cfg)
    model.zero_grad()
    model.train() # Puts the model in training mode.
    
    print("\nModel Initialized on GPU: {}, with device id: {}".format(args.gpu_id, torch.cuda.current_device()))
    
    # Setup optimizers.
    if cfg['train']['optimizer'] == 'adamW':
        optimizer = torch.optim.AdamW([{'params' : model.get_1x_lr_params(), 'lr' : cfg['train']['max_learning_rate']},
                                       {'params' : model.get_10x_lr_params(), 'lr' : cfg['train']['max_learning_rate']}],
                                      eps=cfg['train']['adam_epsilon'], weight_decay=cfg['model']['weight_decay'])
    elif cfg['train']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD([{'params' : model.get_1x_lr_params(), 'lr' : cfg['train']['max_learning_rate']},
                                     {'params' : model.get_10x_lr_params(), 'lr' : cfg['train']['max_learning_rate']}],
                                    momentum=0.9,  weight_decay=cfg['model']['weight_decay'], nesterov=True)
    else:
        raise NotImplementedError(f"{cfg['train']['optimizer']} optimizer is not implemented!")
    
    # This allows for the model to be specified on GPU.
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:1234', world_size=1, rank=0) # This is here only such that the model will load. I have no idea yet how distributed computing works.
    
    if args.gpu_id != '-1': 
        torch.cuda.set_device('cuda:0')
        model.cuda('cuda:0')
        args.batch_size = args.batch_size // args.num_gpus
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=['cuda:0'], find_unused_parameters=True)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    # Setup the LR schedulars.
    if cfg['train']['lr_schedular'] == 'one_cycle':
        lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                          cfg['train']['max_learning_rate'], 
                                                          epochs=args.num_epochs, 
                                                          steps_per_epoch=int(np.ceil(len(train_data.samples)/args.batch_size)),
                                                          pct_start=0.3,
                                                          cycle_momentum=True,
                                                          base_momentum=0.85, 
                                                          max_momentum=0.95, 
                                                          last_epoch=-1,
                                                          div_factor=cfg['train']['lr_divide_factor'],
                                                          final_div_factor=cfg['train']['final_lr_divide_factor'])
    
    elif cfg['train']['lr_schedular'] == 'poly':
        lr_schedule = PolynomialLRDecay(optimizer, 
                                        int(np.ceil((args.num_epochs*len(train_data.samples)) // args.batch_size)),
                                        end_learning_rate=cfg['train']['min_learning_rate'], 
                                        power=0.9)
        
    elif cfg['train']['lr_schedular'] == 'warmup_cos':
        lr_schedule = WarmupCosLR(optimizer, 
                                  total_steps=int(np.ceil((args.num_epochs*len(train_data.samples)) // args.batch_size)), 
                                  warmup_steps=int(np.ceil((5*len(train_data.samples)) // args.batch_size)), 
                                  min_lr_ratio=0.05, 
                                  warmup_init_lr=0, 
                                  no_aug_steps=int(np.ceil((15*len(train_data.samples)) // args.batch_size)))
        
    else:
        raise NotImplementedError(f"{cfg['train']['lr_schedular']} learning rate schedular is not implemented!")
    
    # This is where the checkpoints are loaded. This must be done later as it depends on the optimiser format.
    history = None
    if args.pretrained_model != 'None':
        if os.path.isfile(args.pretrained_model):
            print("Loading checkpoint '{}'".format(args.pretrained_model))
            if args.gpu_id != '-1':
                checkpoint = torch.load(args.pretrained_model)
            else:
                loc = 'cuda:{}'.format(args.gpu_id)
                checkpoint = torch.load(args.pretrained_model, map_location=loc)

            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_schedule' in checkpoint:
                lr_schedule.load_state_dict(checkpoint['lr_schedule'])
            if 'history' in checkpoint: # Allows history to be loaded meaning progress data is preserved.
                history = checkpoint['history']
            print("Loaded checkpoint '{}' (Epoch {})".format(args.pretrained_model, epoch))
        else:
            print("No checkpoint found at '{}'".format(args.pretrained_model))
            
        if args.initial_epoch == 0:
            args.initial_epoch = epoch

    # Turn on cudnn benchmarking for faster performance.
    cudnn.benchmark = True

    # Create the callbacks.
    callbacks_list = []
    
    checkpoint_callback = cb.ModelCheckpoint(save_path=args.save_directory + '/' + cfg['model']['encoder'] + '/model_checkpoint', 
                                             monitor='Val_Loss', verbose=1, 
                                             save_best_only=True,  mode='auto', 
                                             save_freq='epoch', save_history=True)
    callbacks_list.append(checkpoint_callback)
    
    nan_stop = cb.TerminateOnNanOrInf()
    callbacks_list.append(nan_stop)
    
    # Create the visualiser callback.
    visualizer = Visualize(classes=[x for x in range(cfg['model']['num_classes'])], 
                            vis_keys=['image', 'annotations'],
                            train_figure_size=cfg['train']['process_size'], 
                            test_figure_size=cfg['valid']['base_size'],
                            show_results=True, save_dir=None, save_test_images=False, 
                            save_train_images=False,  img_mean=[0.485, 0.456, 0.406], 
                            img_std=[0.229, 0.224, 0.225], img_rescale=255.0, 
                            num_steps=10, show_on_train=True,
                            **{'down_strides' : cfg['model']['strides'], 
                               'num_classes' : cfg['model']['num_classes'], 
                               'conf_thre' : cfg['model']['score_threshold'], 
                               'nms_thre' : cfg['model']['nms_threshold'], 
                               'label_name' : ['COTS']})
    # visualizer = None # Clear the visualiser.
    
    # Creates a dictionary of the metrics functions to be used.
    metrics = {'BBox_IOU' : BboxIOU(ignore_thresh=cfg['model']['iou_threshold'], 
                                    down_strides=cfg['model']['strides'], 
                                    num_classes=cfg['model']['num_classes'], 
                                    conf_thresh=cfg['model']['score_threshold'], 
                                    nms_thresh=cfg['model']['nms_threshold'], 
                                    label_name=['COTS'], 
                                    img_shape=cfg['valid']['process_size'], 
                                    name='BboxIOU'),
               'F2_Score' : F2_Score(iou_thresholds=np.linspace(0.3, 0.8, 11),
                                     down_strides=cfg['model']['strides'], 
                                     num_classes=cfg['model']['num_classes'], 
                                     conf_thresh=cfg['model']['score_threshold'], 
                                     nms_thresh=cfg['model']['nms_threshold'], 
                                     label_name=['COTS'], 
                                     img_shape=cfg['valid']['process_size'], 
                                     name='F2_Score')}

    loss = YOLOXLoss(['COTS'], reid_dim=0, id_nums=None, 
                     strides=cfg['model']['strides'], 
                     in_channels=[256, 512, 1024], 
                     use_l1_epoch=args.num_epochs-cfg['train']['disable_mosaic_epochs'],
                     verbose=1)

    # Pass everything to the NNB pytorch model container.
    pytorch_model = PytorchModel(model, optimizer, loss=loss, 
                                 loss_grouped=True, metrics=metrics, 
                                 lr_schedule=lr_schedule, history=history, 
                                 model_name='YoloX', use_ema=True)
    
    pytorch_model.print_summary(show_model=False)
    
    # Using the NNB pytorch container, train using the simple pre-defined 
    # training loop for ease of use.
    pytorch_model.fit(epochs=args.num_epochs, batch_size=args.batch_size,
                      train_dataset=train_data, 
                      train_steps=len(train_data.samples),
                      initial_epoch=args.initial_epoch,
                      valid_batch_size=args.valid_batch_size,
                      valid_dataset=valid_data, 
                      valid_steps=len(valid_data.samples),
                      callbacks_list=callbacks_list, visualizer=visualizer,
                      apply_fn=None, compute_metrics_on_train=True)
    
    print("\nTraining Complete!\n")
    
def test(): # This is a custom testing function. It ensures that the model averages out the total results.

    print("\nRunning Testing...\n")
    
    # !!! Artificially pass the pre-trained model here!
    args.pretrained_model = args.save_directory + '/' + cfg['model']['encoder'] + '/model_checkpoint'
    
    fig_dir = args.save_directory+'/'+cfg['model']['encoder']+'/results'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    # Create the dataloader objects.
    test_data = cots.COTSDataLoader(args, cfg, mode='test', verbose=0)
            
    # Create model
    model = YOLOX(args, cfg)
    model.zero_grad()
    model.eval() # Puts the model in training mode.
    # set_misc(model)
    
    # This allows for the model to be specified on GPU
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:2334', world_size=1, rank=0) # This is here only such that the model will load. I have no idea yet how distributed computing works.
    
    if args.gpu_id != '-1': 
        torch.cuda.set_device('cuda:0')
        model.cuda('cuda:0')
        args.batch_size = args.batch_size // args.num_gpus
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=['cuda:0'], find_unused_parameters=True)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    print("Model Initialized on GPU: {}, with device id: {}".format(args.gpu_id, torch.cuda.current_device()))
    
    # This is where the checkpoints are loaded. This must be done later as it depends on the optimiser format.
    if args.pretrained_model != 'None':
        if os.path.isfile(args.pretrained_model):
            print("Loading checkpoint '{}'".format(args.pretrained_model))
            if args.gpu_id != '-1':
                checkpoint = torch.load(args.pretrained_model)
            else:
                loc = 'cuda:{}'.format(args.gpu_id)
                checkpoint = torch.load(args.pretrained_model, map_location=loc)

            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            
            print("Loaded checkpoint '{}' (Epoch {})".format(args.pretrained_model, epoch))
        else:
            print("No checkpoint found at '{}'".format(args.pretrained_model))

    cudnn.benchmark = True
    
            
    # Create the visualiser callback.
    visualizer = Visualize(classes=[x for x in range(cfg['model']['num_classes'])], 
                            vis_keys=['image', 'annotations'],
                            train_figure_size=cfg['train']['process_size'], 
                            test_figure_size=cfg['valid']['base_size'],
                            show_results=True, save_dir=None, save_test_images=False, 
                            save_train_images=False,  img_mean=[0.485, 0.456, 0.406], 
                            img_std=[0.229, 0.224, 0.225], img_rescale=255.0, 
                            num_steps=10, show_on_train=False,
                            **{'down_strides' : cfg['model']['strides'], 
                               'num_classes' : cfg['model']['num_classes'], 
                               'conf_thre' : cfg['model']['score_threshold'], 
                               'nms_thre' : cfg['model']['nms_threshold'], 
                               'label_name' : ['COTS']})
    # visualizer = None # Clear the visualiser.
    
    # Creates a dictionary of the metrics functions to be used.
    metrics = {'BBox_IOU' : BboxIOU(ignore_thresh=cfg['model']['iou_threshold'], 
                                    down_strides=cfg['model']['strides'], 
                                    num_classes=cfg['model']['num_classes'], 
                                    conf_thresh=cfg['model']['score_threshold'], 
                                    nms_thresh=cfg['model']['nms_threshold'], 
                                    label_name=['COTS'], 
                                    img_shape=cfg['valid']['process_size'], 
                                    name='BboxIOU'),
               'F2_Score' : F2_Score(iou_thresholds=np.linspace(0.3, 0.8, 11),
                                     down_strides=cfg['model']['strides'], 
                                     num_classes=cfg['model']['num_classes'], 
                                     conf_thresh=cfg['model']['score_threshold'], 
                                     nms_thresh=cfg['model']['nms_threshold'], 
                                     label_name=['COTS'], 
                                     img_shape=cfg['valid']['process_size'], 
                                     name='F2_Score')}

    loss = YOLOXLoss(['COTS'], reid_dim=0, id_nums=None, 
                     strides=cfg['model']['strides'], 
                     in_channels=[256, 512, 1024])

    pytorch_model = PytorchModel(model, loss=loss, loss_grouped=True, 
                                 metrics=metrics, model_name='YoloX')
    
    pytorch_model.print_summary(show_model=False)
    
    pytorch_model.test(test_dataset=test_data, test_steps=len(test_data.samples),
                        batch_size=args.batch_size, visualizer=visualizer)
    
if __name__ == '__main__':
    
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
