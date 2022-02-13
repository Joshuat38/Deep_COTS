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
from torchvision import transforms

# Custom imports
from models import YOLOX
from utils import yolox_post_process
from cots import Preprocess, Normalize, ToTensor

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
    
def get_inference_model(): # This is a custom testing function. It ensures that the model averages out the total results.

    inference_model = YOLOX_Inference(args, cfg)
    
    return inference_model

class YOLOX_Inference:
    
    def __init__(self, args, cfg):
        
        print("\nBuilding inference model...\n")
                
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
        
        self.model = model
        
        self.composed_transforms = transforms.Compose([Preprocess(size=cfg['test']['process_size'], 
                                                                  max_boxes=cfg['model']['num_classes']),
                                                       Normalize(mean=(0.485, 0.456, 0.406), 
                                                                 std=(0.229, 0.224, 0.225),
                                                                 rescale=255.0),
                                                       ToTensor()])
        
        self.post_process_kwargs = {'down_strides' : cfg['model']['strides'], 
                                    'num_classes' : cfg['model']['num_classes'], 
                                    'conf_thre' : cfg['model']['score_threshold'], 
                                    'nms_thre' : cfg['model']['nms_threshold'], 
                                    'label_name' : ['COTS']}
        
    def postprocess(self, y_pred, img_size, pred_size):
        """
        :image_size - size to adjust the bboxes to.
        :pred_size - the size the image is reshaped to.
        """

        ih, iw = pred_size
        h, w = img_size
    
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        
        # Adjust to the new input shape.
        y_pred[:, [0, 2]] = np.clip((y_pred[:, [0, 2]] - dw) / scale, 0., w)
        y_pred[:, [1, 3]] = np.clip((y_pred[:, [1, 3]] - dh) / scale, 0., h)

        return y_pred
        
    def predict(self, image):
        
        with torch.no_grad():
            
            img_h, img_w, _ = image.shape
            sample = {'image': image, 'annotations': []}
            sample = self.composed_transforms(sample)
            inputs = {'image': sample['image']}
            
            cuda_inputs = {key : val.unsqueeze(0) if torch.is_tensor(val) else torch.tensor(val).unsqueeze(0) for key, val in inputs.items()}
            
            outputs = self.model(cuda_inputs)
                
            imgs = inputs['image'].permute(0, 2, 3, 1).numpy()
            img_shape = [[int(imgs[i].shape[0]), int(imgs[i].shape[1])] for i in range(imgs.shape[0])]
            
            outputs = yolox_post_process(outputs=outputs['annotations'].detach().cpu(), 
                                         img_shape=img_shape, **self.post_process_kwargs)
            
            outputs = np.array(outputs[0]).astype(np.float32) # Get the prediction out of the batch format.
            
            outputs = self.postprocess(outputs, img_size=[img_h, img_w], 
                                           pred_size=cfg['test']['process_size']) # Reshape the bboxes to the correct size.
        
        return outputs
        
