# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from components import BaseConv, DWConv, Focus, CSPLayer, SPPBottleneck, Adjustment_Layer

class YOLOX(nn.Module):
    def __init__(self, args, cfg):
        super(YOLOX, self).__init__()

        channels = [256, 512, 1024]
        
        depth = 1.33
        width = 1.25
        depth_wise = False
        # self.encoder = CSPDarknet(dep_mul=depth, wid_mul=width, 
        #                           out_indices=(3, 4, 5), depthwise=depth_wise)
        self.encoder = Encoder(args, cfg)
        
        # define neck
        self.neck = YOLOXPAFPN(depth=depth, width=width, in_channels=channels, 
                               feature_channels=self.encoder.feat_out_channels[2:],
                               depthwise=depth_wise)
        # define head
        self.head = YOLOXHead(num_classes= cfg['model']['num_classes'], 
                              reid_dim=0, width=width, in_channels=channels, 
                              depthwise=depth_wise)

        # self.encoder._init_weights()
        # self.neck._init_weights()
        # self.head._init_weights()

    def forward(self, inputs):#, targets=None, show_time=False):
        # with torch.cuda.amp.autocast(enabled=self.opt.use_amp):
        #     if show_time:
        #         s1 = sync_time(inputs)
        
        image = inputs['image']

        body_feats = self.encoder(image)
        neck_feats = self.neck(body_feats)
        yolo_outputs = self.head(neck_feats)
        # print('yolo_outputs:', [[i.shape, i.dtype, i.device] for i in yolo_outputs])  # float16 when use_amp=True

        # if show_time:
        #     s2 = sync_time(inputs)
        #     print("[inference] batch={} time: {}s".format("x".join([str(i) for i in inputs.shape]), s2 - s1))

        # if targets is not None:
        #     loss = self.loss(yolo_outputs, targets)
        #     # for k, v in loss.items():
        #     #     print(k, v, v.dtype, v.device)  # always float32

        # if targets is not None:
        #     return yolo_outputs, loss
        # else:
            
        return {'annotations' : yolo_outputs}
    
    def get_1x_lr_params(self):  # lr/10 learning rate
        modules = [self.encoder]
        for m in modules:
            yield from m.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.neck, self.head]
        for m in modules:
            yield from m.parameters()

class YOLOXPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_channels=[256, 512, 1024], 
                 feature_channels=[256, 512, 1024], depthwise=False, 
                 act="silu"):
        super().__init__()
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(feature_channels[2], 
                                      int(in_channels[1] * width), 1, 1, 
                                      act=act)
        self.C3_p4 = CSPLayer(feature_channels[1] + int(in_channels[1] * width),
                              int(in_channels[1] * width), round(3 * depth),
                              False, depthwise=depthwise, act=act)  # cat

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), 
                                     int(in_channels[0] * width), 1, 1, 
                                     act=act)
        self.C3_p3 = CSPLayer(feature_channels[0] + int(in_channels[0] * width),
                              int(in_channels[0] * width), round(3 * depth),
                              False, depthwise=depthwise, act=act)

        # bottom-up conv
        self.bu_conv2 = Conv(int(in_channels[0] * width), 
                             int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(int(2 * in_channels[0] * width),
                              int(in_channels[1] * width), round(3 * depth),
                              False, depthwise=depthwise, act=act)

        # bottom-up conv
        self.bu_conv1 = Conv(int(in_channels[1] * width), 
                             int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(int(2 * in_channels[1] * width),
                              int(in_channels[2] * width), round(3 * depth),
                              False, depthwise=depthwise, act=act)
        
        self._init_weights()

    def forward(self, blocks):
        assert len(blocks) == len(self.in_channels)
        [x2, x1, x0] = blocks

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

class YOLOXHead(nn.Module):
    def __init__(self, num_classes=80, reid_dim=0, width=1.0, 
                 in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.reid_dim = reid_dim
        Conv = DWConv if depthwise else BaseConv

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        if self.reid_dim > 0:
            self.reid_preds = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width),
                         out_channels=int(256 * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width),
                                                       out_channels=int(256 * width),
                                                       ksize=3, stride=1, 
                                                       act=act),
                                                  Conv(in_channels=int(256 * width),
                                                       out_channels=int(256 * width),
                                                       ksize=3, stride=1, 
                                                       act=act)]))
            self.reg_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width),
                                                       out_channels=int(256 * width),
                                                       ksize=3, stride=1, 
                                                       act=act),
                                                  Conv(in_channels=int(256 * width),
                                                       out_channels=int(256 * width),
                                                       ksize=3, stride=1,
                                                       act=act)]))
            self.cls_preds.append(nn.Conv2d(in_channels=int(256 * width),
                                            out_channels=self.n_anchors * self.num_classes,
                                            kernel_size=1, stride=1, padding=0))
            self.reg_preds.append(nn.Conv2d(in_channels=int(256 * width),
                                            out_channels=4, kernel_size=1,
                                            stride=1, padding=0))
            self.obj_preds.append(nn.Conv2d(in_channels=int(256 * width),
                                            out_channels=self.n_anchors * 1,
                                            kernel_size=1, stride=1,
                                            padding=0))
            if self.reid_dim > 0:
                self.reid_preds.append(nn.Conv2d(in_channels=int(256 * width),
                                                 out_channels=self.reid_dim,
                                                 kernel_size=1, stride=1,
                                                 padding=0))
                
        self._init_weights(prior_prob=1e-2)

    def forward(self, feats):
        outputs = []
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, feats)):
            x = self.stems[k](x)

            # classify
            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)

            # regress, object, (reid)
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            if self.reid_dim > 0:
                reid_output = self.reid_preds[k](reg_feat)
                output = torch.cat([reg_output, obj_output, cls_output, reid_output], 1)
            else:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)

        return outputs
    
    def _init_weights(self, prior_prob=1e-2):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-np.math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-np.math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
class CSPDarknet(nn.Module):

    def __init__(self, dep_mul=1., wid_mul=1., out_indices=(3, 4, 5), depthwise=False, act="silu"):
        super().__init__()

        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(Conv(base_channels, base_channels * 2, 3, 2, act=act),
                                   CSPLayer(base_channels * 2, base_channels * 2,
                                            n=base_depth, depthwise=depthwise, act=act))

        # dark3
        self.dark3 = nn.Sequential(Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
                                   CSPLayer(base_channels * 4, base_channels * 4,
                                            n=base_depth * 3, depthwise=depthwise, act=act))

        # dark4
        self.dark4 = nn.Sequential(Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
                                   CSPLayer(base_channels * 8, base_channels * 8,
                                            n=base_depth * 3, depthwise=depthwise, act=act))

        # dark5
        self.dark5 = nn.Sequential(Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
                                   SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
                                   CSPLayer(base_channels * 16, base_channels * 16, n=base_depth,
                                            shortcut=False, depthwise=depthwise, act=act))
        
        self.feat_out_channels = [int(base_channels), int(base_channels*2), 
                                  int(base_channels*4), int(base_channels*8), 
                                  int(base_channels*16)]
        
        self._init_weights()

    def forward(self, x):
        outputs = []
        for idx, layer in enumerate([self.stem, self.dark2, self.dark3, self.dark4, self.dark5]):
            x = layer(x)

            # if idx + 1 in self.out_features:
            outputs.append(x)
        return outputs
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
    
class Encoder(nn.Module):
    def __init__(self, args, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        
        if cfg['model']['encoder'] == 'vit_base_patch16_384':
            self.base_model = timm.create_model(cfg['model']['encoder'], pretrained=False)
            self.feat_names = ['2', '5', '8', '11']
            self.feat_out_channels = [0, 768, 768, 768, 768]
            self.input_shape = [384, 384]
            
        elif cfg['model']['encoder'] == 'vit_large_patch16_384':
            self.base_model = timm.create_model(cfg['model']['encoder'], pretrained=False)
            self.feat_names = ['4', '11', '17', '23']
            self.feat_out_channels = [0, 1024, 1024, 1024, 1024]
            self.input_shape = [384, 384]
            
        elif cfg['model']['encoder'] == 'vit_base_resnet50_384':
            self.base_model = timm.create_model(cfg['model']['encoder'], pretrained=False)
            self.feat_names = ['stem.norm', 'stages.0', 'stages.1', '8', '11']
            self.feat_out_channels = [64, 256, 512, 768, 768]
            self.input_shape = [384, 384]
            
        elif cfg['model']['encoder'] == 'swin_base_patch4_window12_384':
            self.base_model = timm.create_model(cfg['model']['encoder'], 
                                                pretrained=False)
            self.feat_names = ['0', '1', '2', '3']
            self.feat_out_channels = [0, 256, 512, 1024, 1024]
            self.input_shape = [384, 384]
            self.adjust32 = Adjustment_Layer(in_channels=self.feat_out_channels[4], 
                                             input_resolution=self.input_shape, 
                                             feature_scale=32, resize_factor=1) # Recover image features, convert to H/32.
            self.adjust16 = Adjustment_Layer(in_channels=self.feat_out_channels[3], 
                                             input_resolution=self.input_shape, 
                                             feature_scale=32, resize_factor=2) # Recover image features, convert to H/16.
            self.adjust8 = Adjustment_Layer(in_channels=self.feat_out_channels[2], 
                                            input_resolution=self.input_shape, 
                                            feature_scale=16, resize_factor=2) # Recover image features, convert to H/8.
            
        elif cfg['model']['encoder'] == 'swin_large_patch4_window12_384':
            self.base_model = timm.create_model(cfg['model']['encoder'],
                                                pretrained=False)
            self.feat_names = ['0', '1', '2', '3']
            self.feat_out_channels = [0, 384, 768, 1536, 1536]
            self.input_shape = [384, 384]
            self.adjust32 = Adjustment_Layer(in_channels=self.feat_out_channels[4], 
                                             input_resolution=self.input_shape, 
                                             feature_scale=32, resize_factor=1) # Recover image features, convert to H/32.
            self.adjust16 = Adjustment_Layer(in_channels=self.feat_out_channels[3], 
                                             input_resolution=self.input_shape, 
                                             feature_scale=32, resize_factor=2) # Recover image features, convert to H/16.
            self.adjust8 = Adjustment_Layer(in_channels=self.feat_out_channels[2], 
                                            input_resolution=self.input_shape, 
                                            feature_scale=16, resize_factor=2) # Recover image features, convert to H/8.
            
        elif cfg['model']['encoder'] == 'efficientnet_b0':
            self.base_model = timm.create_model(model_name='tf_efficientnet_b0_ap', 
                                                features_only=True,
                                                pretrained=False)
            self.feat_out_channels = [16, 24, 40, 112, 320]
            self.base_model.global_pool = nn.Identity()
            self.base_model.classifier = nn.Identity()
            self.input_shape = None
        
        elif cfg['model']['encoder'] == 'efficientnet_b4':
            self.base_model = timm.create_model(model_name='tf_efficientnet_b4_ap', 
                                           features_only=True,
                                           pretrained=False)
            self.feat_out_channels = [24, 32, 56, 160, 448]
            self.base_model.global_pool = nn.Identity()
            self.base_model.classifier = nn.Identity()
            self.input_shape = None
            
        elif cfg['model']['encoder'] == 'efficientnet_b5':
            self.base_model = timm.create_model(model_name='tf_efficientnet_b5_ap', 
                                                features_only=True,
                                                pretrained=False)
            self.feat_out_channels = [24, 40, 64, 176, 512]
            self.base_model.global_pool = nn.Identity()
            self.base_model.classifier = nn.Identity()
            self.input_shape = None
            
        elif cfg['model']['encoder'] == 'efficientnet_b7':
            self.base_model = timm.create_model(model_name='tf_efficientnet_b7_ap', 
                                                features_only=True,
                                                pretrained=False)
            self.feat_out_channels = [32, 48, 80, 224, 640]
            self.base_model.global_pool = nn.Identity()
            self.base_model.classifier = nn.Identity()
            self.input_shape = None

        elif cfg['model']['encoder'] == 'resnet101':
            self.base_model = timm.create_model(model_name='resnet101d',
                                                features_only=True,
                                                pretrained=False)
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
            self.base_model.global_pool = nn.Identity()
            self.base_model.fc = nn.Identity()
            self.input_shape = None
            
        elif cfg['model']['encoder'] == 'resnet50':
            self.base_model = timm.create_model(model_name='resnet50d',
                                                features_only=True,
                                                pretrained=False)
            self.stages = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
            self.base_model.global_pool = nn.Identity()
            self.base_model.fc = nn.Identity()
            self.input_shape = None
            
        elif cfg['model']['encoder'] == 'resnest101':
            self.base_model = timm.create_model(model_name='resnest101e',
                                                features_only=True,
                                                pretrained=False)
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
            self.base_model.global_pool = nn.Identity()
            self.base_model.fc = nn.Identity()
            self.input_shape = None

        elif cfg['model']['encoder'] == 'hrnet64':
            self.base_model = timm.create_model('hrnet_w64', 
                                                features_only=True, 
                                                pretrained=False)
            self.feat_out_channels = [64, 128, 256, 512, 1024]
            self.input_shape = None
            
        elif cfg['model']['encoder'] == 'darknet53':
            self.base_model = CSPDarknet(dep_mul=1.33, wid_mul=1.25, 
                                         depthwise=False, act="silu")
            self.feat_out_channels = self.base_model.feat_out_channels
            self.input_shape = None
            
        else:
            raise NotImplementedError(f"{cfg['model']['encoder']} is not a supported encoder!")

    def forward(self, x):
        
        skip_feat = []
        feature = x
            
        if 'vit' in self.cfg['model']['encoder']:
            # This is a very hacky but effective method to extract features from the vit models.
            for k, v in self.base_model._modules.items():
                if k == 'blocks':
                    for ki, vi in v._modules.items():
                        feature = vi(feature)
                        if ki in self.feat_names:
                            skip_feat.append(feature)
                            
                elif k == 'patch_embed':
                    for ki, vi in v._modules.items():
                        if ki == 'backbone':
                            for kj, vj in vi._modules.items():
                                if kj == 'stem' or kj == 'stages':
                                    for kk, vk in vj._modules.items():
                                        feature = vk(feature)
                                        if kj+'.'+kk in self.feat_names:
                                            skip_feat.append(feature)
                                else:
                                    feature = vj(feature)
                        elif ki == 'proj': 
                            feature = vi(feature).flatten(2).transpose(1, 2) # Hacky way to extract features. Use hooks in future.
                        else:
                            feature = vi(feature)
                        
                else:
                    feature = v(feature)
                    if k in self.feat_names:
                        skip_feat.append(feature)
                        
        elif 'swin' in self.cfg['model']['encoder']:
            orig_size = x.size()[2:]
            feature = F.interpolate(feature, self.input_shape,  mode='bilinear')
            skip_feat.append(x)
            # This is a very hacky but effective method to extract features from the swin models.
            for k, v in self.base_model._modules.items():
                if k == 'layers':
                    for ki, vi in v._modules.items():
                        feature = vi(feature)
                        if ki in self.feat_names:
                            skip_feat.append(feature)
                elif k == 'norm':
                    feature = v(feature).transpose(1, 2) # Hacky way to extract features. Use hooks in future.
                elif k == 'avgpool': 
                    feature = v(feature)
                    feature = torch.flatten(feature, 1) # Hacky way to extract features. Use hooks in future.
                else:
                    feature = v(feature)
                    if k in self.feat_names:
                        skip_feat.append(feature)
                        
            for s, skip in enumerate(skip_feat):
                if s == 2:
                    tmp_skip = self.adjust8(skip)
                    skip_feat[2] = F.interpolate(tmp_skip, [orig_size[0]//8,
                                                            orig_size[1]//8],  
                                                 mode='bilinear') # Resize to the original expected size.
                    # skip_feat[2] = self.adjust8(skip)
                elif s == 3:
                    tmp_skip = self.adjust16(skip)
                    skip_feat[3] = F.interpolate(tmp_skip, [orig_size[0]//16,
                                                            orig_size[1]//16],  
                                                 mode='bilinear') # Resize to the original expected size.
                    # skip_feat[3] = self.adjust16(skip)
                elif s == 4:
                    tmp_skip = self.adjust32(skip)
                    skip_feat[4] = F.interpolate(tmp_skip, [orig_size[0]//32,
                                                            orig_size[1]//32],  
                                                 mode='bilinear') # Resize to the original expected size.
                    # skip_feat[4] = self.adjust32(skip)
                        
        else:
            # This method catches the hrnet or efficinet features at the end of the blocks as is the conventional way.
            skip_feat = self.base_model(feature)
                
        return skip_feat[2:]