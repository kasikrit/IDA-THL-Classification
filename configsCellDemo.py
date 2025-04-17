#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:42:08 2024

@author: kasikritdamkliang
"""
import os, platform
from datetime import datetime 
from yacs.config import CfgNode as CN

# Define default configuration
_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

_C.BASE = str(platform.system())
if platform.system() == 'Windows':
    
    _C.BASEPATH = 'D:\\Thalas'  

if platform.system() == 'Darwin' or platform.system() == 'Linux':
    _C.BASEPATH = '.' 
    _C.DATA.SET = 'IDA-THL-Dataset-Phase-I'
    
_C.DATA.SAVEPATH = os.path.join(_C.BASEPATH, 'models')  
_C.DATA.C = 3
_C.DATA.W = _C.DATA.H = 256
_C.DATA.IMAGE_SIZE = _C.DATA.W
_C.DATA.DIMENSION = _C.DATA.W, _C.DATA.H, _C.DATA.C
_C.DATA.TEST_SIZE = 0.30
_C.DATA.CLASS_LABELS = ['IDA', 'THL']
_C.DATA.CLASSES = [0, 1]
_C.DATA.CLASSDICT = [(0, "IDA"), (1, "THL")]
_C.DATA.COLOR = 'BGR'
# _C.DATA.COLOR = 'RGB'

_C.MODEL = CN()
  
_C.MODEL.NAMES = [
    # 'AlexNet',      
    # 'vit_l32',
    # 'VGG16',
    # 'VGG19',                      
    # 'DenseNet121',
    # 'DenseNet201',
    'EfficientNetV2S',
    # 'MobileNet', 
    # 'MobileNetV2',      
    # 'InceptionResNetV2',                
    # 'InceptionV3',      
    # 'EfficientNetB7',
    # 'EfficientNetV2B3',                   
    # 'ViT_b32',
    # 'ViT_l16',
    # 'EfficientNetV2M',    
    # 'EfficientNetV2L',       
    # 'ConvNeXtTiny',
    # 'ConvNeXtSmall',
    # 'ConvNeXtBase',
    # 'ConvNeXtLarge',
    # 'ConvNeXtXLarge',    
    # 'ResNet50',
    # 'ResNet50V2',
    # 'Xception',
]   
   
# _C.MODEL.INFER_FILE_PATH = 'infer-ens-20250302-12THL.csv'
# _C.MODEL.EVA_FILE_PATH = 'Eva-ens-20250302'

_C.MODEL.NUM_CLASSES = 2

_C.TRAIN = CN()
_C.TRAIN.Enable = True
_C.TRAIN.DATETIME = datetime.now().strftime("%Y%m%d-%H%M")
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR = 1e-4
_C.TRAIN.LR_FACTOR = 0.5
_C.TRAIN.EPOCHS = 120
_C.TRAIN.DROPOUT = 0.4
_C.TRAIN.reduceLR_patience = 2
_C.TRAIN.stop_patience = 12
_C.TRAIN.VERBOSE = 2

# For ViT base
_C.TRAIN.ACTIVATION = 'gelu'

_C.DATA.VERIFY = False
# _C.DATA.VERIFY = True

# _C.TRAIN.SAMPLE = False
_C.TRAIN.SAMPLE = True
if _C.TRAIN.SAMPLE:
    _C.TRAIN.SAMPLE_SIZE = 12
    _C.TRAIN.SAMPLE_SIZE_VAL = 0.2
    _C.TRAIN.BATCH_SIZE = 1
    _C.TRAIN.EPOCHS = 2

_C.TRAIN.Evaluate = True
_C.SAVE = True
# _C.SAVE = False
_C.SAVEHIST = True
    
def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    #update_config(config, args)

    return config

