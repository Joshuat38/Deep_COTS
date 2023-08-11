# -*- coding: utf-8 -*-
"""
@author: Joshua Thompson
@email: joshuat38@gmail.com
@description: Code written for the Kaggle COTS - Great Barrier Reef competiton.
"""

import os
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm
# from sklearn.model_selection import GroupKFold

def num_boxes(annotations):
    annotations = ast.literal_eval(annotations)
    return len(annotations)

# Desired format = x1, y1, x2, y2, class_label
def get_yolo_format_bbox(img_w, img_h, bbox):
    w = bbox['width'] 
    h = bbox['height']
    
    if (bbox['x'] + bbox['width'] > 1280):
        w = 1280 - bbox['x'] 
    if (bbox['y'] + bbox['height'] > 720):
        h = 720 - bbox['y'] 
        
    xc = bbox['x'] + int(np.round(w/2))
    yc = bbox['y'] + int(np.round(h/2)) 

    return [xc/img_w, yc/img_h, w/img_w, h/img_h]

def get_coco_format_bbox(img_w, img_h, bbox):
    w = bbox['width'] 
    h = bbox['height']
    
    if (bbox['x'] + bbox['width'] > img_w):
        w = img_w - bbox['x'] 
    if (bbox['y'] + bbox['height'] > img_h):
        h = img_h - bbox['y'] 

    return [bbox['x'], bbox['y'], bbox['x']+w, bbox['y']+h]

ROOT_DIR = '/media/joshua/Storage_A/Kaggle_Datasets/competitions/tensorflow-great-barrier-reef'
SPLIT_DIR = ROOT_DIR + '/sample_splits/cross-validation'
TRAIN_DIR = ROOT_DIR + '/train_images'
TRAIN_LABELS_DIR = ROOT_DIR + '/5_fold_train_annotations'
VALID_LABELS_DIR = ROOT_DIR + '/5_fold_valid_annotations'
BASE_WIDTH = 1280
BASE_HEIGHT = 720

if not os.path.exists(TRAIN_LABELS_DIR):
    os.makedirs(TRAIN_LABELS_DIR)
if not os.path.exists(VALID_LABELS_DIR):
    os.makedirs(VALID_LABELS_DIR)


# Load the training data.
'''
To get a balanced split that is dased on video sequences as opposed to a random
split, I use the pre-split dataset by:
    @julian3833 - Reef - A CV strategy: subsequences! 
    Source: https://www.kaggle.com/julian3833/reef-a-cv-strategy-subsequences 

This is aimed at ensuring that the validation set will not contain a frame that
is immediately next to each other. In future I will need to modify this since
I will augment using a cropping method to place the COTS in the frames with no
COTS to increase dataset size but also to increase the robustness of the model.
'''
train_metaData = pd.read_csv(SPLIT_DIR + '/train-5folds.csv')
print(train_metaData.head())

# # Load the test data.
# test_metaData = pd.read_csv(ROOT_DIR + 'test.csv')
# print(test_metaData.head())

# Create the names of the video files so that it can open the files directly.
df_train = train_metaData.copy() # This is where we are making the df_train file.

# Here we add a new column to the dataframe that holds the path to the images.
df_train['image_path'] = TRAIN_DIR + '/video_' + df_train['video_id'].astype(str) + '/' + df_train['video_frame'].astype(str) + '.jpg'
df_train['image_name'] = '/video_' + df_train['video_id'].astype(str) + '/' + df_train['video_frame'].astype(str) + '.jpg'
print(df_train.head())

# Print info about the data frames.
print('Dataframe info: ', df_train.info())

df_train['num_bbox'] = df_train['annotations'].apply(lambda x: num_boxes(x))

df_wbbox = df_train[df_train.num_bbox > 0]

print(f'Dataset images with annotations: {len(df_wbbox)}')

train_lines = []
valid_lines = []

# kf = GroupKFold(n_splits = 5) 
# df_train = df_train.reset_index(drop=True)
# df_train['fold'] = -1
# for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y = df_train.video_id.tolist(), groups=df_train.sequence)):
#     df_train.loc[val_idx, 'fold'] = fold

# df_train.head(5)

for index, row in tqdm(df_wbbox.iterrows()):
    annotations = ast.literal_eval(row.annotations)
    bboxes = []
    for ann in annotations:
        # bbox = get_yolo_format_bbox(BASE_WIDTH, BASE_HEIGHT, bbox)
        bbox = get_coco_format_bbox(BASE_WIDTH, BASE_HEIGHT, ann)
        bboxes.append(bbox)
        
    if row.fold != 4:
        file_name = f'/video_{row.video_id}/{row.video_frame}.txt'
        file_path = f'{TRAIN_LABELS_DIR}{file_name}'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        line = f'{row.image_name} {file_name}\n'
        train_lines.append(line)
    else:
        file_name = f'/video_{row.video_id}/{row.video_frame}.txt'
        file_path = f'{VALID_LABELS_DIR}{file_name}'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        line = f'{row.image_name} {file_name}\n'
        valid_lines.append(line)
        
    with open(file_path, 'w') as f:
        for i, bbox in enumerate(bboxes):
            label = 0
            # bbox = [label]+bbox
            bbox.append(label)
            bbox = [str(i) for i in bbox]
            bbox = ' '.join(bbox)
            f.write(bbox)
            f.write('\n')
            
with open(f'{ROOT_DIR}/train_files_cots_5_fold.txt', 'w+') as txtFile:
    txtFile.writelines(train_lines)
with open(f'{ROOT_DIR}/valid_files_cots_5_fold.txt', 'w+') as txtFile:
    txtFile.writelines(valid_lines)
                
print("Annotations in YoloX format for all images created.")
