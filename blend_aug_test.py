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
import random
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Blending algorithms


# 1. Simple color transfer by rgb normalisation
#https://github.com/chia56028/Color-Transfer-between-Images/blob/master/color_transfer.py

def norm_color_transfer(src, dst):

    def get_mean_and_std(x):
        x_mean, x_std = cv2.meanStdDev(x)
        x_mean = np.hstack(np.around(x_mean,2)).reshape(1,1,3)
        x_std = np.hstack(np.around(x_std,2)).reshape(1,1,3)
        return x_mean, x_std

    s = cv2.cvtColor(src,cv2.COLOR_BGR2LAB)
    t = cv2.cvtColor(dst,cv2.COLOR_BGR2LAB)
    s_mean, s_std = get_mean_and_std(s)
    t_mean, t_std = get_mean_and_std(t)

    m = (s-s_mean)*(t_std/s_std)+t_mean
    m = np.round(m)
    m = np.clip(m,0,255).astype(np.uint8)

    m = cv2.cvtColor(m,cv2.COLOR_LAB2BGR)
    return m

# 2. Deep blending (not available)
# https://github.com/owenzlz/DeepImageBlending

# 3. Piosson editing  
# https://github.com/PPPW/poisson-image-editing
def laplacian_matrix(n, m):
    """Generate the Poisson matrix.
    Refer to:
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Note: it's the transpose of the wiki's matrix
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)

    return mat_A

def poisson_edit(source, target, mask, offset=(0,0)):
    """The poisson blending function.
    Refer to:
    Perez et. al., "Poisson Image Editing", 2003.
    """

    # Assume:
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min

    M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
    source = cv2.warpAffine(source,M,(x_range,y_range))

    mask = mask[y_min:y_max, x_min:x_max]
    mask[mask != 0] = 1
    #mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    # corners
    # mask[0, 0]
    # mask[0, y_range-1]
    # mask[x_range-1, 0]
    # mask[x_range-1, y_range-1]

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()

        #concat = source_flat*mask_flat + target_flat*(1-mask_flat)

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]

        x = spsolve(mat_A, mat_b)
        #print(x.shape)
        x = x.reshape((y_range, x_range))
        #print(x.shape)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #print(x.shape)

        target[y_min:y_max, x_min:x_max, channel] = x
    return target

# Helpers
def make_blend_mask(size, object_box):
    x,y,w,h = object_box
    x0=x
    x1=x+w
    y0=y
    y1=y+h


    w,h = size
    mask = np.ones((h,w,3),np.float32)

    for i in range(0,y0):
        mask[i]=i/(y0)
    for i in range(y1,h):
        mask[i]=(h-i)/(h-y1+1)
    for i in range(0,x0):
        mask[:,i]=np.minimum(mask[:,i],i/(x0))
    for i in range(x1,w):
        mask[:,i]=np.minimum(mask[:,i],(w-i)/(w-x1+1))

    return mask

def insert_object(mix, box, crop, mask):
    x,y,w,h = box
    crop = cv2.resize(crop, dsize=(w,h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(w,h), interpolation=cv2.INTER_AREA)

    mix_crop = mix[y:y+h,x:x+w]
    crop = norm_color_transfer(crop, mix_crop)
    mix[y:y+h,x:x+w] = mask*crop +(1-mask)*mix_crop
    return mix

# # Load dummy object
# object_infor = [['00012.jpg', [0,0,1024,848], [230,129,667,623]], # image_file, context_box, object_box
#                 ['00001.jpg', [0,0,767,1023], [14,75,717,897]],
#                 # ['00014.jpg',[0,0,1000,567],[168,87,702,419]],
#                 ['00021.jpg', [680,1144,1244,884], [760,1256,992,724]]]

# def load_dummy_object():

#     cots_object = []
#     for image_file, context_box, object_box in object_infor:

#         image_file = '/media/joshua/Storage_A/Kaggle_Datasets/competitions/tensorflow-great-barrier-reef/' + image_file # Download-0.jpeg'
#         image = cv2.imread(image_file, cv2.IMREAD_COLOR)

#         x,y,w,h = context_box
#         crop = image[y:y+h,x:x+w]
#         object_box = np.array(object_box)-[x,y,0,0]
#         mask = make_blend_mask((w,h), object_box)

#         #image_show('crop',crop, resize=0.5)
#         #image_show('mask',mask, resize=0.5)
#         #cv2.waitKey(0)
#         cots_object.append([crop, mask])
#     return cots_object

# def load_dummy_background():
#     video_id = 1
#     video_frame = 9187 #9287 #9187
#     image_file = 'video_%d/%d.jpg' % (video_id, video_frame)
#     image = cv2.imread('../input/tensorflow-great-barrier-reef/train_images/' + image_file, cv2.IMREAD_COLOR)
#     return image

def load_cots_objects(cots_samples, border_size=10, min_size=30):

    cots_object = []
    for _, row in random_bbox_rows.iterrows():
        
        image_file = row['image_path']
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        
        # Get the object boxes.
        object_boxes = ast.literal_eval(row.annotations) # Get the starfish annotation
        object_boxes = [[val for val in object_dict.values()] for object_dict in object_boxes]
        # print(object_boxes)
        
        if len(object_boxes) > 1:
            num_boxes = random.randint(1, len(object_boxes)-1) # Choose the number of objects to crop if more than 1.
        else:
            num_boxes = 1
        selected_indicies = random.sample(range(len(object_boxes)), num_boxes) # Grab random indices (non-repeating).
        for idx in selected_indicies: # Iterate through the indices.
            object_box = object_boxes[idx]  
            xb, yb, wb, hb = object_box
            if object_box[2] > min_size and object_box[3] > min_size and xb-border_size > 0 and yb-border_size > 0:
        
                context_box = [xb-border_size, yb-border_size, wb+border_size*2, hb+border_size*2]
                # context_box = [0 if box < 0 else box for box in context_box]
                # print(context_box)
                x,y,w,h = context_box
                crop = image[y:y+h,x:x+w]
                object_box = np.array(object_box)-[x,y,0,0] # Get the new edge of the object. (h and w stay the same).
                mask = make_blend_mask((w,h), object_box)
                
                if random.random() < 0.5: # Random left right flip.
                    _, width, _ = crop.shape
                    crop = np.fliplr(crop) # Or image = image[:, ::-1, :]
                    mask = np.fliplr(mask)
                    object_box[0] = width - object_box[0] # Switch the positions and subtract the width.
                if random.random() < 0.5: # Random up down flip.
                    height, _, _ = crop.shape
                    crop = np.flipud(crop)
                    mask = np.flipud(mask)
                    object_box[1] = height - object_box[1] # Switch the positions and subtract the width.

                cots_object.append([crop, mask, object_box])
    return cots_object

def num_boxes(annotations):
    annotations = ast.literal_eval(annotations)
    return len(annotations)

# !!! I will need to change this to give me everything in x1y1x2y2 format for success to be mine.
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

def doBoxesIntersect(a, b):
    return ((abs((a[0] + a[2]//2) - (b[0] + b[2]//2)) * 2 < (a[2] + b[2])) and
           (abs((a[1] + a[3]//2) - (b[1] + b[3]//2)) * 2 < (a[3] + b[3])))


ROOT_DIR = '/media/joshua/Storage_A/Kaggle_Datasets/competitions/tensorflow-great-barrier-reef'
SPLIT_DIR = ROOT_DIR + '/sample_splits/cross-validation'
TRAIN_DIR = ROOT_DIR + '/train_images'
TRAIN_LABELS_DIR = ROOT_DIR + '/blended_train_annotations'
VALID_LABELS_DIR = ROOT_DIR + '/blended_valid_annotations'
BASE_WIDTH = 1280
BASE_HEIGHT = 720

BLENDED_DIR = TRAIN_DIR + '/blended_train_images'
BLENDED_ANN_DIR = TRAIN_LABELS_DIR + '/blended_ann'

VALID_FOLD = 4

if not os.path.exists(TRAIN_LABELS_DIR):
    os.makedirs(TRAIN_LABELS_DIR)
if not os.path.exists(VALID_LABELS_DIR):
    os.makedirs(VALID_LABELS_DIR)
if not os.path.exists(BLENDED_DIR):
    os.makedirs(BLENDED_DIR)
if not os.path.exists(BLENDED_ANN_DIR):
    os.makedirs(BLENDED_ANN_DIR)

train_metaData = pd.read_csv(SPLIT_DIR + '/train-5folds.csv')
print(train_metaData.head())

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

train_lines = []
valid_lines = []

# !!! Write the files files with bounding boxes.
print(f'Dataset images with annotations: {len(df_wbbox)}')

for index, row in tqdm(df_wbbox.iterrows()):
    annotations = ast.literal_eval(row.annotations)
    bboxes = []
    for ann in annotations:
        # bbox = get_yolo_format_bbox(BASE_WIDTH, BASE_HEIGHT, bbox)
        bbox = get_coco_format_bbox(BASE_WIDTH, BASE_HEIGHT, ann)
        bboxes.append(bbox)
        
    if row.fold != VALID_FOLD:
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

df_no_bbox = df_train[df_train.num_bbox == 0]
# df_aug_ready = df_no_bbox[df_no_bbox.index % 4 != 0]  # Excludes every 4th row starting from 0
# df_no_bbox = df_no_bbox.iloc[::4, :] # Get every 4th row.

# !!! Write all files without a bounding box. Training data only.
print(f'Dataset images without annotations: {len(df_no_bbox)}')

for index, row in tqdm(df_no_bbox.iterrows()):
        
    if row.fold != VALID_FOLD:
        file_name = f'/video_{row.video_id}/{row.video_frame}.txt'
        file_path = f'{TRAIN_LABELS_DIR}{file_name}'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        line = f'{row.image_name} {file_name}\n'
        train_lines.append(line)
        
        with open(file_path, 'w') as f:
            f.write('\n')
 
df_wbbox_train = df_wbbox[df_wbbox.fold != VALID_FOLD] # !!! VERY IMPORTANT! We only want to use COTS from the training folds!
df_no_bbox = df_no_bbox[df_train.fold != VALID_FOLD] # !!! VERY IMPORTANT! We only want to use backgrounds from the training folds!

# Random test to see if this works!
max_blended = 5000
for k in range(max_blended):
    
    accept = 'n'
    img_name = f'/blend_{k}.jpg'
    img_line = f'/blended_train_images/blend_{k}.jpg'
    ann_name = f'/blended_ann/blend_{k}.txt'
    ann_path = f'{BLENDED_ANN_DIR}/blend_{k}.txt'
    border_size = 20
    hsv_lower = (95, 120, 160)
    hsv_upper = (108, 255, 255)
    max_retrys = 100

    while accept == 'n':
        bboxes = []
        ann_bboxes = []
        # Randomly select a background.
        random_background_row = df_no_bbox.sample().reset_index(drop=True)
        # print(random_background_row['image_path'].iloc[0])
        random_bakground_path = random_background_row['image_path'].iloc[0]
        # print(random_bakground_path)
        background = cv2.imread(random_bakground_path, cv2.IMREAD_COLOR)
        if random.random() < 0.5: # Randomly flip the background to ensure variation.
            background = np.fliplr(background) # Or image = image[:, ::-1, :]
        
        # cots_object = load_dummy_object()
        # background = load_dummy_background()
        random_bbox_rows = df_wbbox_train.sample(n=random.randint(1, 10))
        cots_objects = load_cots_objects(random_bbox_rows, 
                                         border_size=border_size, min_size=50)
        
        retry_count = 0 # Place this here so we can force the thing to quit.
        
        if len(cots_objects) > 0:
            
            max_h = 280
            min_h = 80
            max_w = 280
            min_w = 80
            min_x = 100
            max_x = 1280-max_w-100
            min_y = 100
            max_y = 720-max_h-100
            
            mix = background.copy()
            mix1 = None
            bbox = []
            for cots_object in cots_objects:
                crop, mask, object_box = cots_object
                
                # Adjust hw sizes.
                hw_ratio = object_box[3]/object_box[2]
                if hw_ratio > 0:
                    max_w = max_w * hw_ratio
                else:
                    max_h = max_h * hw_ratio
                
                redo = True
                while redo == True:
                    x = random.randint(min_x, max_x)
                    y = random.randint(min_y, max_y)
                    
                    scalar = random.uniform(1.5, 2)
                    
                    h = object_box[3] * scalar
                    w = object_box[2] * scalar
                    if h < min_h:
                        h = min_h
                    if h > max_h:
                        h = max_h
                    if w < min_w:
                        w = min_w
                    if w > max_w:
                        w = max_w
                    x = int(x)
                    y = int(y)
                    h = int(h)
                    w = int(w)
                    # box.append([x, y, w, h])
                    
                    if len(bboxes) > 0:
                        needs_redo = False
                        ann_bbox = [int(x+border_size/2*scalar), int(y+border_size/2*scalar), 
                                    int(w-border_size*scalar), int(h-border_size*scalar)]
                        for tmp_bbox in bboxes:
                            if doBoxesIntersect(ann_bbox, tmp_bbox) == True:
                                needs_redo = True
                                break
                        if needs_redo == False:
                            redo = False
                    else:
                        redo = False
                        
                    # Check to see if the target area is likely to be ocean. 
                    target_area = mix[y:y+h,x:x+w]
                    target_area_hsv = cv2.cvtColor(target_area, cv2.COLOR_BGR2HSV)
                    valid_mask = cv2.inRange(target_area_hsv, hsv_lower, hsv_upper)/255
                    # print('H: ', np.amax(target_area_hsv[:, :, 0]), np.amin(target_area_hsv[:, :, 0]),
                    #       ', S: ', np.amax(target_area_hsv[:, :, 1]), np.amin(target_area_hsv[:, :, 1]),
                    #       ', V: ', np.amax(target_area_hsv[:, :, 2]), np.amin(target_area_hsv[:, :, 2]),
                    #       ', x, y: ', x, y,
                    #       ', mask max: ', np.amax(valid_mask), np.amin(valid_mask))
                    if valid_mask.mean() > 0.75: # If 75% is ocean, do not place, retry.
                        # print('\nRe-doing the placement due to bad location!')
                        redo = True
                        retry_count += 1
                        
                    if retry_count == max_retrys: # Break from the redo loop.
                        break
                    
                if retry_count == max_retrys and redo == True: # Break from the cots object placement loop.
                    break
                                
                #mix = insert_object (mix, box, crop, mask*0.5) ------
                crop = cv2.resize(crop, dsize=(w,h), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, dsize=(w,h), interpolation=cv2.INTER_AREA)
            
                mix_crop = mix[y:y+h,x:x+w]
                crop = norm_color_transfer(crop, mix_crop) # This one seems to work better.
                # crop = poisson_edit(crop, mix_crop, (mask[:,:,0]>0.5).astype(np.float32), offset=(0,0))
            
                mask = mask*0.9  #mixup ratio
                mix[y:y+h,x:x+w] = mask*crop +(1-mask)*mix_crop
                
                bbox = [int(x+border_size/2*scalar), int(y+border_size/2*scalar), 
                        int(w-border_size*scalar), int(h-border_size*scalar)]
                bboxes.append(bbox)
                ann_bboxes.append(get_coco_format_bbox(BASE_WIDTH, BASE_HEIGHT, 
                                                       {'x' : bbox[0], 'y' : bbox[1],
                                                        'width' : bbox[2], 'height' : bbox[3]}))
                
            if retry_count == max_retrys and redo == True: # Break from the cots object placement loop.
                print('\nSkipping this background due to retry limit being exceeded. Randomly generating a new sample...')
                accept = 'n'
                
            else:
                accept = 'y'
                # Show the bounding boxes.
                # mix1= mix.copy()
                # for i, bbox in enumerate(bboxes):
                #     x1,y1,w1,h1 = bbox
                #     cv2.rectangle(mix1, (x1,y1), (x1+w1,y1+h1), (0,255,0), 2)
                    
                # # plt.figure(figsize=(15,20))
                # # plt.title('original background')
                # # plt.imshow(background[...,::-1])
                # # plt.show()
                
                # # plt.figure(figsize=(15,20))
                # # plt.title('augmeted image')
                # # plt.imshow(mix[...,::-1])
                # # plt.show()
                
                # plt.figure(figsize=(15,20))
                # plt.title('same augmeted image with marking')
                # plt.imshow(mix1[...,::-1])
                # plt.show()
                
                # accept = input('\nDo you accept these augmented images? [y/n]: ')
                # accept = accept.lower()
                # while accept != 'n' and accept != 'y' and accept != 'q':
                #     accept = input('\nDo you accept these augmented images? Please use [y/n]: ')
                #     accept = accept.lower()
                    
                # if accept == 'y':
                print(f'Saving sample {k}...')
                cv2.imwrite(BLENDED_DIR+img_name, mix)
                line = f'{img_line} {ann_name}\n'
                train_lines.append(line)
                
                with open(ann_path, 'w') as f:
                    for i, bbox in enumerate(bboxes):
                        label = 0
                        # bbox = [label]+bbox
                        bbox.append(label)
                        bbox = [str(i) for i in bbox]
                        bbox = ' '.join(bbox)
                        f.write(bbox)
                        f.write('\n')
                # else:
                #     print('Randomly generating a new sample...')
                    
            # if accept == 'q':
            #     break
        
with open(f'{ROOT_DIR}/train_files_cots_5_fold_blended.txt', 'w+') as txtFile:
    txtFile.writelines(train_lines)
with open(f'{ROOT_DIR}/valid_files_cots_5_fold_blended.txt', 'w+') as txtFile:
    txtFile.writelines(valid_lines)
                
print("Annotations in YoloX format for all images created.")