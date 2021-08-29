'''
generate '*_gt.json' and mask
usage: python gen_gt_json.py
'''

import os
import glob
import _pickle as cPickle
import json
import cv2
import numpy as np
from tqdm import tqdm


img_dir = 'datasets/lm_renders_blender/renders'
max_depth = 2000

folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
# del folder_list[0:6]
# folder_list = ['duck']

for folder in tqdm(folder_list):
    print(folder + ':')
    imfomation = {}
    img_paths = glob.glob(os.path.join(img_dir, folder, '*.jpg'))
    img_paths = sorted(img_paths)
    miss_num = 0
    for img_path in tqdm(img_paths):
        # img_path = 'datasets/lm_renders_blender/renders/duck/406.jpg'
        all_exist = os.path.exists(img_path.replace('.jpg', '_RT.pkl')) and \
                    os.path.exists(img_path.replace('.jpg', '_depth.png'))
        if not all_exist:
            miss_num += 1
            continue
        objects_info = []
        img_name = os.path.basename(img_path)
        img_ind = img_name.split('.')[0]
        object_info = {}
        #RT
        RT_path = img_path.replace('.jpg', '_RT.pkl')
        with open(RT_path, 'rb') as f:
            RT = cPickle.load(f)
        object_info['cam_R_m2c'] = RT['RT'][:3, :3].tolist()
        object_info['cam_t_m2c'] = RT['RT'][:3, 3].tolist()
        #save mask image
        depth_path = img_path.replace('.jpg', '_depth.png') 
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask = depth!=max_depth
        area = mask.sum()
        #save mask
        mask = mask*255
        mask_path = depth_path.replace('depth', 'mask')
        cv2.imwrite(mask_path, mask)        
        if area < 1:
            miss_num += 1
            object_info['bbox_visib'] = [-1, -1, -1, -1]        
            # continue
        else:
            #bboxes
            horizontal_indicies = np.where(np.any(mask, axis=0))[0]
            vertical_indicies = np.where(np.any(mask, axis=1))[0]
            assert horizontal_indicies.shape[0], print(img_path)
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            w, h = x2 - x1, y2 - y1
            if np.any(np.logical_or(w > 600, h > 440)):
                miss_num += 1
                continue            
            object_info['bbox_visib'] = [float(x1), float(y1), float(w), float(h)]

        objects_info.append(object_info)
        imfomation[img_ind] = objects_info
    print('miss_num: {}'.format(miss_num))
    with open(os.path.join(img_dir, '{}_gt.json'.format(folder)), "w") as f:
        json.dump(imfomation, f, indent=4)    
