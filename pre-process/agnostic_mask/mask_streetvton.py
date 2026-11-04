import sys
import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

PATH = './data/streetvton/test'

def get_black_while_mask_street(im, im_parse, pose_data):
    parse_array = np.array(im_parse)
    parse_head = ((parse_array == 1).astype(np.float32) +
                    (parse_array == 2).astype(np.float32) +
                    (parse_array == 3).astype(np.float32) +
                    (parse_array == 11).astype(np.float32) +
                    (parse_array == 17).astype(np.float32))
    parse_lower = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 8).astype(np.float32) +
                    (parse_array == 9).astype(np.float32) +
                    (parse_array == 10).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))

    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)
    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[11] - pose_data[8])
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

    r = int(length_a / 16) + 1

    for i in [8, 11]:   
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 8]], 'gray', width=r*6)
    
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'gray', width=r*6)
    
    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'gray', width=r*12)
    
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 11, 8]], 'gray', 'gray')

    
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

    
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
    
    
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
    
    for i in [3, 4, 6, 7]:
        if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        
    for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
        mask_arm = Image.new('L', im.size, 'white')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == -2.0 and pose_data[i-1, 1] == -2.0) or (pose_data[i, 0] == -2.0 and pose_data[i, 1] == -2.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
        mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')
        
        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
        
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    return agnostic


def get_all_black_white_street():
    for im_name in tqdm(os.listdir(osp.join(PATH, 'image'))):
    # for im_name in tqdm(['010846.jpg']):
        image_name = im_name.replace('.jpg','').replace('.png','')
        img = Image.open(osp.join(PATH,'image', image_name+'.jpg')).convert('RGB')
        width,height = img.size
        im = np.ones((height,width))
        im = Image.fromarray(np.uint8(im * 255), 'L')
        with open(osp.join(PATH, 'keypoints', image_name+'_keypoint.json'), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
            pose_data[:,0]  = pose_data[:,0] * width
            pose_data[:,1]  = pose_data[:,1] * height
            pose_data = np.where(pose_data < 0.0, -2.0, pose_data)
            # print(pose_data)
        im_label = Image.open(osp.join(PATH, 'segm_atr_labels', image_name + '.png')).convert('P').resize(size=(width, height))
        agnostic = get_black_while_mask_street(im, im_label, pose_data)
        agnostic = np.array(agnostic)
        agnostic = (agnostic <= 254).astype(int)
        agnostic = agnostic * 255.0
        agnostic = Image.fromarray(np.uint8(agnostic)).convert('RGB')
        agnostic.save(osp.join(PATH, 'mask', image_name+'.png'))

def get_agnmask_street():
    for im_name in tqdm(os.listdir(osp.join(PATH, 'image'))):
    # for im_name in tqdm(['010846.jpg']):
        image_name = im_name.replace('.jpg','').replace('.png','')
        img = Image.open(osp.join(PATH,'image', image_name+'.jpg')).convert('RGB')
        mask = Image.open(osp.join(PATH,'mask', image_name+'.png')).convert('RGB')
        agn_mask = np.array(mask) * 128
        img.paste(Image.fromarray(np.uint8(agn_mask)), None, mask=mask.convert('L'))
        img.save(osp.join(PATH, 'agnostic', image_name+'.png'))
        # exit()
        
if __name__ == '__main__':
    # get_all_black_white_street()
    get_agnmask_street()
        
        
        # width,height = img.size
        # im = np.ones((height,width))
        # im = Image.fromarray(np.uint8(im * 255), 'L')
        # with open(osp.join(PATH, 'keypoints', image_name+'_keypoint.json'), 'r') as f:
        #     pose_label = json.load(f)
        #     pose_data = pose_label['people'][0]['pose_keypoints_2d']
        #     pose_data = np.array(pose_data)
        #     pose_data = pose_data.reshape((-1, 3))[:, :2]
        #     pose_data[:,0]  = pose_data[:,0] * width
        #     pose_data[:,1]  = pose_data[:,1] * height
        #     pose_data = np.where(pose_data < 0.0, -2.0, pose_data)
        #     # print(pose_data)
        # im_label = Image.open(osp.join(PATH, 'segm_atr_labels', image_name + '.png')).convert('P').resize(size=(width, height))
        # agnostic = get_black_while_mask_street(im, im_label, pose_data)
        # agnostic = np.array(agnostic)

        # agnostic = Image.fromarray(np.uint8(agnostic)).convert('RGB')
        # agnostic.save(osp.join(PATH, 'mask', image_name+'.jpg'))
