import os
import numpy as np
import torch

from transformers import BertTokenizer
from PIL import Image
import cv2

def gen_scene_ids(
    cloth_desc:str,
    agn_desc:str,
    pre_filled_desc:str,
    tokenzer:BertTokenizer,
    max_length:int=512):
    
    '''
    convert conditional scene text into id tensors
    '''
    # sep_str = ' ' + tokenzer.sep_token + ' '
    final_desc = pre_filled_desc # + sep_str
    final_tokens = tokenzer.tokenize(final_desc)
    final_length = len(final_tokens)
    gap_length = max_length - final_length - 1
    if gap_length > 0:
        final_tokens = ['[CLS]'] + final_tokens + [tokenzer.pad_token] * gap_length
        final_ids = tokenzer.convert_tokens_to_ids(final_tokens)
        attention_mask = [1] * (final_length + 1) + [0] * gap_length
        
    elif gap_length <= 0:
        final_tokens = ['[CLS]'] + final_tokens[:max_length - 1]
        final_ids = tokenzer.convert_tokens_to_ids(final_tokens)
        attention_mask = [1] * max_length
        
    token_type_ids = [0] * max_length
    final_ids = torch.tensor(final_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    
    return final_ids, attention_mask, token_type_ids


def cat_images(image1:Image.Image, image2:Image.Image, direct='hor')->Image.Image:
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    cant_arr = np.concatenate([arr1, arr2], axis=1)
    cant = Image.fromarray(cant_arr)
    return cant

def cat_images_cv2(image1:cv2.typing.MatLike, image2:cv2.typing.MatLike, direct='hor')->cv2.typing.MatLike:
    arr1 = image1
    arr2 = image2
    cant_arr = np.concatenate([arr1, arr2], axis=1)
    return cant_arr


def cat_image_edge(agn:cv2.typing.MatLike, 
                   mask:cv2.typing.MatLike, 
                   cloth:cv2.typing.MatLike) -> cv2.typing.MatLike:
    
    mask = (mask >= 128).astype(np.uint8)

    if len(mask.shape) == 2:
        mask = mask[:,:,None]
        mask = np.repeat(mask,
                        repeats=3,
                        axis=2)
    elif len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = np.repeat(mask,
                        repeats=3,
                        axis=2)
    elif len(mask.shape) == 3 and mask.shape[2] == 3:
        pass
    else:
        raise TypeError('can not process shape:{}'.format(mask.shape))
    height, width, channles = agn.shape
    
    # crop agn
    sum_mask_width = np.sum(mask[:,:,0], axis=0)
    width_list = np.nonzero(sum_mask_width)
    widths = [width_list[0][0] - 1 if width_list[0][0]>1 else 0, 
                  width_list[0][-1] + 1 if width_list[0][-1] != width else width]

    sum_mask_height = np.sum(mask[:,:,0], axis=1)
    height_list = np.nonzero(sum_mask_height)
    heights = [height_list[0][0] - 1 if height_list[0][0]>1 else 0, 
                  height_list[0][-1] + 1 if height_list[0][-1] != height else height] 

    target_size = (widths[1] - widths[0],
                   heights[1] - heights[0])
    fitted_cloth = cv2.resize(cloth, target_size)
    
    no_cloth = agn * (1 - mask)
    
    small_cloth = fitted_cloth * mask[heights[0]: heights[1], widths[0]:widths[1],:]
    with_cloth = np.zeros(agn.shape)
    with_cloth[heights[0]: heights[1], widths[0]:widths[1],:] += small_cloth

    pre_fill = no_cloth.astype(np.uint8) + with_cloth.astype(np.uint8)

    return pre_fill

def crop_cloth_func(cloth:cv2.typing.MatLike,
            cloth_mask:cv2.typing.MatLike) -> cv2.typing.MatLike:
    cloth_mask = (cloth_mask >= 128).astype(np.uint8)
    if len(cloth_mask.shape) == 2:
        cloth_mask = cloth_mask[:,:,None]
        cloth_mask = np.repeat(cloth_mask,
                        repeats=3,
                        axis=2)
    elif len(cloth_mask.shape) == 3 and cloth_mask.shape[2] == 1:
        cloth_mask = np.repeat(cloth_mask,
                        repeats=3,
                        axis=2)
    elif len(cloth_mask.shape) == 3 and cloth_mask.shape[2] == 3:
        pass
    else:
        raise TypeError('can not process shape:{}'.format(cloth_mask.shape))
    
    cloth_height, cloth_width, channles = cloth.shape
    sum_cloth_width = np.sum(cloth_mask[:,:,0], axis=0)
    cloth_width_list = np.nonzero(sum_cloth_width)
    cloth_widths = [cloth_width_list[0][0] - 1 if cloth_width_list[0][0]>1 else 0, 
                cloth_width_list[0][-1] + 1 if cloth_width_list[0][-1] != cloth_width else cloth_width]

    sum_cloth_height = np.sum(cloth_mask[:,:,0], axis=1)
    cloth_height_list = np.nonzero(sum_cloth_height)
    cloth_heights = [cloth_height_list[0][0] - 1 if cloth_height_list[0][0]>1 else 0, 
                cloth_height_list[0][-1] + 1 if cloth_height_list[0][-1] < cloth_height else cloth_height]
    
    cloth = cloth[cloth_heights[0]:cloth_heights[1], cloth_widths[0]:cloth_widths[1]]

    return cloth