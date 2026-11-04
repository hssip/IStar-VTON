
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import numpy as np
import json
import torch
from tqdm import tqdm

from transformers import(
    AutoModelForCausalLM,
    AutoProcessor
    )
from PIL import Image
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_pre_fill(agn:cv2.typing.MatLike, 
                   mask:cv2.typing.MatLike, 
                   cloth:cv2.typing.MatLike,
                   cloth_mask:cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    with RGB mode
    '''
    
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

    #### get croped cloth
    height, width, channles = cloth_mask.shape
    sum_mask_width = np.sum(cloth_mask[:,:,0], axis=0)
    width_list = np.nonzero(sum_mask_width)
    widths = [width_list[0][0] - 1 if width_list[0][0]>1 else 0, 
                  width_list[0][-1] + 1 if width_list[0][-1] != width else width]
    sum_mask_height = np.sum(cloth_mask[:,:,0], axis=1)
    height_list = np.nonzero(sum_mask_height)
    heights = [height_list[0][0] - 1 if height_list[0][0]>1 else 0, 
                  height_list[0][-1] + 1 if height_list[0][-1] != height else height]
    crop_cloth = cloth[heights[0]:heights[1], widths[0]:widths[1], :]
    crop_cloth_mask = cloth_mask[heights[0]:heights[1], widths[0]:widths[1], :]
    
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
    fitted_cloth = cv2.resize(crop_cloth, target_size)
    no_cloth = agn * (1 - mask)
    small_cloth = fitted_cloth * mask[heights[0]: heights[1], widths[0]:widths[1],:]
    yes_cloth = np.zeros(agn.shape)
    yes_cloth[heights[0]: heights[1], widths[0]:widths[1],:] += small_cloth

    pre_fill = no_cloth.astype(np.uint8) + yes_cloth.astype(np.uint8)
    pre_fill = pre_fill.astype(np.uint8)

    return pre_fill


def cat_images(image1:Image.Image, image2:Image.Image, direct='hor')->Image.Image:
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    cant_arr = np.concatenate([arr1, arr2], axis=1)
    cant = Image.fromarray(cant_arr)
    return cant


def load_florence_model(model_path='gokaygokay-florence-2-sd3-captioner'):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True
    )
    return model, processor


def load_files_viton(data_root=None):


    result = {
        'train':[],
        'test':[]
    }

    for kind in ['test','train']:
        infilename = os.path.join(data_root,'{}_pairs.txt'.format(kind))
        with open(infilename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line.strip('\n') for line in lines]
        for line in lines:
            first,second = line.strip('\n').split(' ')
            first_name = first.split('.')[0]
            second_name = second.split('.')[0]
            agn_path = os.path.join(data_root, kind,'agnostic-v3.2','{}.jpg'.format(first_name))
            agn_mask_path = os.path.join(data_root, kind,'agnostic-mask','{}_mask.png'.format(first_name))
            cloth_path = os.path.join(data_root, kind,'cloth','{}.jpg'.format(first_name))
            cloth_mask_path = os.path.join(data_root, kind,'cloth-mask','{}.jpg'.format(first_name))
            unpair_cloth_path = os.path.join(data_root, kind,'cloth','{}.jpg'.format(second_name))
            unpair_cloth_mask_path = os.path.join(data_root, kind,'cloth-mask','{}.jpg'.format(second_name))
            result[kind].append({
                'agn_path':agn_path,
                'agn_mask_path':agn_mask_path,
                'cloth_path':cloth_path,
                'cloth_mask_path':cloth_mask_path,
                'unpair_cloth_path':unpair_cloth_path,
                'unpair_cloth_mask_path':unpair_cloth_mask_path,
            })

    return result


def generate_caption_viton():


    florence_model, florence_processor = load_florence_model()

    fashion_model = florence_model
    fashion_processor = florence_processor


    DATA_ROOT='/data/zalando-hd-resized'
    files_info = load_files_viton(data_root=DATA_ROOT)


    for kind, pair_list in files_info.items():
        print('---- processing {} split ----'.format(kind))
        outfile = open(os.path.join(DATA_ROOT,'{}_captions.json'.format(kind)), mode='w', encoding='utf-8')
        single_image_dict = {}
        prefilled_image_list = []
        unpair_prefilled_image_list = []
        for p_names in tqdm(pair_list):
            agn_image = Image.open(p_names['agn_path']).convert('RGB')
            agn_mask_image = Image.open(p_names['agn_mask_path']).convert('RGB')
            cloth_image = Image.open(p_names['cloth_path']).convert('RGB')
            cloth_mask_image = Image.open(p_names['cloth_mask_path']).convert('RGB')
            unpair_cloth_image = Image.open(p_names['unpair_cloth_path']).convert('RGB')
            unpair_cloth_mask_image = Image.open(p_names['unpair_cloth_mask_path']).convert('RGB')

            
            pair_prefilled_image = cat_images(agn_image, cloth_image)
            unpair_prefilled_image = cat_images(agn_image, unpair_cloth_image)
            
            
            used_name = p_names['agn_path'].split('/')[-1].split('.')[0]
            if used_name not in single_image_dict.keys():
                images = [agn_image, cloth_image, pair_prefilled_image, unpair_prefilled_image]
            else:
                images = [pair_prefilled_image, unpair_prefilled_image]

            task_prompt = '<DESCRIPTION>'
            text_prompt_list = [task_prompt + ' Imagine and describe this person’s actions, facial expression, and body posture in the image. \nDo not describe their clothing and the gray region which is masked. \nProvide a detailed and realistic depiction.',
                                task_prompt + ' Describe the fashion garment in this image with great details.',
                                task_prompt + ' Imagine and describe the details feature of the combining the person and right garment. \nDo not include the masked gray region.',
                                task_prompt + ' Imagine and describe the details feature of the combining the person and right garment. \nDo not include the masked gray region.'
                                ]
            input_prompt = text_prompt_list[4-len(images):4]
            inputs = fashion_processor(images=images, text=input_prompt, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(DEVICE)
            pixel_values = inputs['pixel_values'].to(DEVICE)
            
            generated_ids = fashion_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=256, num_beams=3)
            res = fashion_processor.batch_decode(generated_ids, skip_special_tokens=False)
            res = [fashion_processor.post_process_generation(r, task=task_prompt, image_size=(agn_image.width, agn_image.height)) 
                   for r in res]
            if used_name not in single_image_dict.keys():
                single_image_dict[used_name]={
                    'agn':res[0][task_prompt].replace('<pad>',''),
                    'cloth':res[1][task_prompt].replace('<pad>',''),
                }
                
                prefilled_image_list.append(res[2][task_prompt].replace('<pad>',''))
                unpair_prefilled_image_list.append(res[3][task_prompt].replace('<pad>',''))
            else:
                prefilled_image_list.append(res[0][task_prompt].replace('<pad>',''))
                unpair_prefilled_image_list.append(res[1][task_prompt].replace('<pad>',''))
        outdata = {
            'single_image_dict':single_image_dict,
            'prefilled_image_list':prefilled_image_list,
            'unpair_prefilled_image_list':unpair_prefilled_image_list
        }
        json.dump(
            outdata,
            outfile,
            ensure_ascii=False,
            indent=2,
        )
        outfile.close()


def load_files_dresscode(data_root=None, pair='pair'):
    
    result = {
        'train':[],
        'test':{
            'paired':[],
            'unpaired':[]
        }
    }
    
    for data_kind in ['train', 'test']:
        
        if data_kind == 'train':
            pair_filename = os.path.join(data_root, 'train_pairs.txt')
            with open(pair_filename, mode='r', encoding='utf-8') as f:
                pair_lines = f.readlines()
            
            for line in pair_lines:
                used_name = line.strip('\n').split('\t')[0].split('.')[0].split('_')[0]
                agn_path = os.path.join(data_root,'agnostic','{}_0.png'.format(used_name))
                cloth_path = os.path.join(data_root,'images','{}_1.jpg'.format(used_name))
                result[data_kind].append({
                    'agn_path':agn_path,
                    'cloth_path':cloth_path,
                    'unpair_cloth_path':None
                })
            unpair_lines = None
        else:
            pair_filename = os.path.join(data_root,'test_pairs_paired.txt')
            with open(pair_filename, mode='r', encoding='utf-8') as f:
                pair_lines = f.readlines()
            for line in pair_lines:
                used_name = line.strip('\n').split('\t')[0].split('.')[0].split('_')[0]
                agn_path = os.path.join(data_root,'agnostic','{}_0.png'.format(used_name))
                cloth_path = os.path.join(data_root,'images','{}_1.jpg'.format(used_name))
                result[data_kind]['paired'].append({
                    'agn_path':agn_path,
                    'cloth_path':cloth_path,
                    'unpair_cloth_path':None,
                }) 
            
            unpair_filename = os.path.join(data_root,'test_pairs_unpaired.txt')
            with open(unpair_filename, mode='r', encoding='utf-8') as f:
                unpair_lines = f.readlines()
            for line in unpair_lines:
                used_name = line.strip('\n').split('\t')[0].split('.')[0].split('_')[0]
                second_name  = line.strip('\n').split('\t')[1].split('.')[0].split('_')[0]
                agn_path = os.path.join(data_root,'agnostic','{}_0.png'.format(used_name))
                cloth_path = os.path.join(data_root,'images','{}_1.jpg'.format(used_name))
                unpair_cloth_path = os.path.join(data_root,'images','{}_1.jpg'.format(second_name))
                result[data_kind]['unpaired'].append({
                    'agn_path':agn_path,
                    'cloth_path':cloth_path,
                    'unpair_cloth_path':unpair_cloth_path,
                })
    return result
        

def generate_caption_dresscode():

    florence_model, florence_processor = load_florence_model()

    fashion_model = florence_model
    fashion_processor = florence_processor

    DATA_PART='lower_body'
    DATA_ROOT='./data/DressCode/{}'.format(DATA_PART)
    files_info = load_files_dresscode(data_root=DATA_ROOT)


    for kind, used_pair_list in files_info.items():
        print('---- processing {} split ----'.format(kind))
        if kind == 'train':
            single_image_dict = {}
            prefilled_image_list = []
            unpair_prefilled_image_list = []
            outfile = open(os.path.join(DATA_ROOT,'{}_captions.json'.format(kind)), mode='w', encoding='utf-8')
            for p_names in tqdm(used_pair_list):
                agn_image = Image.open(p_names['agn_path']).convert('RGB')
                cloth_image = Image.open(p_names['cloth_path']).convert('RGB')
                
                pair_prefilled_image = cat_images(agn_image, cloth_image)
                
                
                used_name = p_names['agn_path'].split('/')[-1].split('.')[0].split('_')[0]
                if used_name not in single_image_dict.keys():
                    images = [agn_image, cloth_image, pair_prefilled_image]
                elif used_name in single_image_dict.keys():
                    images = [pair_prefilled_image]
                
                task_prompt = '<DESCRIPTION>'
                text_prompt_list = [task_prompt + 'Imagine and describe this person’s actions, facial expression, and body posture in the image. \nDo not describe their clothing and the gray region which is masked. \nProvide a detailed and realistic depiction.',
                                    task_prompt + 'Describe the fashion garment in this image with great details.',
                                    task_prompt + 'Imagine and describe the details feature of the combining the person and right garment. \nDo not include the masked gray region.',
                                    ]
                input_prompt = text_prompt_list[3-len(images):3]
                inputs = fashion_processor(images=images, text=input_prompt, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(DEVICE)
                pixel_values = inputs['pixel_values'].to(DEVICE)
                
                generated_ids = fashion_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=256, num_beams=3)
                res = fashion_processor.batch_decode(generated_ids, skip_special_tokens=False)
                res = [fashion_processor.post_process_generation(r, task=task_prompt, image_size=(agn_image.width, agn_image.height)) 
                    for r in res]
                # print(res)
                if used_name not in single_image_dict.keys():
                    single_image_dict[used_name]={
                        'agn':res[0][task_prompt].replace('<pad>',''),
                        'cloth':res[1][task_prompt].replace('<pad>',''),
                    }
                prefilled_image_list.append(res[2][task_prompt].replace('<pad>',''))
            outdata = {
                'single_image_dict':single_image_dict,
                'prefilled_image_list':prefilled_image_list,
                'unpair_prefilled_image_list':[]
            }
            json.dump(
                outdata,
                outfile,
                ensure_ascii=False,
                indent=2,
            )
            outfile.close()
        elif kind == 'test':
            for list_name in ['paired', 'unpaired']:
                single_image_dict = {}
                prefilled_image_list = []
                unpair_prefilled_image_list = []
                outfile = open(os.path.join(DATA_ROOT,'{}_captions_{}.json'.format(kind,list_name)), mode='w', encoding='utf-8')
                pp_list = used_pair_list[list_name]
                for p_names in tqdm(pp_list):
                    agn_image = Image.open(p_names['agn_path']).convert('RGB')
                    cloth_image = Image.open(p_names['cloth_path']).convert('RGB')
                    unpair_cloth_image = Image.open(p_names['unpair_cloth_path']).convert('RGB') if  list_name == 'unpaired' else None
                    
                    pair_prefilled_image = cat_images(agn_image, cloth_image)
                    if list_name == 'unpaired':
                        unpair_prefilled_image = cat_images(agn_image, unpair_cloth_image)
                        
                    used_name = p_names['agn_path'].split('/')[-1].split('.')[0].split('_')[0]
                    if used_name not in single_image_dict.keys():
                        images = [agn_image, cloth_image, pair_prefilled_image]
                        if list_name == 'unpaired':
                            images.append(unpair_prefilled_image)
                    elif used_name in single_image_dict.keys():
                        images = [pair_prefilled_image]
                        if list_name == 'unpaired':
                            images.append(unpair_prefilled_image)
                    task_prompt = ' <DESCRIPTION> '
                    text_prompt_list = [task_prompt + 'Imagine and describe this person’s actions, facial expression, and body posture in the image. \nDo not describe their clothing and the gray region which is masked. \nProvide a detailed and realistic depiction.',
                                        task_prompt + 'Describe the fashion garment in this image with great details.',
                                        task_prompt + 'Imagine and describe the details feature of the combining the person and right garment. \nDo not include the masked gray region.',
                                        task_prompt + 'Imagine and describe the details feature of the combining the person and right garment. \nDo not include the masked gray region.'
                                        ]
                    if len(images) == 1:
                        input_prompt = text_prompt_list[2:3]
                    elif len(images) == 2:
                        input_prompt = text_prompt_list[2:4]
                    elif len(images) == 3:
                        input_prompt = text_prompt_list[0:3]
                    elif len(images) == 4:
                        input_prompt = text_prompt_list[0:4]
                    
                    inputs = fashion_processor(images=images, text=input_prompt, return_tensors='pt', padding=True)
                    input_ids = inputs['input_ids'].to(DEVICE)
                    pixel_values = inputs['pixel_values'].to(DEVICE)
                    
                    generated_ids = fashion_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=256, num_beams=3)
                    res = fashion_processor.batch_decode(generated_ids, skip_special_tokens=False)
                    res = [fashion_processor.post_process_generation(r, task=task_prompt, image_size=(agn_image.width, agn_image.height)) 
                        for r in res]
                    # print(res)
                    if used_name not in single_image_dict.keys():
                        single_image_dict[used_name]={
                            'agn':res[0][task_prompt].replace('<pad>',''),
                            'cloth':res[1][task_prompt].replace('<pad>',''),
                        }
                        prefilled_image_list.append(res[2][task_prompt].replace('<pad>',''))
                        if list_name == 'unpaired':
                            unpair_prefilled_image_list.append(res[3][task_prompt].replace('<pad>',''))
                    else:
                        prefilled_image_list.append(res[0][task_prompt].replace('<pad>',''))
                        if list_name == 'unpaired':
                            unpair_prefilled_image_list.append(res[1][task_prompt].replace('<pad>',''))
                outdata = {
                    'single_image_dict':single_image_dict,
                    'prefilled_image_list':prefilled_image_list,
                    'unpair_prefilled_image_list':unpair_prefilled_image_list
                }
                json.dump(
                    outdata,
                    outfile,
                    ensure_ascii=False,
                    indent=2,
                )
                outfile.close()


def load_files_streetvton(data_root=None):

    result = {
        'test':[]
    }

    for kind in ['test']:
        infilename = os.path.join(data_root,'{}_pairs.txt'.format(kind))
        with open(infilename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line.strip('\n') for line in lines]
        for line in lines:
            first,second = line.strip('\n').split(' ')
            first_name = first.split('.')[0]
            second_name = second.split('.')[0]
            agn_path = os.path.join(data_root, kind,'agnostic','{}.png'.format(first_name))
            unpair_cloth_path = os.path.join(data_root, kind,'cloth','{}.jpg'.format(second_name))
            result[kind].append({
                'agn_path':agn_path,
                'unpair_cloth_path':unpair_cloth_path,
            })

    return result


def generate_caption_streetvton():
    florence_model, florence_processor = load_florence_model()

    fashion_model = florence_model
    fashion_processor = florence_processor



    DATA_ROOT='/data/streetvton'
    files_info = load_files_streetvton(data_root=DATA_ROOT)


    for kind, pair_list in files_info.items():
        print('---- processing {} split ----'.format(kind))
        outfile = open(os.path.join(DATA_ROOT,'{}_captions.json'.format(kind)), mode='w', encoding='utf-8')
        single_image_dict = {}
        unpair_prefilled_image_list = []
        for p_names in tqdm(pair_list):
            agn_image = Image.open(p_names['agn_path']).convert('RGB').resize(size=(768,1024))
            unpair_cloth_image = Image.open(p_names['unpair_cloth_path']).convert('RGB')

            
            unpair_prefilled_image = cat_images(agn_image, unpair_cloth_image)
            
            
            used_name = p_names['agn_path'].split('/')[-1].split('.')[0]
            
            images = [agn_image, unpair_cloth_image, unpair_prefilled_image]

            task_prompt = '<DESCRIPTION>'
            text_prompt_list = [task_prompt + ' Imagine and describe this person’s actions, facial expression, and body posture in the image. \nDo not describe their clothing and the gray region which is masked. \nProvide a detailed and realistic depiction.',
                                task_prompt + ' Describe the fashion garment in this image with great details.',
                                task_prompt + ' Imagine and describe the details feature of the combining the person and right garment. \nDo not include the masked gray region.'
                                ]
            input_prompt = text_prompt_list
            inputs = fashion_processor(images=images, text=input_prompt, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(DEVICE)
            pixel_values = inputs['pixel_values'].to(DEVICE)
            
            generated_ids = fashion_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=256, num_beams=3)
            res = fashion_processor.batch_decode(generated_ids, skip_special_tokens=False)
            res = [fashion_processor.post_process_generation(r, task=task_prompt, image_size=(agn_image.width, agn_image.height)) 
                   for r in res]
            single_image_dict[used_name]={
                'agn':res[0][task_prompt].replace('<pad>',''),
                'cloth':res[1][task_prompt].replace('<pad>',''),
            }
                
            unpair_prefilled_image_list.append(res[2][task_prompt].replace('<pad>',''))
        outdata = {
            'single_image_dict':single_image_dict,
            'unpair_prefilled_image_list':unpair_prefilled_image_list,
            'prefilled_image_list':[]
        }
        json.dump(
            outdata,
            outfile,
            ensure_ascii=False,
            indent=2,
        )
        outfile.close()


def prepare_streetvton_list():
    
    PATH = './data/streetvton'
    infilename = 'shop2street_test_pairs_top.csv'
    outfilename = 'test_pairs.txt'
    with open(os.path.join(PATH, infilename), mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    outstr = ''
    for line in tqdm(lines[1:]):
        index,garment,person = line.strip().split(',')
        outstr += '{} {}\n'.format(person,garment)
    
    outfile = open(os.path.join(PATH, outfilename) , mode='w', encoding='utf-8')
    outfile.write(outstr)
    outfile.close()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    # generate_caption_viton()
    # rewrite_captions()
    # generate_caption_dresscode()
    # prepare_streetvton_list()
    generate_caption_streetvton()