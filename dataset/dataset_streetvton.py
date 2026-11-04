import os
import cv2
import numpy as np
import json
import albumentations as A
from torch.utils.data import Dataset
from .dataset_utils import gen_scene_ids, cat_images_cv2, cat_image_edge,crop_cloth_func


def safe_imread(p: str):
    base, _ = os.path.splitext(p)
    if os.path.exists(p):
        return cv2.imread(p)

    for ext in ('.png', '.jpeg', '.jpg'):  
        alt_path = base + ext
        if alt_path == p:
            continue
        if os.path.exists(alt_path):
            return cv2.imread(alt_path)
    return None

def imread(
        p, h, w, 
        is_mask=False, 
        in_inverse_mask=False, 
        img=None,
):
    if img is None:
        img = safe_imread(p)
    if img is None:
        raise FileNotFoundError(f"Cannot read image from path or alternatives: {p}")

    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

def imread_for_albu(
        p, 
        is_mask=False, 
        in_inverse_mask=False, 
        cloth_mask_check=False, 
        use_resize=False, 
        height=512, 
        width=384,
):
    img = safe_imread(p)
    if img is None:
        raise FileNotFoundError(f"Cannot read image from path or alternatives: {p}")

    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img*255.0)
    return img

def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img

class StreetVTONDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H,          # 512
            img_W,          # 512
            vit_img_H,      # 224
            vit_img_W,      # 224
            is_paired=True, 
            is_test=False, 
            is_sorted=False, 
           transform_content=None, 
            transform_color=None,
            tokenizer=None,
            max_scene_length=512,
            cat_type='plain',
            crop_cloth=False,
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.vit_img_H = vit_img_H
        self.vit_img_W = vit_img_W
        assert not is_paired
        self.pair_key = "paired" if is_paired else "unpaired"
        assert is_test
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0
        self.tokenizer = tokenizer
        self.max_scene_length = max_scene_length
        self.cat_type = cat_type
        self.crop_cloth = crop_cloth

        self.transform_content = transform_content
        self.transform_color = transform_color

        transform_content_input_lst = []
        transform_content_all_lst = []
        transform_color_lst = []
        
        self.transform_content_input_impl = None
        self.transform_content_all_impl = None
        self.transform_color_impl = None
        self.transform_size_impl = None
        
        
        transform_size_lst = [A.Resize(int(img_H*self.resize_ratio_H), int(img_W*self.resize_ratio_W))]



        if self.transform_content is not None:
            if "hflip" in self.transform_content:
                transform_content_all_lst.append(A.HorizontalFlip(p=0.5))

            if "shiftscale" in self.transform_content:
                transform_content_input_lst.append(
                    A.ShiftScaleRotate(
                        rotate_limit=30, 
                        shift_limit=0.2, 
                        scale_limit=(-0.2, 0.2), 
                        border_mode=cv2.BORDER_REPLICATE, 
                        p=0.4)
                    )

            self.transform_content_input_impl = A.Compose(
                    transform_content_input_lst,
                    additional_targets={"agn":"image", 
                                        "agn_mask":"image", 
                                        "image_densepose":"image", 
                                        }
            )
            self.transform_content_all_impl = A.Compose(
                transform_content_all_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "image_densepose":"image", 
                                    "cloth":"image",
                                    }
            )

        self.transform_size_impl = A.Compose(
            transform_size_lst,
            additional_targets={"agn":"image", 
                                "agn_mask":"image",
                                "image_densepose":"image", 
                                "cloth":"image",
                                },
            is_check_shapes=False
        )
        
        if self.transform_color is not None:
            if 'hsv' in transform_color:
                transform_color_lst.append(A.HueSaturationValue(5,5,5,p=0.5))
            if "bright_contrast" in transform_color:
                transform_color_lst.append(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))
            if 'gaussnoise' in transform_color:
                transform_color_lst.append(A.GaussNoise(std_range=(0.01,0.1),mean_range=(0,0), per_channel=True, p=0.5))
            self.transform_color_impl = A.Compose(
                transform_color_lst,
                additional_targets={"agn":"image",
                                    "cloth":"image",
                                    "agn_cloth":"image",
                                    },
                is_check_shapes=False
            )
         
           
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(os.path.join(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names
        
        self.single_image_dict, self.prefilled_image_list, self.unpair_prefilled_image_list = self.__load_scene_desc()
        
    def __load_scene_desc(self):
        scene_desc_filename = os.path.join(self.drd,'{}_captions.json'.format(self.data_type))
        infile = open(scene_desc_filename, mode='r', encoding='utf-8')
        indata = json.load(infile)
        infile.close()
        single_image_dict, prefilled_image_list, unpair_prefilled_image_list = indata['single_image_dict'], \
                                                                                indata['prefilled_image_list'], \
                                                                                indata['unpair_prefilled_image_list']
        return single_image_dict, prefilled_image_list, unpair_prefilled_image_list

    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        
        
        #### process secne desc
        agn_desc = self.single_image_dict[img_fn.split('.')[0]]['agn']
        cloth_desc = self.single_image_dict[img_fn.split('.')[0]]['cloth']
        if self.pair_key == 'unpaired':
            pre_filled_desc = self.unpair_prefilled_image_list[idx]
        else:
            pre_filled_desc = self.prefilled_image_list[idx]
        scene_ids, scene_attention_mask, scene_token_types = gen_scene_ids(
            cloth_desc=cloth_desc,
            agn_desc=agn_desc,
            pre_filled_desc = pre_filled_desc,
            tokenzer=self.tokenizer,
            max_length=self.max_scene_length
        )
        
        agn = imread_for_albu(os.path.join(self.drd, self.data_type, "agnostic", self.im_names[idx].replace('.jpg','.png')))
        agn_mask = imread_for_albu(os.path.join(self.drd, self.data_type, "mask", self.im_names[idx].replace(".jpg", ".png")), is_mask=True)
        cloth = imread_for_albu(os.path.join(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
        # cloth = image_wavelet(cloth)
        image = imread_for_albu(os.path.join(self.drd, self.data_type, "image", self.im_names[idx]))
        image_densepose = imread_for_albu(os.path.join(self.drd, self.data_type, "image-densepose", self.im_names[idx]))
        
        if self.crop_cloth:
            cloth_mask = imread_for_albu(os.path.join(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]))
            cloth = crop_cloth_func(cloth=cloth, cloth_mask=cloth_mask)
            
        if self.transform_size_impl is not None:
            transformed = self.transform_size_impl(
                image=image, 
                agn=agn, 
                agn_mask=agn_mask, 
                image_densepose=image_densepose,
                cloth=cloth, 
            )

            image=transformed["image"]
            agn=transformed["agn"]
            agn_mask=transformed["agn_mask"]
            image_densepose=transformed["image_densepose"]
            cloth=transformed["cloth"]
            
        if self.transform_content_input_impl is not None:
            transformed_image = self.transform_content_input_impl(
                image=image,
                agn=agn,
                agn_mask=agn_mask,
                image_densepose=image_densepose,
            )

            image=transformed_image["image"]
            agn=transformed_image["agn"]
            agn_mask=transformed_image["agn_mask"]
            image_densepose=transformed_image["image_densepose"]
            
        if self.transform_content_all_impl is not None:
            transformed_image = self.transform_content_all_impl(
                image=image,
                agn=agn,
                agn_mask=agn_mask,
                image_densepose=image_densepose,
                cloth=cloth,
            )

            image=transformed_image["image"]
            agn=transformed_image["agn"]
            agn_mask=transformed_image["agn_mask"]
            image_densepose=transformed_image["image_densepose"]
            cloth=transformed_image["cloth"]
  
        agn_mask = 255 - agn_mask
        if self.transform_color_impl is not None:
            transformed = self.transform_color_impl(
                image=person_image, 
                agn=agn, 
                cloth=cloth,
            )


            person_image=transformed["image"]
            agn=transformed["agn"]
            cloth=transformed["cloth"]

            agn = agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)
        
        if self.cat_type == 'plain':
            agn_cloth = cat_images_cv2(image1=agn, image2=cloth)
        elif self.cat_type =='edge':
            agn_cloth = cat_image_edge(agn=agn, mask=agn_mask, cloth=cloth)    
        
        cloth_generation = cloth
        agn_cloth_generation = agn_cloth
        cloth=cv2.resize(cloth, (self.vit_img_H, self.vit_img_W))
        agn_cloth=cv2.resize(agn_cloth, (self.vit_img_H, self.vit_img_W))

        agn = norm_for_albu(agn)
        agn_mask = norm_for_albu(agn_mask, is_mask=True)
        image = norm_for_albu(image)
        image_densepose = norm_for_albu(image_densepose)
        agn_cloth_generation = norm_for_albu(agn_cloth_generation)
        cloth = norm_for_albu(cloth)
        agn_cloth = norm_for_albu(agn_cloth)
        cloth_generation = norm_for_albu(cloth_generation)
        agn_mask = np.repeat(agn_mask, 3, axis=2) 
        
        return dict(
            agn=agn.transpose((2,0,1)),                         # A gray-cloth masked
            agn_mask=agn_mask.transpose((2,0,1)),               # A white-black  white others
            image=image.transpose((2,0,1)),                     # A final colorful result
            image_densepose=image_densepose.transpose((2,0,1)), # A segment, body segment
            cloth=cloth.transpose((2,0,1)),                     # B cloth color
            agn_cloth=agn_cloth.transpose((2,0,1)),             # agn
            agn_cloth_generation = agn_cloth_generation.transpose((2,0,1)),
            img_fn=img_fn,                                      # A name
            cloth_fn=cloth_fn,                                  # B name
            scene_ids=scene_ids,
            scene_attention_mask=scene_attention_mask,
            scene_token_types=scene_token_types,
            cloth_generation=cloth_generation.transpose((2,0,1))
        )
    