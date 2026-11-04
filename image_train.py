import argparse

import torch
import torch.distributed as dist
import random
import numpy as np

from masked_diffusion import dist_util, logger
from masked_diffusion.resample import create_named_schedule_sampler
from masked_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from masked_diffusion.train_util import TrainLoop
from masked_diffusion import create_diffusion
import masked_diffusion.models as models_dict
import os

from dataset.dataset_vitonhd import VITONHDDataset
from dataset.dataset_dresscode import DressCodeDataset
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer,BertConfig

def set_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def main():
    args,diffusion_defaults, model_defaults = create_argparser()
    args = args.parse_args()
    args.crop_cloth = args.crop_cloth.lower() == 'true'
    args.mask_bkg = args.mask_bkg.lower() == 'true'
    model_and_diffusion_defaults = diffusion_defaults | model_defaults

    dist_util.setup_dist_multinode(args)
    if dist.get_rank() == 0 and not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
        logger.configure(dir=args.work_dir,format_strs=['tensorboard','csv'])
    logger.log("creating model and diffusion...")
    configs = args_to_dict(args, model_and_diffusion_defaults.keys())
    set_seed(args.seed)
    print(configs)
    print(args)
 
    image_size = configs['image_size']
    latent_size = image_size // 8
    bert_config = BertConfig.from_pretrained(args.scene_encoder_path)
    bert_config.layer_norm_eps = 1e-6
    model = models_dict.__dict__[args.model_type](input_size=latent_size, mask_ratio=args.mask_ratio, decode_layer=args.decode_layer, 
                                            is_initial=True,scene_config=bert_config,vit_size=args.vit_size).to(dist_util.dev())

    diffusion = create_diffusion(**args_to_dict(args, diffusion_defaults.keys()))
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    tokenizer = BertTokenizer.from_pretrained(args.scene_encoder_path)
    train_dataset = []

    dataset_kind_list = str(args.dataset_kind).split('_')
    if 'vton' in dataset_kind_list:
        logger.log("loading vton dataset...")
        dataset = VITONHDDataset(
            data_root_dir=os.path.join(args.data_dir, 'zalando-hd-resized'), 
            img_H=args.image_size, 
            img_W=args.image_size, 
            vit_img_H=args.vit_image_size,
            vit_img_W=args.vit_image_size,
            transform_size=args.transform_size, 
            transform_color=args.transform_color,
            tokenizer=tokenizer,
            max_scene_length=args.max_scene_length,
            cat_type=args.cat_type,
            crop_cloth=args.crop_cloth 
        )
        train_dataset.append(dataset)
    if 'dresscode' in dataset_kind_list:
        logger.log("loading dresscode dataset...")
        dc_data = ['DressCode/dresses', 'DressCode/lower_body', 'DressCode/upper_body']
        for d_path in dc_data:
            dataset = DressCodeDataset(
                data_root_dir=os.path.join(args.data_dir, d_path), 
                img_H=args.image_size, 
                img_W=args.image_size, 
                vit_img_H=args.vit_image_size,
                vit_img_W=args.vit_image_size,
                transform_size=args.transform_size, 
                transform_color=args.transform_color,
                tokenizer=tokenizer,
                max_scene_length=args.max_scene_length,
                cat_type=args.cat_type,
                crop_cloth= args.crop_cloth
                )
            train_dataset.append(dataset)
    assert len(train_dataset) > 0
    train_dataset = ConcatDataset(train_dataset)
    train_sampler = DistributedSampler(dataset=train_dataset,
                                 drop_last=False,
                                 shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=64,
        batch_size=args.batch_size,  
        pin_memory=True,
        sampler=train_sampler
    )

    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        opt_type=args.opt_type,
        epochs=args.epochs,
        vae_model_path=args.vae_model_path,
        image_encoder_path=args.image_encoder_path,
        scene_encoder_path = args.scene_encoder_path,
        accumulate=args.accumulate,
        vit_size=args.vit_size,
        warm_up_ratio=args.warm_up_ratio,
        mask_bkg=args.mask_bkg,
        lr_sche_type = args.lr_sche_type,
        min_lr_rate = args.min_lr_rate,
    )
    
    trainer.run_loop()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1, 
        ema_rate="0.9999",  
        save_interval=10000, 
        resume_checkpoint="",
        fp16_scale_growth=1e-3,
        model_type="IStar",
        mask_ratio=None,
        image_size=512,
        vit_image_size=224,
        decode_layer=4,
    )
    diffsuion_defualts_config = {
        'learn_sigma':True,
        'diffusion_steps':1000,
        'noise_schedule':"linear",
        'timestep_respacing':"",
        'use_kl':False,
        'predict_xstart':False,
        'sigma_small':False,
        'rescale_learned_sigmas':False
    }
    image_model_defaults_config = {
        'image_size':256,
        'mask_ratio':None,
        'decode_layer':None,
        'class_cond':True,
    }
    image_model_defaults_config.update(diffsuion_defualts_config)
    defaults.update(image_model_defaults_config)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True, default='experiments', help="Path to saving the expriment")
    parser.add_argument("--seed", type=int, required=True, default=3407)

    #for dataset
    parser.add_argument('--transform_size', default='shiftscale')
    parser.add_argument('--transform_color', default='hsv bright_contrast gaussnoise')
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    
    
    # for training
    parser.add_argument('--opt_type', default='adan', type=str)
    parser.add_argument('--vae_model_path', default='', required=True, type=str)
    parser.add_argument('--image_encoder_path', default='', required=True, type=str)
    parser.add_argument('--scene_encoder_path', default='', required=True, type=str)
    parser.add_argument('--max_scene_length', default=512, required=True, type=int)
    parser.add_argument('--log_interval', default=10, required=True, type=int)
    parser.add_argument('--dataset_kind', default='vton', required=True, type=str)
    parser.add_argument('--accumulate', default=1, required=True, type=int)
    parser.add_argument('--cat_type', default='plain', required=True, type=str)
    parser.add_argument('--crop_cloth', default='True', required=True, type=str, choices=['True','False'])
    parser.add_argument('--min_lr_rate', default=0.01, required=True, type=float)
    parser.add_argument('--warm_up_ratio', default=0.01, required=False, type=float)
    parser.add_argument('--vit_size', default='vitg14', required=True, type=str)
    parser.add_argument('--use_fp16', default=False, required=False, type=bool)
    parser.add_argument('--mask_bkg', default='False', required=True, type=str)
    parser.add_argument('--lr_sche_type', default='cosine', required=True, type=str)
    

    add_dict_to_argparser(parser, defaults)
    return parser,diffsuion_defualts_config, image_model_defaults_config

if __name__ == "__main__":

    main()
