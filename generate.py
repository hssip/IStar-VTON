import torch
from torchvision.utils import save_image
from masked_diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from dinov2.backbones import dinov2_vitg14, dinov2_vitl14, dinov2_vitb14, dinov2_vits14
from transformers import BertTokenizer,BertConfig,BertModel
from dataset.dataset_vitonhd import VITONHDDataset
from dataset.dataset_dresscode import DressCodeDataset
from dataset.dataset_streetvton import StreetVTONDataset
import os
from torch.cuda.amp import autocast
from importlib import import_module
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from torch.utils.data import SequentialSampler
import masked_diffusion.models as models_dict
from PIL import Image
from diffusers import AutoencoderKLFlux2

class VAE:
    def __init__(self, vae, device=None):
        self.vae = vae
        self.scaling_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 1.0
        self.device = device

    def encode(self, x):
        z = self.vae.encode(x).latent_dist.sample()
        z = z * self.scaling_factor
        if self.device:
            z = z.to(self.device)
        return z

    def decode(self, z):
        z = z / self.scaling_factor
        if self.device:
            z = z.to(self.device)
        return self.vae.decode(z).sample

# Function to shrink the mask by eroding it
def shrink_mask(mask, erosion_width=4):
    # Erode the mask by applying max pooling with a kernel size equal to the erosion width
    eroded_mask = F.max_pool2d(1 - mask.unsqueeze(0).float(), kernel_size=erosion_width, stride=1, padding=erosion_width // 2).squeeze(0)
    eroded_mask = 1 - eroded_mask  # Revert the mask to original values
    return eroded_mask.clamp(0, 1)  # Ensure the values remain in [0, 1]

def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    model_path = args.model_path
    image_size = args.image_size
    vit_img_size = args.vit_img_size
    num_sampling_steps = args.num_sampling_steps
    cfg_scale = args.cfg_scale
    pow_scale = args.pow_scale
    batch_size = args.batch_size
    unpair = args.unpair
    generate_dataset = args.generate_dataset
    generate_subdata = args.generate_subdata
    assert generate_dataset in ['vitonhd','dresscode','streetvton']
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_H = image_size
    img_W = image_size
    latent_size = image_size // 8
    vit_img_H = vit_img_size
    vit_img_W = vit_img_size

    parent, filename = os.path.split(model_path) 
    model_name, _ = os.path.splitext(filename) 
    _, parent_folder = os.path.split(parent)
    
    bert_config = BertConfig.from_pretrained(args.scene_encoder_path)
    scene_encoder = BertModel(bert_config, add_pooling_layer=False).to(device)
    state = torch.load(os.path.join(args.scene_encoder_path,'pytorch_model.bin'),
                           map_location='cpu')
    state = {k.replace('bert.', ''):v for k,v in state.items() 
                if 'pooler.' not in k and 'cls.' not in k}
    state = {k.replace('.beta', '.bias'):v for k,v in state.items()}
    state = {k.replace('.gamma', '.weight'):v for k,v in state.items()}
    scene_encoder.load_state_dict(state, strict=True)
    scene_encoder.eval()
    for param in scene_encoder.parameters():
        param.requires_grad = False
    
    model = models_dict.__dict__[args.model_type](input_size=latent_size, 
                                            decode_layer=args.decode_layer,
                                            scene_config=bert_config,
                                            vit_size=args.vit_size,
                                            in_channels=32 if 'flux.2-dev' in args.vae_path else 4).to(device)

    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(num_sampling_steps))
    if 'flux.2-dev' not in args.vae_path:
        vae_ = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=args.vae_path).to(device)
    else:
        vae_ = AutoencoderKLFlux2.from_pretrained(pretrained_model_name_or_path=args.vae_path).to(device) 
    vae = VAE(vae_, device=device)
    vae_.eval()
    
    if args.vit_size == 'vitg14':
        cloth_encoder = dinov2_vitg14().to(device)
    elif args.vit_size == 'vitl14':
        cloth_encoder = dinov2_vitl14().to(device)
    elif args.vit_size == 'vitb14':
        cloth_encoder = dinov2_vitb14().to(device)
    elif args.vit_size == 'vits14':
        cloth_encoder = dinov2_vits14().to(device)
    state_dict = torch.load(os.path.join(args.cloth_encoder_path, 'dinov2_{}_pretrain.pth'.format(args.vit_size)),
                                map_location='cpu')
    cloth_encoder.load_state_dict(state_dict, strict=True)
    cloth_encoder.eval()
    for param in cloth_encoder.parameters():
        param.requires_grad = False
    
    unpair = True if generate_dataset == 'streetvton' else unpair
    if unpair:
        save_dir = os.path.join(output_dir, f"{parent_folder}_{model_name}{args.gen_name}", generate_dataset, 'unpaired', generate_subdata)
    else:
        save_dir = os.path.join(output_dir, f"{parent_folder}_{model_name}{args.gen_name}", generate_dataset, 'paired', generate_subdata)
    num_samples = batch_size
    tokenizer = BertTokenizer.from_pretrained(args.scene_encoder_path)
    if generate_dataset == 'vitonhd':
        dataset = VITONHDDataset(
            data_root_dir=os.path.join(data_dir, 'zalando-hd-resized'), 
            img_H=img_H,
            img_W=img_W,
            vit_img_H=vit_img_H,
            vit_img_W=vit_img_W,
            is_paired=not unpair,
            is_test=True,
            is_sorted=True,
            tokenizer=tokenizer,
            max_scene_length=args.max_scene_length,
            cat_type=args.cat_type,
            crop_cloth=args.crop_cloth
            )
    elif generate_dataset == 'dresscode':
        dataset = DressCodeDataset(
            data_root_dir=os.path.join(data_dir, 'DressCode', generate_subdata), 
            img_H=img_H,
            img_W=img_W,
            vit_img_H=vit_img_H,
            vit_img_W=vit_img_W,
            is_paired=not unpair,
            is_test=True,
            is_sorted=True,
            tokenizer=tokenizer,
            max_scene_length=args.max_scene_length,
            cat_type=args.cat_type,
            crop_cloth=args.crop_cloth# and generate_subdata != 'dresses'
            )
    elif generate_dataset == 'streetvton':
        dataset = StreetVTONDataset(
            data_root_dir=os.path.join(data_dir, 'streetvton'), 
            img_H=img_H,
            img_W=img_W,
            vit_img_H=vit_img_H,
            vit_img_W=vit_img_W,
            is_paired=not unpair,
            is_test=True,
            is_sorted=True,
            tokenizer=tokenizer,
            max_scene_length=args.max_scene_length,
            cat_type=args.cat_type,
            crop_cloth=args.crop_cloth
            )
    sampler = SequentialSampler(
        data_source=dataset,
    )
    dataloader = DataLoader(
        dataset,
        num_workers=64, 
        batch_size=max(batch_size, 1), 
        shuffle=False, 
        pin_memory=True,
        sampler=sampler
        )

    cond_keys = ['agn', 'agn_mask', 'image_densepose', 
                'cloth', 'agn_cloth', 
                'scene_ids', 'scene_attention_mask', 'scene_token_types',
                'cloth_generation',
            ]

    total_iter = len(dataloader)
    total_image = len(dataset)
    generated_image_count = 0 
    for i, batch in enumerate(dataloader):
        cond = {key: batch[key].to(device) for key in cond_keys}

        batch_size_now = len(batch['cloth_fn']) 
        generated_image_count += batch_size_now

        print(f'Iteration {i}/{total_iter}, {generated_image_count}/{total_image} images.')

        repaint_agn = cond['agn']
        repaint_agn_mask = cond['agn_mask']

        if len(batch['cloth_fn']) != num_samples:
            num_samples = len(batch['cloth_fn'])

        with torch.no_grad():
            cond['agn'] = vae.encode(cond['agn'])
            cond['agn_mask'] = vae.encode(cond['agn_mask'])
            cond['image_densepose'] = vae.encode(cond['image_densepose'])
            cond['cloth_generation'] = vae.encode(cond['cloth_generation'])
            
            outputs = scene_encoder(
                input_ids = cond['scene_ids'],
                attention_mask = cond['scene_attention_mask'],
                token_type_ids = cond['scene_token_types'],
                return_dict=True
                )
            cond['scene_states'] = outputs.last_hidden_state
            with autocast(dtype=torch.bfloat16):
                cond['cloth'] = torch.stack(cloth_encoder(cond['cloth'])[1],
                                            dim=1)
                cond['agn_cloth'] = cloth_encoder(cond['agn_cloth'])[0]
            

        # Create sampling noise
        channels=32 if 'flux.2-dev' in args.vae_path else 4
        z = torch.randn(num_samples, channels, latent_size, latent_size, device=device)

        model_kwargs = dict(cfg_scale=cfg_scale, scale_pow=pow_scale, 
                            cond=cond
                            )

        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples = vae.decode(samples)


        if True: 
            repainted_samples = []
            for i, (sample_img, agn_img, mask_img) in enumerate(zip(samples, repaint_agn, repaint_agn_mask)):
                sample_img = sample_img.detach().cpu()
                agn_img = agn_img.detach().cpu()
                mask_img = mask_img.detach().cpu()

                # Resize the mask if necessary to match the sample size
                if mask_img.shape != sample_img.shape:
                    mask_img = F.interpolate(mask_img.unsqueeze(0), size=sample_img.shape[1:], mode='nearest').squeeze(0)

                # Shrink the mask to exclude boundary pixels
                smaller_mask = shrink_mask(mask_img, erosion_width=4)

                # Resize smaller_mask to match sample_img size to avoid dimension mismatch
                if smaller_mask.shape != sample_img.shape:
                    smaller_mask = F.interpolate(smaller_mask.unsqueeze(0), size=sample_img.shape[1:], mode='nearest').squeeze(0)

                # Blend the images using the smaller mask
                repainted_img = agn_img * smaller_mask + sample_img * (1 - smaller_mask)

                # Append the repainted image to the output list
                repainted_samples.append(repainted_img)

            repainted_samples = torch.stack(repainted_samples).to(samples.device)
            samples = repainted_samples

        # Save 
        os.makedirs(save_dir, exist_ok=True)
        for i, img_fn in enumerate(batch['img_fn']):
            new_filename = img_fn.replace('.jpg','.png')
            new_image_name = os.path.join(save_dir, new_filename)
            save_image(samples[i].squeeze(), new_image_name, normalize=True, value_range=(-1, 1))
        
    print('dataset:{}, paire:{}, subdata:{} done!'.format(generate_dataset, ('unpaired' if unpair else 'paired'), generate_subdata))


if __name__ == '__main__':
    # argparse argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--vit_img_size', type=int)
    parser.add_argument('--num_sampling_steps', type=int)
    parser.add_argument('--cfg_scale', type=float)
    parser.add_argument('--pow_scale', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unpair', type=str, choices=['true', 'false', 'True', 'False'],help="Must be 'true' or 'false'.")
    parser.add_argument('--vae_path', type=str)
    parser.add_argument('--cloth_encoder_path', type=str)
    parser.add_argument('--scene_encoder_path', type=str)
    parser.add_argument('--max_scene_length', type=int)
    parser.add_argument('--generate_dataset', type=str)
    parser.add_argument('--generate_subdata', type=str, choices=['dresses', 'upper_body', 'lower_body'])
    parser.add_argument('--cat_type', type=str, choices=['plain', 'edge'])
    parser.add_argument('--crop_cloth', type=str)
    parser.add_argument('--vit_size', type=str)
    parser.add_argument('--gen_name', type=str)
    parser.add_argument('--sub_gen', type=int)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--decode_layer', type=int)
    parser.add_argument('--generate_height', type=int)
    parser.add_argument('--generate_width', type=int)
    args = parser.parse_args() 
    args.unpair = args.unpair.lower() == 'true'  # Convert string to boolean
    args.crop_cloth = args.crop_cloth.lower() == 'true'  # Convert string to boolean
    args.sub_gen = args.sub_gen == 1  # Convert string to boolean
    print(args)
    main(args)