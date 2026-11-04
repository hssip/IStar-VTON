import argparse
import json
import os
from typing import List, Tuple, Dict

import PIL.Image
import torch
import fid
from torch.utils.data import DataLoader 
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm
import shutil
import time

USED_CATEGORY = ['mix', 'dresses', 'lower_body', 'upper_body']
torch.hub.set_dir(os.path.dirname(__file__))

def make_custom_stats_fid(dresscode_dataroot: str, 
                          vitonhd_dataroot: str, 
                          streetvton_dataroot: str,
                          save_path=None):
    if dresscode_dataroot is not None:
        for category in ['lower_body', 'upper_body', 'dresses']:
            dresscode_filesplit = os.path.join(dresscode_dataroot, category,f"test_pairs_paired.txt")
            with open(dresscode_filesplit, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
            if not fid.test_stats_exists(f"dresscode_{category}", mode='clean'):
                paths = [os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]) for line in lines]
                tmp_folder = f"/tmp/dresscode/{category}"
                os.makedirs(tmp_folder, exist_ok=True)
                for path in tqdm(paths):
                    shutil.copy(path, tmp_folder)
                fid.make_custom_stats(f"dresscode_{category}", tmp_folder, mode="clean", verbose=True, save_path=save_path)

    if vitonhd_dataroot is not None:
        if not fid.test_stats_exists(f"vitonhd_mix", mode='clean'):
            fid.make_custom_stats(f"vitonhd_mix", os.path.join(vitonhd_dataroot, 'test', 'image'), mode="clean",
                                  verbose=True, save_path=save_path)
            
    if streetvton_dataroot is not None:
        if not fid.test_stats_exists(f"streetvton_mix", mode='clean'):
            fid.make_custom_stats(f"streetvton_mix", os.path.join(streetvton_dataroot, 'test', 'image-shop2street'), mode="clean",
                                  verbose=True, save_path=save_path)
        
def make_custom_stats_kid(dresscode_dataroot: str = None, 
                          vitonhd_dataroot: str = None,
                          streetvton_dataroot: str = None,
                          subdata: str=None):
    
    if dresscode_dataroot is not None:
        assert subdata is not None
        category = subdata
        dresscode_filesplit = os.path.join(dresscode_dataroot, category,f"test_pairs_paired.txt")
        with open(dresscode_filesplit, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        tmp_folder = os.path.join(os.path.dirname(__file__),"temp/dresscode/".format(category))
        paths = [os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]) for line in lines]
        tmp_folder = os.path.join(os.path.dirname(__file__),"temp/dresscode/".format(category))
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
            for path in tqdm(paths):
                shutil.copy(path, tmp_folder)
        return tmp_folder
        

    if vitonhd_dataroot is not None:
        return os.path.join(vitonhd_dataroot, 'test', 'image')
    
    if streetvton_dataroot is not None:
        return os.path.join(streetvton_dataroot, 'test', 'image-shop2street')
    



class GTTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot: str, dataset: str, category: str, transform: transforms.Compose):
        assert dataset in ['dresscode', 'vitonhd', 'streetvton'], 'Unsupported dataset'
        assert category in USED_CATEGORY, 'Unsupported category'

        self.dataset = dataset
        self.category = category
        self.transform = transform
        self.paths = []
        if dataset == 'dresscode':
            assert category in ['lower_body', 'upper_body', 'dresses']
            filepath = os.path.join(dataroot, f"test_pairs_paired.txt")
            with open(filepath, 'r') as f:
                lines = f.readlines()
            self.paths = sorted(
                [os.path.join(dataroot, category, 'images', line.strip().split()[0]) for line in lines if
                    os.path.exists(os.path.join(dataroot, category, 'images', line.strip().split()[0]))])
        elif dataset == 'vitonhd':
            filepath = os.path.join(dataroot, f"test_pairs.txt")
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()
            self.paths = sorted([os.path.join(dataroot, 'test', 'image', line.strip().split()[0]) for line in lines])
        elif dataset == 'streetvton':
            filepath = os.path.join(dataroot, f"test_pairs.txt")
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()
            self.paths = sorted([os.path.join(dataroot, 'test', 'image-shop2street', line.strip().split()[0]) for line in lines])

        # Debug print
        print(f"GTTestDataset initialized with {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        img = self.transform(PIL.Image.open(path).convert('RGB'))
        return img, name

class GenTestDataset(torch.utils.data.Dataset):
    def __init__(self, gen_folder: str, dataset: str, category: str, transform: transforms.Compose):
        assert dataset in ['dresscode', 'vitonhd', 'streetvton'], 'Unsupported dataset'
        assert category in USED_CATEGORY, 'Unsupported category'

        self.dataset = dataset
        self.category = category
        self.transform = transform
        self.gen_folder = gen_folder
        self.paths = []

        if dataset == 'dresscode':
            assert category in ['lower_body', 'upper_body', 'dresses']
            self.paths = sorted(
                [os.path.join(gen_folder, category, name) 
                 for name in os.listdir(os.path.join(gen_folder, category)) 
                 if name.lower().endswith(('.png', '.jpg', '.jpeg'))])
        elif dataset == 'vitonhd':

            self.paths = sorted(
                [os.path.join(gen_folder, name) 
                 for name in os.listdir(gen_folder) 
                 if name.lower().endswith(('.png', '.jpg', '.jpeg'))])
        elif dataset == 'streetvton':

            self.paths = sorted(
                [os.path.join(gen_folder, name) 
                 for name in os.listdir(gen_folder) 
                 if name.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # Debug print to confirm the number of images found
        print(f"GenTestDataset initialized with {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        img = self.transform(PIL.Image.open(path).convert('RGB'))
        return img, name

def compute_metrics(gen_folder: str, 
                    test_order: str, 
                    dataset: str, 
                    category: str, 
                    metrics2compute: List[str],
                    generated_size: Tuple[int, int] = (512, 512),
                    batch_size: int = 16,
                    workers: int = 8,
                    dresscode_dataroot:str=None,
                    vitonhd_dataroot:str=None,
                    streetvton_dataroot:str=None,
                    gt_folder:str=None) -> Dict[str, float]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert test_order in ['paired', 'unpaired']
    assert dataset in ['dresscode', 'vitonhd', 'streetvton'], 'Unsupported dataset'
    assert category in USED_CATEGORY, 'Unsupported category'


    for m in metrics2compute:
        assert m in ['all_metric', 
                     'ssim_score', 
                     'fid_score',
                     'psnr_score',
                     'kid_score',
                     ], 'Unsupported metric'
    if 'all_metric' in metrics2compute:
        if test_order == 'paired':
            metrics2compute = ['ssim_score', 
                               'fid_score', 
                               'psnr_score',
                               'kid_score', 
                               ]
        elif test_order == 'unpaired':
            metrics2compute = ['fid_score', 'kid_score']
            
    if dataset == 'vitonhd':
        gt_folder = vitonhd_dataroot
    elif dataset == 'streetvton':
        gt_folder = streetvton_dataroot
        assert test_order == 'unpaired'
    elif dataset == 'dresscode':
        gt_folder = dresscode_dataroot
    else:
        raise ValueError('can not process dataset {}'.format(dataset))

    if 'fid_score' in metrics2compute or 'kid_score' in metrics2compute:
        print('make custom stats!')
        make_custom_stats_fid(dresscode_dataroot=dresscode_dataroot, 
                              vitonhd_dataroot=vitonhd_dataroot, 
                              streetvton_dataroot=streetvton_dataroot,
                              save_path=None)
        print('computing fid score!')
        if 'fid_score' in metrics2compute:
            fid_score = fid.compute_fid(os.path.join(gen_folder, category if dataset == 'dresscode' else ''), 
                                        dataset_name=f"{dataset}_{category}", mode='clean', dataset_split="custom",
                                            verbose=True, use_dataparallel=False)
        print('computing kid score!')
        if 'kid_score' in metrics2compute:
            kid_score = fid.compute_kid(os.path.join(gen_folder, category if dataset == 'dresscode' else ''), 
                                        dataset_name=f"{dataset}_{category}", mode='clean', dataset_split="custom",
                                            verbose=True, use_dataparallel=False) * 1000

        
    if test_order == 'paired':
        trans = transforms.Compose([
            transforms.Resize(generated_size),
            transforms.ToTensor(),
        ])
        
        print("gen_folder path:", gen_folder)
        print("gt_folder path:", gt_folder)
        gen_dataset = GenTestDataset(gen_folder, dataset, category, transform=trans)
        gt_dataset = GTTestDataset(gt_folder, dataset, category, transform=trans)

        gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        gt_loader = DataLoader(gt_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        assert len(gen_loader) == len(gt_loader)
        
        for i,f in enumerate(gen_loader.dataset.paths):
            if f.split('/')[-1].split('.')[0] != gt_loader.dataset.paths[i].split('/')[-1].split('.')[0]:
                print(f)
                print(gt_loader.dataset.paths[i])
                raise ValueError('error!')

        if "ssim_score" in metrics2compute or "all" in metrics2compute:
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        if "lpips_score_s" in metrics2compute or "all" in metrics2compute:
            lpips_s = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True,
                                                            ).to(device)
            
        if "psnr_score" in metrics2compute or "all" in metrics2compute:
            psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

        # Debug print for data loader size
        print(f"Total batches in gen_loader: {len(gen_loader)}")
        print(f"Total batches in gt_loader: {len(gt_loader)}")
        for idx, (gen_batch, gt_batch) in tqdm(enumerate(zip(gen_loader, gt_loader)), total=len(gt_loader)):
            gen_images, gen_names = gen_batch
            gt_images, gt_names = gt_batch
            
            if len(gen_images) == 0 or len(gt_images) == 0:
                print(f"Empty batch at index {idx}")
                continue

            assert gen_names == gt_names, f"Names do not match at index {idx}"

            gen_images = gen_images.to(device)
            gt_images = gt_images.to(device)

            if "ssim_score" in metrics2compute or "all" in metrics2compute:
                ssim.update(gen_images, gt_images)

            if "lpips_score_s" in metrics2compute or "all" in metrics2compute:
                lpips_s.update(gen_images, gt_images)
            if "psnr_score" in metrics2compute or "all" in metrics2compute:
                psnr.update(gen_images, gt_images)

        
        if "ssim_score" in metrics2compute or "all" in metrics2compute:
            ssim_score = ssim.compute()
        if "lpips_score_s" in metrics2compute or "all" in metrics2compute:
            lpips_score_s = lpips_s.compute()
        if "psnr_score" in metrics2compute or "all" in metrics2compute:
            psnr_score = psnr.compute()

    results = {}

    for m in metrics2compute:
        if torch.is_tensor(locals()[m]):
            results[m] = locals()[m].item()
        else:
            results[m] = locals()[m]
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the metrics for the generated images")
    parser.add_argument("--gen_folder", default='', type=str, help="Path to the generated images")
    parser.add_argument("--gt_folder", default='', type=str, help="Path to the generated images")
    parser.add_argument('--dresscode_dataroot', default='', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', default='', type=str, help='VitonHD dataroot')
    parser.add_argument('--streetvton_dataroot', default='', type=str, help='streetvton_dataroot dataroot')
    parser.add_argument("--test_order", default='paired', type=str, choices=['paired', 'unpaired'])
    parser.add_argument("--dataset", type=str, default='vitonhd', choices=['dresscode', 'vitonhd', 'streetvton'], help="Dataset to use for the metrics")
    parser.add_argument("--category", type=str, default='all', choices=['all', 'lower_body', 'upper_body', 'dresses'])
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the dataloaders")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for the dataloaders")
    parser.add_argument("--eval_image_height", type=int, default=512, help="Number of workers for the dataloaders")
    parser.add_argument("--eval_image_width", type=int, default=512, help="Number of workers for the dataloaders")

    args = parser.parse_args()


    GENERATED_SIZE = (args.eval_image_height, args.eval_image_width)
    if (args.dataset == "vitonhd" or args.dataset == 'streetvton'):
        args.category = 'mix'  # vitonhd does not have sub-categories

    if not os.path.exists(args.gen_folder):
        raise ValueError("The generated images folder does not exist")

    metrics = compute_metrics(gen_folder=args.gen_folder,
                              gt_folder=args.gt_folder, 
                              test_order=args.test_order, 
                              dataset=args.dataset, 
                              category=args.category, 
                              metrics2compute=['all_metric'],
                              dresscode_dataroot=args.dresscode_dataroot, 
                              vitonhd_dataroot=args.vitonhd_dataroot, 
                              streetvton_dataroot=args.streetvton_dataroot, 
                              batch_size=args.batch_size,
                              workers=args.workers,
                              generated_size=GENERATED_SIZE)
    print(args)
    print(args.gen_folder)
    print('generated_size: {}'.format(GENERATED_SIZE))
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    with open(os.path.join(args.gen_folder, f"metrics_{args.test_order}_{args.category}.json"), "w+") as f:
        json.dump(metrics, f, indent=4)
     
    time.sleep(0.5)