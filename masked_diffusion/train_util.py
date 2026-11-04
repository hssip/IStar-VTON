import copy
import functools
import os
import blobfile as bf
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import SGD, AdamW
from torch.amp import autocast
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler 
from adan import Adan
from torch.distributed.optim import ZeroRedundancyOptimizer

from diffusers.models import AutoencoderKL
from diffusers import AutoencoderKLFlux2
from transformers import Dinov2Model
from transformers import BertModel, BertTokenizer,BertConfig
from dinov2.backbones import dinov2_vitg14, dinov2_vitl14, dinov2_vitb14, dinov2_vits14
from transformers.optimization import (
    get_cosine_with_min_lr_schedule_with_warmup, 
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule
)
from masked_diffusion.models import IStar


INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model:IStar,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        scale_factor=0.18215, 
        opt_type='adan',
        use_zero=True,
        epochs=30,
        vae_model_path=None,
        image_encoder_path=None,
        scene_encoder_path=None,
        accumulate=1,
        min_lr_rate=0.01,
        vit_size='vitg14',
        warm_up_ratio=0.02,
        mask_bkg=False,
        lr_sche_type='cosine',
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.cur_lr = self.lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.scale_factor = scale_factor
        self.vae_model_path = vae_model_path
        self.image_encoder_path = image_encoder_path
        self.scene_encoder_path = scene_encoder_path
        self.accumulate = accumulate
        self.epochs = epochs
        self.min_lr_rate = min_lr_rate
        self.vit_size = vit_size
        self.opt_type = opt_type
        self.warm_up_ratio = warm_up_ratio
        self.mask_bkg = mask_bkg
        self.lr_sche_type = lr_sche_type
        self.init_recons_weight = (1.0 / self.diffusion.num_timesteps) * 0.001
        self.recons_epoch = [0]
        

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = torch.cuda.is_available()
        
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        if opt_type=='adamw':
            if use_zero:
                self.opt = ZeroRedundancyOptimizer(
                    self.mp_trainer.master_params,
                    optimizer_class=AdamW,
                    lr=self.lr,
                     weight_decay=self.weight_decay
                )
            else:
                self.opt = AdamW(
                    self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
                )
        elif opt_type=='adan':
            if use_zero:
                self.opt = ZeroRedundancyOptimizer(
                    self.mp_trainer.master_params,
                    optimizer_class=Adan,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    max_grad_norm=1, fused=True
                )
                
            else:
                self.opt = Adan(
                    self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay, max_grad_norm=1, fused=True)
        
        elif opt_type=='sgd':
            if use_zero:
                self.opt = ZeroRedundancyOptimizer(
                    self.mp_trainer.master_params,
                    optimizer_class=SGD,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    momentum=0.9, nesterov=True
                )
                
            else:
                self.opt = SGD(
                    self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay,
                    momentum=0.9, nesterov=True)
        
        num_training_steps = self.epochs * len(self.data) // self.accumulate
        if self.lr_sche_type == 'cosine':
            self.lr_sche = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer=self.opt,
                num_warmup_steps= self.warm_up_ratio * num_training_steps,
                num_training_steps=num_training_steps,
                min_lr_rate=self.min_lr_rate
            )
        elif self.lr_sche_type == 'poly':
            self.lr_sche = get_polynomial_decay_schedule_with_warmup(
                optimizer=self.opt,
                num_warmup_steps=self.warm_up_ratio * num_training_steps,
                num_training_steps=num_training_steps,
                lr_end=self.min_lr_rate * self.lr,
                power=2
            )
        elif self.lr_sche_type == 'constant':
            self.lr_sche = get_constant_schedule(
                optimizer=self.opt)
        else:
            raise ValueError('can not use scheduler: {} '.format(self.lr_sche_type))
        self.initial_load_model()
        self._load_and_sync_parameters()
        if torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        
        
        if self.resume_step != 0:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        
        
    def initial_load_model(self):
        print('loading vae ...')
        if 'flux.2-dev' not in  self.vae_model_path:
            self.source_encoder = AutoencoderKL.from_pretrained(pretrained_model_name_or_path=self.vae_model_path).to(dist_util.dev())
        else:
            self.source_encoder = AutoencoderKLFlux2.from_pretrained(pretrained_model_name_or_path=self.vae_model_path).to(dist_util.dev()) 
        self.source_encoder.eval()
        self.source_encoder.train = False
        for param in self.source_encoder.parameters():
            param.requires_grad = False
        
        print('loading image_encoder ...'))
        if self.vit_size == 'vitg14':
            self.cloth_encoder = dinov2_vitg14().to(dist_util.dev())
        elif self.vit_size == 'vitl14':
            self.cloth_encoder = dinov2_vitl14().to(dist_util.dev())
        elif self.vit_size == 'vitb14':
            self.cloth_encoder = dinov2_vitb14().to(dist_util.dev())
        elif self.vit_size == 'vits14':
            self.cloth_encoder = dinov2_vits14().to(dist_util.dev())
        if 'vit' in self.vit_size:
            state_dict = torch.load(os.path.join(self.image_encoder_path, 'dinov2_{}_pretrain.pth'.format(self.vit_size)),
                                    map_location='cpu')
            self.cloth_encoder.load_state_dict(state_dict, strict=True)
        self.cloth_encoder = self.cloth_encoder.to(dist_util.dev())
        self.cloth_encoder.eval()
        for param in self.cloth_encoder.parameters():
            param.requires_grad = False
        
        print('loading scene encoder ...')
        bert_config = BertConfig.from_pretrained(self.scene_encoder_path)
        self.scene_encoder = BertModel(bert_config, add_pooling_layer=False).to(dist_util.dev())
        self.scene_encoder.eval()
        for param in self.scene_encoder.parameters():
            param.requires_grad = False
        
        state = torch.load(os.path.join(self.scene_encoder_path,'pytorch_model.bin'),
                           map_location='cpu')
        state = {k.replace('bert.', ''):v for k,v in state.items() 
                 if 'pooler.' not in k and 'cls.' not in k}
        state = {k.replace('.beta', '.bias'):v for k,v in state.items()}
        state = {k.replace('.gamma', '.weight'):v for k,v in state.items()}
        
        self.scene_encoder.load_state_dict(state, strict=True)
        
            
    @torch.no_grad()
    def source_encode(self, x):
            encoder_posterior = self.source_encoder.encode(x).latent_dist.sample()     
            z = encoder_posterior
            scaling_factor = 1. if 'flux.2-dev' in  self.vae_model_path else self.source_encoder.config.scaling_factor
            return z.to(dist_util.dev()) * scaling_factor
        
    @torch.no_grad()
    def scene_encode(self, scene_ids, scene_attention_mask, scene_token_types):
        outputs = self.scene_encoder(
            input_ids = scene_ids,
            attention_mask = scene_attention_mask,
            token_type_ids = scene_token_types,
            return_dict=True
        )
        scene_states = outputs.last_hidden_state 
        return scene_states
    

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            
            if dist.get_rank() == 0:
                print(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        for name, param in self.model.named_parameters():
            param.data = param.data.to(dist_util.dev())
            if param.grad is not None:
                param.grad = param.grad.to(dist_util.dev())        

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            
            if dist.get_rank() == 0:
                print(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

            for group in self.opt.param_groups:
                for param in group['params']:
                    if param is not None:
                        param.data = param.data.to(dist_util.dev())
                        if param.grad is not None:
                            param.grad = param.grad.to(dist_util.dev())

        sche_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"sche{self.resume_step:06}.pt"
        )
        if bf.exists(sche_checkpoint):
            print(f"loading sche state from checkpoint: {sche_checkpoint}")
            state_dict = dist_util.load_state_dict(
                sche_checkpoint, map_location=dist_util.dev()
            )
            self.lr_sche.load_state_dict(state_dict)
  
    def run_loop(self):
        
        start_epoch = 0
        resume_done = True
        if self.resume_step > 0:
            start_epoch = self.resume_step // len(self.data)
            resume_done = False
            gap_steps = self.resume_step - (start_epoch * len(self.data))
            self.step = self.resume_step - 1
        for epoch in range(start_epoch, self.epochs):
            if self.resume_step > 0 and not resume_done:
                start_step = gap_steps
                self.lr_sche.step()
                self.cur_lr = self.opt.param_groups[0]['lr']
            else:
                start_step = 0
            self.recons_weight = self.init_recons_weight if epoch in self.recons_epoch else 0
            
            for step_in_epoch, batch in enumerate(self.data):
                if (not resume_done) and (step_in_epoch < start_step):
                    print('skip trained done step:{}/{}/{}'.format(step_in_epoch, start_step,epoch))
                    continue
                resume_done = True
                cond_keys = ['agn', 'agn_mask', 'image_densepose', 
                            'cloth', 'agn_cloth', 'cloth_generation',
                            'scene_ids', 'scene_attention_mask', 'scene_token_types'
                            ]
                cond = {key: batch[key].to(dist_util.dev()) for key in cond_keys}

                with torch.no_grad():
                    image = batch['image'].to(dist_util.dev())
                    target = self.source_encode(image)

                    cond['cloth_generation'] = self.source_encode(cond['cloth_generation'])
                    cond['agn_mask_raw'] = copy.deepcopy(cond['agn_mask'])
                      
                    cond['agn'] = self.source_encode(cond['agn'])
                    cond['agn_mask'] = self.source_encode(cond['agn_mask'])
                    cond['image_densepose'] = self.source_encode(cond['image_densepose'])
                    
                    cond['scene_states'] = self.scene_encode(cond['scene_ids'],
                                                    scene_attention_mask=cond['scene_attention_mask'],
                                                    scene_token_types=cond['scene_token_types'])
                    

                    with autocast(dtype=torch.bfloat16, device_type='cuda'):

                        cond['cloth'] = torch.stack(self.cloth_encoder(cond['cloth'])[1],
                                                    dim=1) 
                        cond['agn_cloth'] = self.cloth_encoder(cond['agn_cloth'].to(dist_util.dev()))[0]
                        
                    

                self.run_step(target, cond)

                if (self.step) % self.log_interval == 0 and dist.get_rank() == 0:
                    out = logger.dumpkvs()
                    if logger.get_rank_without_mpi_import()==0:
                        print(f"[Step {self.step}/{len(self.data)}|epoch:{epoch}/{self.epochs}] loss:{out.get('loss'):.5f},  mse:{out.get('mse'):.5f}, lr:{self.cur_lr:.7f}")
                if (self.step+1) % self.save_interval == 0:
                    self.opt.consolidate_state_dict()
                    self.save()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1

            # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, target, cond):
        self.forward_backward(target, cond)
        if (self.step + 1) % self.accumulate == 0:
            self._anneal_lr()
            took_step = self.mp_trainer.optimize(self.opt)
            self.mp_trainer.zero_grad()
            self._update_ema()
            
        self.log_step() 

    def forward_backward(self, target, cond):
        t, weights = self.schedule_sampler.sample(target.shape[0], dist_util.dev())

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            target,
            t,
            model_kwargs={"cond": cond,"enable_mask": False},
        )

        compute_losses_mask = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            target,
            t,
            model_kwargs={"cond": cond, "enable_mask": True},
        )
        
        losses = compute_losses()
        losses_mask = compute_losses_mask()
        losses['loss'] = losses['loss'] + self.recons_weight * losses['sub_recons']
        losses_mask['loss']  = losses_mask['loss'] + self.recons_weight * losses_mask['sub_recons']
        
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach() + losses_mask["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean() + (losses_mask["loss"] * weights).mean()
        loss = loss / self.accumulate
        if dist.get_rank() == 0:
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            log_loss_dict(
                self.diffusion, t, {'m_'+k: v * weights for k, v in losses_mask.items()}
            )
        self.mp_trainer.backward(loss)


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        self.lr_sche.step()
        self.cur_lr = self.opt.param_groups[0]['lr']

    def log_step(self):
        if dist.get_rank() == 0:
            logger.logkv("step", self.step)
            logger.logkv("samples", (self.step + 1) * self.global_batch)
            logger.logkv("lr", self.cur_lr)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                print(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + 1):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + 1):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            print(f"saving opt {(self.step + 1):06d}...")
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step + 1):06d}.pt"),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)

        if dist.get_rank() == 0:
            print(f"saving sche {(self.step + 1):06d}...")
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"sche{(self.step + 1):06d}.pt"),
                "wb",
            ) as f:
                torch.save(self.lr_sche.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


