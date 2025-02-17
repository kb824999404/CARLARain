"""
wild mixture of
https://github.com/CompVis/stable-diffusion
https://github.com/DifanLiu/ASSET
-- merci
"""
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm.util import default,instantiate_from_config

from ldm.models.diffusion.ddpm import DDPM,LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler

from asset.models.cond_transformer import Net2NetTransformer

import ipdb


class HRIGTransformer(Net2NetTransformer):
    def __init__(self,
                 mask_key,
                 monitor="val/loss_transformer",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_key = mask_key
        self.scale_type = False
        self.monitor = monitor
    def shared_step(self, batch, mask_key = "mask"):
        mask_tensor = self.get_input(mask_key, batch)  # bs, 1, 256, 256, on cuda 
        x, c = self.get_xc(batch)
        xh = x.shape[2]
        xw = x.shape[3]
        resized_mask_tensor = F.interpolate(mask_tensor, size=(xh // (2 ** self.num_downsampling), xw // (2 ** self.num_downsampling)))
        # one step to produce the logits
        # z_code: bs, 128, 16, 16
        # z_indices: bs, 256
        z_code, z_indices = self.encode_to_z(x, mask_tensor=mask_tensor)  # bs, 256
        c_code, c_indices = self.encode_to_c(c)  # bs, 256
        
        resized_mask = resized_mask_tensor[:, 0, :, :].cpu().numpy()  # bs, 16, 16
        single_T = z_indices.shape[1] # 256
        
        # mask input image tokens
        a_indices = z_indices.clone()
        for bid in range(resized_mask.shape[0]):
            flatten_np = resized_mask[bid].flatten()
            indices_unknown = np.nonzero(flatten_np)[0]  # positions in z_indices
            a_indices[bid, indices_unknown] = self.mask_token_id
        
        # decoder_input_ids
        decoder_input_ids = torch.arange(single_T + 1).unsqueeze(0).expand(resized_mask.shape[0], -1).to(self.device)  # B, 257
        decoder_input_ids[:, 1:] = z_indices.clone()
        decoder_input_ids[:, 0] = self.start_token_id  # [start]
        temp = self.transformer(input_ids=a_indices, cond_ids=c_indices, decoder_input_ids=decoder_input_ids)
        logits = temp[0]
        logits = logits[:, :-1, :]  # the last one is a redundant one
        # compute losses
        logits_list = []
        target_list = []
        fake_z_indices = z_indices.clone()
        for bid in range(resized_mask.shape[0]):
            flatten_np = resized_mask[bid].flatten()
            indices_unknown = np.nonzero(flatten_np)[0]
            logits_list.append(logits[bid, indices_unknown, :])
            target_list.append(z_indices[bid, indices_unknown])
            # get predict result
            fake_z_indices[bid, indices_unknown] = torch.argmax(logits[bid, indices_unknown, :],dim=1)
        logits = torch.cat(logits_list, 0)
        target = torch.cat(target_list, 0)     
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        # z_code: [1, 128, 16, 16] bchw
        # z_indices: [1, 256]   b h*w
        fake_z_indices = self.permuter(fake_z_indices, reverse=True)
        zshape = z_code.shape
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        fake_z_code = self.first_stage_model.quantize.get_codebook_entry(
        fake_z_indices.reshape(-1), shape=bhwc)
        
        prefix = 'train' if self.training else 'val'
        loss_dict = dict()
        loss_dict.update({f'{prefix}/loss_transformer': loss})

        return loss,loss_dict,fake_z_code
    
    @torch.no_grad()
    def get_output(self, batch, mask_key = "mask"):
        _, c = self.get_xc(batch)
        x = c
        mask_tensor = self.get_input(mask_key, batch)  # bs, 1, 256, 256, on cuda 
        x = x.to(self.device)
        c = c.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        xh = x.shape[2]
        xw = x.shape[3]
        resized_mask_tensor = F.interpolate(mask_tensor, size=(xh // (2 ** self.num_downsampling), xw // (2 ** self.num_downsampling)))
        # one step to produce the logits
        # z_code: bs, 128, 16, 16
        # z_indices: bs, 256
        z_code, z_indices = self.encode_to_z(x, mask_tensor=mask_tensor)  # bs, 256
        c_code, c_indices = self.encode_to_c(c)  # bs, 256
        
        latent_mask = resized_mask_tensor.squeeze().cpu().numpy()  # [0, 1]
        fake_z_indices_batch = []
        for bid in range(x.shape[0]):
            fake_z_indices = self.autoregressive_sample_fast256(z_indices[bid:bid+1], c_indices[bid:bid+1],
                                                        c_code.shape[2], c_code.shape[3],
                                                        latent_mask[bid:bid+1], batch_size=1)
            fake_z_indices_batch.append(fake_z_indices)
        fake_z_indices_batch = torch.cat(fake_z_indices_batch,dim=0)
        fake_z_indices_batch = self.permuter(fake_z_indices_batch, reverse=True)

        zshape = z_code.shape
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        fake_z_code = self.first_stage_model.quantize.get_codebook_entry(
        fake_z_indices_batch.reshape(-1), shape=bhwc)
        fake_x = self.first_stage_model.decode(fake_z_code)
        
        return fake_x,fake_z_code,mask_tensor

    def training_step(self, batch, batch_idx):
        loss,loss_dict,_ = self.shared_step(batch, mask_key=self.mask_key)  
        
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _,loss_dict,_ = self.shared_step(batch, mask_key=self.mask_key)  

        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)    

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()

        x, c = self.get_xc(batch)
        log["inputs"] = x
        log["condition"] = c

        pred_x,_,mask_tensor = self.get_output(batch,mask_key=self.mask_key)
        log["prediction"] = pred_x
        log["mask"] = mask_tensor

        return log

class Upsample(torch.nn.Module):
    def __init__(self, scale_factor, in_channels=3 , channels = []):
        super().__init__()
        self.scale_factor = scale_factor
        convBlock = torch.nn.ModuleList()

        in_ch = in_channels
        channels.append(in_channels)
        for channel in channels:
            out_ch = channel
            convBlock.append(
                torch.nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
            )
            convBlock.append(torch.nn.ReLU())
            in_ch = channel
        self.convBlock = convBlock

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor)
        for m in self.convBlock:
            x = m(x)
        return x
    
class HRIGScaleTransformer(HRIGTransformer):
    def __init__(self,
                 upsampler_config,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsampler = self.config_upsampler(upsampler_config)
        self.scale_type = True
        

    def shared_step(self, batch, mask_key = "mask_1"):
        loss_transformer,loss_dict,fake_z_code = super().shared_step(batch, mask_key)
        fake_z_code_high = self.upsampler(fake_z_code)

        x, _ = self.get_xc(batch)
        x_high = F.interpolate(x, scale_factor=self.upsampler.scale_factor)
        z_code_high, _ = self.encode_to_z(x_high)
        loss_upsampler = self.get_loss_upsampler(fake_z_code_high,z_code_high)

        loss = loss_transformer + loss_upsampler

        prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{prefix}/loss_upsampler': loss_upsampler})

        return loss,loss_dict,fake_z_code_high

    def get_loss_upsampler(self, pred, target, mean=True):
        if self.upsampler_loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.upsampler_loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def config_upsampler(self,config):
        self.upsampler_loss_type = config.loss_type
        model = Upsample(config.scale_factor,config.in_channels,config.channels)
        return model

    def training_step(self, batch, batch_idx):
        loss,loss_dict,_ = self.shared_step(batch, mask_key=self.mask_key)  
        
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _,loss_dict,_ = self.shared_step(batch, mask_key=self.mask_key)  

        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def get_output(self, batch, mask_key = "mask_1"):
        fake_x,fake_z_code,mask_tensor = super().get_output(batch,mask_key)
        fake_z_code_high = self.upsampler(fake_z_code)
        fake_x_high = self.first_stage_model.decode(fake_z_code_high)
        return fake_x,fake_x_high,fake_z_code_high,mask_tensor

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()

        x, c = self.get_xc(batch)
        log["inputs"] = x
        log["condition"] = c

        pred_x,pred_x_high,_,mask_tensor = self.get_output(batch,mask_key=self.mask_key)
        log["prediction_low"] = pred_x
        log["prediction_high"] = pred_x_high
        log["mask"] = mask_tensor

        return log



# High-resolution Rainy Image Generation: ASSET+LDM
class HRIGDiffusion(LatentDiffusion):
    def __init__(self, 
                 first_stage_config,
                 cond_stage_config,
                 asset_config,
                 num_timesteps_cond=None,
                 first_stage_key="image_gt_0",
                 cond_stage_key="masked_image_0",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                # ASSET
                 asset_learning_rate=3.24e-06,
                 transformer_trainable=False,
                 *args, **kwargs):
        
        # --------LatentDiffusion--------
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        self.conditioning_key = conditioning_key
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        DDPM.__init__(self,conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        # --------ASSET--------
        self.asset = instantiate_from_config(asset_config)
        self.asset.learning_rate = asset_learning_rate
        self.transformer_trainable = transformer_trainable
        

    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss_asset,loss_dict_asset,cond_asset= self.asset.shared_step(batch, mask_key=self.asset.mask_key)  
        if optimizer_idx == 0:
            z, c = self.get_input(batch, self.first_stage_key)
            if self.conditioning_key == "concat":
                cond = torch.cat([c, cond_asset], dim=1)
            elif self.conditioning_key == "hybrid":
                cond = {
                    "c_concat": [c],
                    "c_crossattn": [cond_asset],
                }
            loss_ldm, loss_dict = self(z,cond)
            loss = loss_ldm
        
        if optimizer_idx == 1:
            loss = loss_asset
            loss_dict = loss_dict_asset

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _,loss_dict_asset,cond_asset = self.asset.shared_step(batch, mask_key=self.asset.mask_key)  
        z, c = self.get_input(batch, self.first_stage_key)
        if self.conditioning_key == "concat":
            cond = torch.cat([c, cond_asset], dim=1)
        elif self.conditioning_key == "hybrid":
            cond = {
                "c_concat": [c],
                "c_crossattn": [cond_asset],
            }
        _, loss_dict_no_ema = self(z,cond)
        loss_dict_no_ema.update(loss_dict_asset)

        with self.ema_scope():
            _, loss_dict_ema = self(z,cond)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizers = []
        opt_ldm = super().configure_optimizers()
        optimizers.append(opt_ldm)
        if self.transformer_trainable:
            opt_asset = self.asset.configure_optimizers()
            optimizers.append(opt_asset)
        
        return optimizers, []



    @torch.no_grad()
    def get_samples(self, batch, N=8, ddim_steps=20, ddim_eta=1., return_mediate=False):
        if self.asset.scale_type:
            cond_x,cond_x_high,cond_asset,mask_tensor = self.asset.get_output(batch,mask_key=self.asset.mask_key)
        else:
            cond_x,cond_asset,mask_tensor = self.asset.get_output(batch,mask_key=self.asset.mask_key)
        _, c = self.get_input(batch, self.first_stage_key)
        if self.conditioning_key == "concat":
            cond = torch.cat([c, cond_asset], dim=1)
        elif self.conditioning_key == "hybrid":
            cond = {
                "c_concat": [c],
                "c_crossattn": [cond_asset],
            }
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, _ =ddim_sampler.sample(ddim_steps,N,
                                                shape,cond,
                                                eta=ddim_eta,
                                                verbose=False)
        x_samples = self.decode_first_stage(samples)
        if return_mediate:
            if self.asset.scale_type:
                return x_samples,cond_x,cond_x_high,mask_tensor
            else:
                return x_samples,cond_x,mask_tensor
        else:
            return x_samples

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=20, ddim_eta=1., return_keys=None,
                    **kwargs):

        log = dict()
        _, _, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["condition"] = xc
        if sample:
            if self.asset.scale_type:
                x_samples,cond_x,cond_x_high,mask_tensor = self.get_samples(batch, N, ddim_steps, ddim_eta, True )
                log["condition_asset_low"] = cond_x
                log["condition_asset_high"] = cond_x_high
            else:
                x_samples,cond_x,mask_tensor = self.get_samples(batch, N, ddim_steps, ddim_eta, True )
                log["condition_asset"] = cond_x
                
            log["samples"] = x_samples
            log["mask"] = mask_tensor

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log


class GuidingDiffusion(LatentDiffusion):
    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss, loss_dict, output = self(x, c)
        return loss, loss_dict, output

    def training_step(self, batch, batch_idx):
        loss, loss_dict, _ = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema, _ = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema, _ = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_gdm_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gdm_gamma': loss.mean()})
            loss_dict.update({'logvar_gdm': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_gdm_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss_gdm': loss})

        return loss, loss_dict, model_output
        
    
    def get_output(self, batch, N=8, ddim_steps=20, ddim_eta=1):
        _, c = self.get_input(batch, self.first_stage_key)
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, _ =ddim_sampler.sample(ddim_steps,N,
                                                shape,c,verbose=False)
        x_samples = self.decode_first_stage(samples)
        return x_samples, samples

# High-resolution Rainy Image Generation: Guiding Diffusion+LDM
class HRIGDiffusionGDM(LatentDiffusion):
    def __init__(self, 
                 first_stage_config,
                 cond_stage_config,
                 gdm_config,
                 num_timesteps_cond=None,
                 first_stage_key="image_gt_0",
                 cond_stage_key="masked_image_0",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                # Guiding Diffusion Model
                 gdm_learning_rate=3.24e-06,
                 gdm_trainable=False,
                 *args, **kwargs):
        
        # --------LatentDiffusion--------
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        self.conditioning_key = conditioning_key
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        DDPM.__init__(self,conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        # --------Guiding Diffusion Model--------
        self.gdm = instantiate_from_config(gdm_config)
        self.gdm.learning_rate = gdm_learning_rate
        self.gdm_trainable = gdm_trainable
        

    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss_gdm,loss_dict_gdm,cond_gdm= self.gdm.shared_step(batch)  
        if optimizer_idx == 0:
            z, c = self.get_input(batch, self.first_stage_key)
            if self.conditioning_key == "concat":
                cond = torch.cat([c, cond_gdm], dim=1)
            elif self.conditioning_key == "hybrid":
                cond = {
                    "c_concat": [c],
                    "c_crossattn": [cond_gdm],
                }
            loss_ldm, loss_dict = self(z,cond)
            loss = loss_ldm
        
        if optimizer_idx == 1:
            loss = loss_gdm
            loss_dict = loss_dict_gdm

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _,loss_dict_gdm,cond_gdm = self.gdm.shared_step(batch)  
        z, c = self.get_input(batch, self.first_stage_key)
        if self.conditioning_key == "concat":
            cond = torch.cat([c, cond_gdm], dim=1)
        elif self.conditioning_key == "hybrid":
            cond = {
                "c_concat": [c],
                "c_crossattn": [cond_gdm],
            }
        _, loss_dict_no_ema = self(z,cond)
        loss_dict_no_ema.update(loss_dict_gdm)

        with self.ema_scope():
            _, loss_dict_ema = self(z,cond)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizers = []
        opt_ldm = super().configure_optimizers()
        optimizers.append(opt_ldm)
        if self.gdm_trainable:
            opt_gdm = self.gdm.configure_optimizers()
            optimizers.append(opt_gdm)
        
        return optimizers, []



    @torch.no_grad()
    def get_samples(self, batch, N=8, ddim_steps=20, ddim_eta=1., return_mediate=False):
        cond_x,cond_gdm = self.gdm.get_output(batch, N, ddim_steps, ddim_eta)
        _, c = self.get_input(batch, self.first_stage_key)
        if self.conditioning_key == "concat":
            cond = torch.cat([c, cond_gdm], dim=1)
        elif self.conditioning_key == "hybrid":
            cond = {
                "c_concat": [c],
                "c_crossattn": [cond_gdm],
            }
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, _ =ddim_sampler.sample(ddim_steps,N,
                                                shape,cond,
                                                eta=ddim_eta,
                                                verbose=False)
        x_samples = self.decode_first_stage(samples)
        if return_mediate:
            return x_samples,cond_x
        else:
            return x_samples

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=20, ddim_eta=1., return_keys=None,
                    **kwargs):

        log = dict()
        _, _, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["condition"] = xc
        if sample:
            x_samples,cond_x = self.get_samples(batch, N, ddim_steps, ddim_eta, True )
            log["condition_gdm"] = cond_x
            log["samples"] = x_samples

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
