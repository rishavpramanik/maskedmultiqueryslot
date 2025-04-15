import random
import torch
from torch import nn
from torch.nn import functional as F

from utils import assert_shape

from datas.extract_utils import get_model

from decoder.decoders import PatchDecoder
from attention import MultiQuerySlot,SlotAttention


class PatchModel(nn.Module):
    def __init__(
        self,
        num_slots: int,
        num_iterations,
        in_channels: int = 3,
        slot_size: int = 384,
        empty_cache: bool=False,
        random_masking: bool=False,
        slot_attention: bool=False,
        masking: float=0.1):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.masking=1-masking
        self.random_masking=random_masking
        if self.masking==1:
            self.masking=False
        
        # assert masking>-0.0000001 and masking<1

        self.encoder = None
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(slot_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.slot_size))
        self.slot_size=slot_size
        self.decoder=PatchDecoder(self.slot_size, self.slot_size, 400)
        if slot_attention:
            self.slot_attention=SlotAttention(
            in_features=self.slot_size,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=512)
        else:
            self.slot_attention=MultiQuerySlot(
            in_features=self.slot_size,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=512)
        

        self.feat_out = {}
        self.dino, self.dino_transform, self.patch_size, self.num_heads = self.get_dino()

    def forward(self, x, dino):

        if self.empty_cache:
            torch.cuda.empty_cache()


        dino = self.get_dino_feat(
            dino, self.dino, self.patch_size, self.num_heads)
        batch_size, num_channels, height, width = x.shape
        encoder_out = torch.clone(dino['k'])
        if self.masking and self.training:
            if self.random_masking:
                args=torch.tensor([random.sample(range(encoder_out.shape[1]),k=int(len(encoder_out[0])*(1-self.masking))) for _ in range(batch_size)],device=encoder_out.device)
                mask = torch.zeros_like(encoder_out,device=encoder_out.device)
                batch_indices = torch.arange(encoder_out.shape[0]).unsqueeze(1).unsqueeze(2)
                encoder_out[batch_indices, args.unsqueeze(2)] = mask[batch_indices, args.unsqueeze(2)]
            else:
                mean=torch.mean(encoder_out,dim=2)
                args=torch.argsort(mean,dim=1,descending=False)
                args=args[:,int(encoder_out.shape[1]*self.masking):]
                mask = torch.zeros_like(encoder_out,device=encoder_out.device)
                batch_indices = torch.arange(encoder_out.shape[0]).unsqueeze(1).unsqueeze(2)
                encoder_out[batch_indices, args.unsqueeze(2)] = mask[batch_indices, args.unsqueeze(2)]
                    
        encoder_out = self.encoder_out_layer(encoder_out)


        slots = self.slot_attention(encoder_out)
        assert_shape(slots.size(), (batch_size,
                                    self.num_slots, self.slot_size))

        batch_size, num_slots, slot_size = slots.shape

        out = self.decoder(slots, dino['k'], x)

        return out.reconstruction, out.masks, out.masks_as_image, out.target, slots

    def loss_function(self, input, dino):
        recon_combined, recons, masks, target, slots = self.forward(
            input, dino)
        loss = F.mse_loss(recon_combined, target)
        return {
            "loss": loss,
        }

    def hook_fn_forward_qkv(self, module, input, output):
        self.feat_out["qkv"] = output

    def get_dino(self):
        model, val_transform, patch_size, num_heads = get_model(
            "dino_vits16", "vit_small",patch_size=16)

        model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
            self.hook_fn_forward_qkv)
        return model, None, patch_size, num_heads

    def get_dino_feat(self, img, model, patch_size, num_heads):
        P = patch_size
        B, C, H, W = img.shape
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1
        img = img[:, :, :H_pad, :W_pad]
        model.get_intermediate_layers(img)[0].squeeze(0)
        output_dict = {}
        output_qkv = self.feat_out["qkv"].reshape(
            B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
        output_dict['q'] = output_qkv[0].transpose(
            1, 2).reshape(B, T, -1)[:, 1:, :]
        output_dict['k'] = output_qkv[1].transpose(
            1, 2).reshape(B, T, -1)[:, 1:, :]
        output_dict['v'] = output_qkv[2].transpose(
            1, 2).reshape(B, T, -1)[:, 1:, :]
        return output_dict
