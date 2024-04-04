import torch
from torch import nn

import datas.vision_transformer as vit

class DinoFeatExtractor(nn.Module):
    def __init__(self,name,strict_load=True,verbose=True,patch_size=8,version=1):
        self.force_reload=strict_load
        self.verbose=verbose
        self.patch_size=patch_size
        self.arch=name
        self.version=version

    def _getaddr(self):
        if self.arch == "vit_small" and self.patch_size == 16 and self.version==1:
            path = "/home/rishavp/projects/def-mpederso/rishavp/pretrained/dino/dino_deitsmall16_pretrain.pth"
        elif self.arch == "vit_small" and self.patch_size == 8 and self.version==1:
            path = "/home/rishavp/projects/def-mpederso/rishavp/pretrained/dino/dino_deitsmall8_pretrain.pth"
        elif self.arch == "vit_base" and self.patch_size == 16 and self.version==1:
            path = "/home/rishavp/projects/def-mpederso/rishavp/pretrained/dino/dino_vitbase16_pretrain.pth"
        elif self.arch == "vit_base" and self.patch_size == 8 and self.version==1:
            path = "/home/rishavp/projects/def-mpederso/rishavp/pretrained/dino/dino_vitbase8_pretrain.pth"
        elif self.arch == "vit_small" and self.patch_size == 14 and self.version==2:
            path = "/home/rishavp/projects/def-mpederso/rishavp/pretrained/dino_v2/dinov2_vits14_pretrain.pth"
        elif self.arch == "vit_base" and self.patch_size == 14 and self.version==2:
            path = "/home/rishavp/projects/def-mpederso/rishavp/pretrained/dino_v2/dinov2_vitb14_pretrain.pth"
        elif self.arch == "vit_large" and self.patch_size == 14 and self.version==2:
            path = "/home/rishavp/projects/def-mpederso/rishavp/pretrained/dino_v2/dinov2_vitl14_pretrain.pth"
        elif self.arch == "vit_giant" and self.patch_size == 14 and self.version==2:
            path = "/home/rishavp/projects/def-mpederso/rishavp/pretrained/dino_v2/dinov2_vitg14_pretrain.pth"
        else:
            raise ModuleNotFoundError(f'{self.arch} not found in folder')
        return path
    

    def load_dino_backbone(self):
        model=vit.__dict__[self.arch](patch_size=self.patch_size,num_classes=0)
        model.load_state_dict(torch.load(self._getaddr()),strict=self.force_reload)
        model.eval()
        return model
        
        