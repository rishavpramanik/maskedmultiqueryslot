from data2 import TrainTransforms, DinoTransforms
from data_voc import VOCSegmentation
import torch
import torch.nn as nn

import os
from torchvision import transforms

from params import SlotAttentionParams
from model2 import PatchModel

from cluster import UnsupervisedMaskIoUMetric, AverageBestOverlapMetric, BboxCorLocMetric


params = SlotAttentionParams()



class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PatchModel(num_slots=params.num_slots,
                                num_iterations=3,
                                empty_cache=params.empty_cache,
                                slot_size=384)

    def forward(self, x, y):
        return self.model(x, y)


model_key = Model2()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val=43047446
addr = f'/home/rishavp/projects/def-mpederso/rishavp/object_unsup/lightning_logs/version_{val}/checkpoints/'+os.listdir(
    f'/home/rishavp/projects/def-mpederso/rishavp/object_unsup/lightning_logs/version_{val}/checkpoints/')[-1]
ckpt_key = torch.load(addr)

model_key.load_state_dict(ckpt_key['state_dict'])
model_key.eval()
model_key.to(device)


invTrans = transforms.Compose([transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
) 
])

val = VOCSegmentation(
    root=params.data_root,
    year='2012',
    image_set='val',
    transform=TrainTransforms().transforms,
    dino_transform=DinoTransforms().transforms,
    evo=True
)

val_dataloader = torch.utils.data.DataLoader(val, batch_size=1,
                                              shuffle=True,num_workers=1)

corloc = BboxCorLocMetric().to(
    'cuda' if torch.cuda.is_available() else 'cpu')
mbo = AverageBestOverlapMetric().to(
    'cuda' if torch.cuda.is_available() else 'cpu')
iou = UnsupervisedMaskIoUMetric(ignore_background=False).to(
    'cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    for img, dino, lal in val_dataloader:
        model_key.eval()
        model_key.training = False
        torch.cuda.empty_cache()
        img, dino, lal = img.to('cuda' if params.gpus > 0 else 'cpu'), dino.to(
            'cuda' if params.gpus > 0 else 'cpu'), lal.to('cuda' if params.gpus > 0 else 'cpu')
        reconstruction, mask, mask_as_image, target, slots = model_key(
            img, dino)
        msk0 = torch.argmax(mask_as_image, dim=1).detach()
        msk0 = torch.nn.functional.one_hot(
            msk0, num_classes=8).permute(0, 3, 1, 2)

        cor=corloc(msk0,lal)
        bo=mbo(msk0,lal)
        acc = iou(msk0, lal)

    print( f" {round(float(iou.compute().detach().cpu()),4)} corloc: {round(float(corloc.compute().detach().cpu()),4)} mbo: {round(float(mbo.compute().detach().cpu()),4)}")
