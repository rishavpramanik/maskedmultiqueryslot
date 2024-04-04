import numpy as np
import torch
from torch import nn
from model2 import PatchModel
from params import SlotAttentionParams
import os

from data_voc import VOCSegmentation
from data2 import  DinoValTransforms, TrainTransforms
from matplotlib import pyplot as plt

from torchvision import transforms

create_pascal_label_colormap=np.array([
      [255, 0, 0],
      [0, 128, 0],
      [0, 0, 128],
      [128, 255, 0],
      [0, 0, 255],
      [128, 0, 255],
      [0, 128, 255],
      [255, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [0, 255, 204]])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img=torch.randn(4,3,320,320).to(device)
dino=torch.randn(4,3,320,320).to(device)
params = SlotAttentionParams()
val = VOCSegmentation(
        root=params.data_root,
        # segments_dir=params.data_root+'/VOCdevkit/VOC2007/SegmentationClass',
        year='2012',
        image_set='val',
        transform=TrainTransforms().transforms,
        dino_transform=DinoValTransforms().transforms,
        evo=True
    )
val_dataloader = torch.utils.data.DataLoader(
        val, batch_size=1, shuffle=True,
        num_workers=params.num_workers)


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PatchModel(num_slots=params.num_slots,
                                num_iterations=3,
                                empty_cache=params.empty_cache,
                                slot_size=384,
                                masking=params.masking_ratio)

    def forward(self, x, y):
        return self.model(x, y)


model_best = Model2()
device = torch.device(device)
val = 42886497
addr = f'/home/rishavp/projects/def-mpederso/rishavp/object_unsup/lightning_logs/version_{val}/checkpoints/'+os.listdir(
    f'/home/rishavp/projects/def-mpederso/rishavp/object_unsup/lightning_logs/version_{val}/checkpoints/')[-1]
ckpt_key = torch.load(addr)
model_best.load_state_dict(ckpt_key['state_dict'])
model_best.eval()
model_best.to(device)


def visualize_seg_mask(image: np.ndarray, mask: np.ndarray, filename):
    "visualize the masks"
    #    print(image.shape)
    color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    palette = np.array(create_pascal_label_colormap)
    for label, color in enumerate(palette):
        color_seg[mask == label, :] = color
    color_seg = color_seg[..., ::-1]  # convert to BGR


    tmg = image*0.98 + color_seg * 0.02  # plot the image with the segmentation map

    plt.axis("off")
    plt.imshow(tmg)
    plt.savefig(filename,bbox_inches='tight')

invTrans = transforms.Compose([transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )  # ,
        # transforms.Resize((320,320))
    ])

#If you want a custom image feel free to pass it accordingly

img, dino,label=next(iter(val_dataloader))
img,dino,label=img.to(device),dino.to(device),label.to(device)

invTrans = transforms.Compose([transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )  # ,
        # transforms.Resize((320,320))
    ])
reconstruction, mask, mask_as_image, target, slots = model_best(img, dino)

mskb = torch.argmax(mask_as_image[0], dim=0).detach()
img = invTrans(img)
image = img[0].permute(1, 2, 0).cpu().numpy()

plt.axis("off")
plt.imshow(image)
plt.savefig('/home/rishavp/projects/def-mpederso/rishavp/raw_img.png',bbox_inches='tight')

visualize_seg_mask(image,mskb.cpu().numpy(),'/home/rishavp/projects/def-mpederso/rishavp/mask.png')