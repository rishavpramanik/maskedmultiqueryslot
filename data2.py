from torchvision.transforms import transforms


class TrainTransforms(object):
    def __init__(self):
        self.transforms = transforms.Compose(
            [      #Noaugmentations please
                # transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize((320,320)),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)

class DinoTransforms(object):
    def __init__(self):
        self.transforms = transforms.Compose(
            [      #Noaugmentations please
                transforms.ToTensor(),
                # rescale between -1 and 1
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.CenterCrop((320,320)),
            ]
        )
class DinoValTransforms(object):
    def __init__(self):
        self.transforms = transforms.Compose(
            [      #Noaugmentations please
                transforms.ToTensor(),
                # rescale between -1 and 1
                transforms.Resize((320,320)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.CenterCrop((320,320)),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)