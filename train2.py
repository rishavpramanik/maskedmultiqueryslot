from typing import Optional
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import torch


from data_voc import VOCSegmentation
from method2 import MultiqueryslotMethod
from model2 import PatchModel
from data2 import TrainTransforms,DinoTransforms
from params import SlotAttentionParams

    
def main(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    assert params.num_slots > 1, "Must have at least 2 slots."


    train = VOCSegmentation(
        root=params.data_root,
        year='2012',
        image_set='train',
        transform=TrainTransforms().transforms,
        dino_transform=DinoTransforms().transforms
    )
    val = VOCSegmentation(
        root=params.data_root,
        year='2012',
        image_set='val',
        transform=TrainTransforms().transforms,
        dino_transform=DinoTransforms().transforms
    )

    masking=params.masking_ratio # type: ignore
    train_dataloader=torch.utils.data.DataLoader(train, batch_size=params.batch_size,shuffle=True, num_workers=params.num_workers)
    val_dataloader=torch.utils.data.DataLoader(val, batch_size=params.val_batch_size, shuffle=False, num_workers=params.num_workers)
    model = PatchModel(
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
        slot_size=384,
        masking=masking
    )

    method = MultiqueryslotMethod(model=model, datamodule=train, params=params)

    trainer = Trainer(
        accelerator="gpu" if params.gpus > 0 else None,
        devices=params.gpus,
        strategy='ddp_find_unused_parameters_true',
        num_sanity_val_steps=params.num_sanity_val_steps,
        max_epochs=params.max_epochs,
        log_every_n_steps=10,
        callbacks=[LearningRateMonitor("step")] if params.is_logger_enabled else [],
    )
    trainer.fit(method,train_dataloader,val_dataloader)
   
    model.eval()
    model=model.to('cuda' if params.gpus>0 else 'cpu')
    val = VOCSegmentation(
        root=params.data_root,
        year='2012',
        image_set='trainval',
        transform=TrainTransforms().transforms,
        dino_transform=DinoTransforms().transforms,
        evo=True
    )


if __name__ == "__main__":
    main()
