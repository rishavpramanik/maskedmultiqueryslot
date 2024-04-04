import pytorch_lightning as pl
from torch import optim

from model2 import PatchModel
from params import SlotAttentionParams
from utils import Tensor





class MultiqueryslotMethod(pl.LightningModule):
    def __init__(self, model: PatchModel, datamodule, params: SlotAttentionParams):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params

    def forward(self, input: Tensor, dino: Tensor, **kwargs) -> Tensor:
        return self.model(input,dino, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        img,dino=batch
        train_loss = self.model.loss_function(img,dino)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True,prog_bar=True)
        return train_loss


    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        img,dino=batch
        val_loss = self.model.loss_function(img,dino)
        self.val_loss=val_loss
        return val_loss

    def on_validation_epoch_end(self):
        # avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_val_loss": self.val_loss,
        }
        self.log_dict(self.val_loss, sync_dist=True)
        print("; ".join([f"{k}: {v}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.datamodule)

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct*total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )
