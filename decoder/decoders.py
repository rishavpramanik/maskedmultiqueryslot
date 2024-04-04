from decoder.utils import *
import torch.nn as nn
import torch
from typing import *
import functools
from decoder.utils import build_mlp

    
class PatchDecoder(nn.Module):
    """Decoder that takes object representations and reconstructs patches.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        decoder: Function that returns backbone to use for decoding. Function takes input and output
            dimensions and should return module that takes inputs of shape (B * K), P, N, and produce
            outputs of shape (B * K), P, M, where K is the number of objects, N is the number of
            input dimensions and M the number of output dimensions.
        decoder_input_dim: Input dimension to decoder backbone. If specified, a linear
            transformation from object to decoder dimension is added. If not specified, the object
            dimension is used and no linear transform is added.
    """

    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        num_patches: int,
        decoder: Callable[[int, int], nn.Module] = functools.partial(build_mlp,features=[1024,1024,1024]),
        decoder_input_dim: Optional[int] = None,
        upsample_target: Optional[float] = None,
        resize_mode: str = "bilinear",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.upsample_target = upsample_target
        self.resize_mode = resize_mode

        if decoder_input_dim is not None:
            self.inp_transform = nn.Linear(object_dim, decoder_input_dim, bias=True)
            nn.init.xavier_uniform_(self.inp_transform.weight)
            nn.init.zeros_(self.inp_transform.bias)
        else:
            self.inp_transform = None
            decoder_input_dim = object_dim

        self.decoder = decoder(decoder_input_dim, output_dim + 1)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, decoder_input_dim) * 0.02)

    def forward(
        self,
        object_features: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ):
        assert object_features.dim() >= 3  # Image or video data.
        if self.upsample_target is not None and target is not None:
            target = (
                resize_patches_to_image(
                    target.detach().transpose(-2, -1),
                    scale_factor=self.upsample_target,
                    resize_mode=self.resize_mode,
                )
                .flatten(-2, -1)
                .transpose(-2, -1)
            )

        initial_shape = object_features.shape[:-1]
        object_features = object_features.flatten(0, -2)

        if self.inp_transform is not None:
            object_features = self.inp_transform(object_features)

        object_features = object_features.unsqueeze(1).expand(-1, self.num_patches, -1)

        # Simple learned additive embedding as in ViT
        object_features = object_features + self.pos_embed

        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        masks = alpha.squeeze(-1)

        if image is not None:
            masks_as_image = resize_patches_to_image(
                masks, size=image.shape[-1], resize_mode="bilinear"
            )
        else:
            masks_as_image = None

        return PatchReconstructionOutput(
            reconstruction=reconstruction,
            masks=alpha.squeeze(-1),
            masks_as_image=masks_as_image,
            target=target if target is not None else None,
        )