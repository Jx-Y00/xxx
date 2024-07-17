import math
import os
join = os.path.join
from typing import Type, List

import torch
import torch.nn as nn

from segment_anything.modeling import Sam
from segment_anything.modeling.common import MLPBlock


class AdaptMLPBlock(nn.Module):
    def __init__(
        self,
        mlp: MLPBlock,
        adapter_bn: int = 64,
        adapter_act: Type[nn.Module] = nn.GELU,
        adapter_dropout: float = 0.1,
        adapter_scalar: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        
        embedding_dim = mlp.embedding_dim
        self.adapter_down = nn.Linear(embedding_dim, adapter_bn)
        self.adapter_up = nn.Linear(adapter_bn, embedding_dim)
        self.adapter_act = adapter_act()
        self.adapter_dropout = adapter_dropout
        if adapter_scalar is None:
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = adapter_scalar
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x: torch.Tensor,
                organ: List[torch.Tensor]) -> torch.Tensor:
        adpt = self.adapter_up(nn.functional.dropout(
            self.adapter_act(self.adapter_down(x)), 
            p=self.adapter_dropout, training=self.training))
        return self.mlp(x) + adpt * self.adapter_scale

class SAM_Adapter(nn.Module):
    """Applies Adapter to a SAM's image encoder.

    Args:
        sam: segment anything model, see 'segment_anything' dir
        bottleneck: bottleneck dimension of adapter
        pos: which layer to apply adapter
    """

    def __init__(self, sam: Sam, bottleneck: int, pos: list = None):
        super(SAM_Adapter, self).__init__()

        assert bottleneck > 0
        
        # assign Adapter layer position (all layers by default)
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(sam.image_encoder.blocks)))
        
        # freeze SAM image and prompt encoder
        # if sam.image_encoder.img_size != 1024:
        #     for n, p in sam.image_encoder.named_parameters():
        #         if 'pos' not in n:
        #             p.requires_grad = False
        # else:
        #     for param in sam.image_encoder.parameters():
        #         param.requires_grad = False
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False
        
        # apply Adapter to SAM image encoder
        for idx, blk in enumerate(sam.image_encoder.blocks):
            if idx not in self.pos:
                continue
            
            # create adapter layers
            blk.mlp = AdaptMLPBlock(
                blk.mlp,
                adapter_bn=bottleneck,
            )
        
        self.sam = sam

    def save_parameters(self) -> dict:
        r"""save both adapter and mask decoder parameters.
        """
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        
        # save image encoder parameters
        image_encoder_tensors = {}
        # for key, value in state_dict.items():
        #     if 'pos' in key:
        #         image_encoder_tensors[key] = value
        
        # save prompt encoder parameters
        prompt_encoder_tensors = {}
        # for key, value in state_dict.items():
        #     if 'prompt_encoder' in key:
        #         prompt_encoder_tensors[key] = value
        
        # save mask decoder parameters
        mask_decoder_tensors = {}
        for key, value in state_dict.items():
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
        
        # save adapter parameters
        adapter_tensors = {}
        for key, value in state_dict.items():
            if 'adapter' in key:
                adapter_tensors[key] = value

        merged_dict = {**adapter_tensors, **image_encoder_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        return merged_dict

    def load_parameters(self, state_dict) -> None:
        r"""load both adapter and mask decoder parameters.
        """
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()
        
        # load image encoder parameters
        # image_encoder_keys = [k for k in sam_keys if 'pos' in k]
        # image_encoder_values = [state_dict[k] for k in image_encoder_keys]
        # image_encoder_new_state_dict = {k: v for k, v in zip(image_encoder_keys, image_encoder_values)}
        # sam_dict.update(image_encoder_new_state_dict)
        
        # load prompt encoder parameters
        # prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        # prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        # prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        # sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder parameters
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)

        # load adapter parameters
        adapter_keys = [k for k in sam_keys if 'adapter' in k]
        adapter_values = [state_dict[k] for k in adapter_keys]
        adapter_new_state_dict = {k: v for k, v in zip(adapter_keys, adapter_values)}
        sam_dict.update(adapter_new_state_dict)
        
        self.sam.load_state_dict(sam_dict)

    def forward(self, data):
        img, box = data['img'], data['box']
        
        # prompt encoder
        if len(box.shape) == 2:
            box = box[:, None, :]  # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=box,
            masks=None,
        )
        
        # Adapter image encoder
        input_image = self.sam.preprocess(img) # (B, 3, 1024, 1024)
        image_embedding = self.sam.image_encoder(input_image) # (B, 256, 64, 64)
        
        # predicted masks
        mask_predictions, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # (B, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=True,
          )
        
        return mask_predictions
