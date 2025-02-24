import math
import os
join = os.path.join

import torch
import torch.nn as nn

from segment_anything.modeling import Sam


class qkv_LoRA(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            q_lora_a: nn.Module,
            q_lora_b: nn.Module,
            k_lora_a: nn.Module,
            k_lora_b: nn.Module,
            v_lora_a: nn.Module,
            v_lora_b: nn.Module,
            mode: str = "qv",
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.mode = mode
        self.q_lora_a = q_lora_a
        self.q_lora_b = q_lora_b
        self.k_lora_a = k_lora_a
        self.k_lora_b = k_lora_b
        self.v_lora_a = v_lora_a
        self.v_lora_b = v_lora_b

    def forward(self, x):
        qkv = self.qkv(x)
        
        if 'q' in self.mode:
            q_lora = self.q_lora_b(self.q_lora_a(x))
            qkv[:, :, :, :self.dim] += q_lora
        if 'k' in self.mode:
            k_lora = self.k_lora_b(self.k_lora_a(x))
            qkv[:, :, :, self.dim:self.dim * 2] += k_lora
        if 'v' in self.mode:
            v_lora = self.v_lora_b(self.v_lora_a(x))
            qkv[:, :, :, -self.dim:] += v_lora
        
        return qkv

class proj_LoRA(nn.Module):
    def __init__(self, proj: nn.Module, proj_lora_a: nn.Module, proj_lora_b: nn.Module):
        super().__init__()
        self.proj = proj
        self.proj_lora_a = proj_lora_a
        self.proj_lora_b = proj_lora_b

    def forward(self, x):
        o = self.proj(x)
        o += self.proj_lora_b(self.proj_lora_a(x))
        return o

class SAM_LoRA(nn.Module):
    """Applies low-rank adaptation to a SAM's image encoder.

    Args:
        sam: segment anything model, see 'segment_anything' dir
        r: rank of LoRA
        mode: which part of 'attn' to apply LoRA
        pos: which layer to apply LoRA
    """

    def __init__(self, sam: Sam, r: int, mode: str='qv', pos=None):
        super(SAM_LoRA, self).__init__()

        assert r > 0
        self.mode = mode
        
        # assign LoRA layer position (all layers by default)
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(sam.image_encoder.blocks)))
        
        # freeze SAM image encoder and prompt encoder
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False
            
        # create LoRA layers for storage, then we can init them or load weights
        self.w_As = []
        self.w_Bs = []
        
        # apply LoRA to SAM image encoder
        for idx, blk in enumerate(sam.image_encoder.blocks):
            if idx not in self.pos:
                continue
            
            qkv = blk.attn.qkv
            self.dim = qkv.in_features
            
            if 'q' in mode:
                q_lora_a = nn.Linear(self.dim, r, bias=False)
                q_lora_b = nn.Linear(r, self.dim, bias=False)
                self.w_As.append(q_lora_a)
                self.w_Bs.append(q_lora_b)
            else:
                q_lora_a = None
                q_lora_b = None
            if 'k' in mode:
                k_lora_a = nn.Linear(self.dim, r, bias=False)
                k_lora_b = nn.Linear(r, self.dim, bias=False)
                self.w_As.append(k_lora_a)
                self.w_Bs.append(k_lora_b)
            else:
                k_lora_a = None
                k_lora_b = None
            if 'v' in mode:
                v_lora_a = nn.Linear(self.dim, r, bias=False)
                v_lora_b = nn.Linear(r, self.dim, bias=False)
                self.w_As.append(v_lora_a)
                self.w_Bs.append(v_lora_b)
            else:
                v_lora_a = None
                v_lora_b = None
            if 'o' in mode:
                o_lora_a = nn.Linear(self.dim, r, bias=False)
                o_lora_b = nn.Linear(r, self.dim, bias=False)
                self.w_As.append(o_lora_a)
                self.w_Bs.append(o_lora_b)
            else:
                o_lora_a = None
                o_lora_b = None
            
            blk.attn.qkv = qkv_LoRA(
                qkv,
                q_lora_a,
                q_lora_b,
                k_lora_a,
                k_lora_b,
                v_lora_a,
                v_lora_b,
                mode=mode,
            )
            if 'o' in mode:
                blk.attn.proj = proj_LoRA(
                    blk.attn.proj,
                    o_lora_a,
                    o_lora_b,
                )
        
        # init LoRA layer parameters
        self.reset_parameters()
        self.sam = sam

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def save_parameters(self) -> dict:
        r"""save both lora and mask decoder parameters.
        """
        
        # save lora parameters
        num_lora_weight = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_lora_weight)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_lora_weight)}
        
        # save mask decoder parameters
        mask_decoder_tensors = {}
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **mask_decoder_tensors}
        return merged_dict

    def load_parameters(self, state_dict) -> None:
        r"""load both lora and mask decoder parameters.
        """

        # load lora parameters
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = nn.Parameter(saved_tensor)
        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = nn.Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load mask decoder parameters
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        
        self.sam.load_state_dict(sam_dict)

    def forward(self, data):
        img, box = data['img'], data['box']
        
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )
        
        # LoRA image encoder
        input_image = self.sam.preprocess(img) # (1, 3, 1024, 1024)
        image_embedding = self.sam.image_encoder(input_image) # (1, 256, 64, 64)
        
        # predicted masks
        mask_predictions, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=True,
        )
        
        return mask_predictions
