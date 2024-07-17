# setup environment
import argparse
from datetime import datetime
import os
from model import *
join = os.path.join
import random
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import numpy as np
import monai

import matplotlib.pyplot as plt
from tqdm import tqdm

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.dataset import TaskDataset, GeneralDataset
from utils.datainfo import modal_dict
from utils.SurfaceDice import compute_dice_coefficients

# set seeds
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.empty_cache()
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# set up parser
parser = argparse.ArgumentParser("UniversalMedSAM evaluating", add_help=False)
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--task_name", type=str, default="SAM_1024")
parser.add_argument("--method", type=str, default="lora")
parser.add_argument("--lora_rank", type=int, default=32)
parser.add_argument("--bottleneck_dim", type=int, default=32)
parser.add_argument("--checkpoint", type=str, default="/home/yjx/weaksup-sam/checkpoint/sam_vit_b_01ec64.pth")
parser.add_argument("--work_dir", type=str, default="/home/yjx_sam/weaksup-sam/work_dir")
parser.add_argument("--data_path", type=str, default="/data/sam/baselines/nnUNet/yjx/weaksup/KiTS19-npy/test_KiTS19")
parser.add_argument("--metric", type=str, default=["dsc"], nargs='+',
                    help="evaluation metrics (e.g dsc, nsd)")
parser.add_argument("--data_dim", type=str, default="2D&3D")
parser.add_argument("--device", type=str, default="cuda:4")
parser.add_argument("--device_ids", type=int, default=[4, 5, 6, 7], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
# test
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=32)
parser.add_argument("--resume", type=str,
                    default="/home/yjx_sam/merge_sam/work_dir/samlora-KiPA22-20-32/universal_sam_model_lowest.pth",
                    help="Resuming training from checkpoint")
parser.add_argument("--use_amp", action="store_true", default=False,
                    help="Use amp")
args = parser.parse_args()

device = torch.device(args.device)


def main():
    sam_model = sam_model_registry[args.model_type](image_size=256, keep_resolution=True, checkpoint=args.checkpoint)
    if args.method == "medsamv1":                        
        universal_sam_model = MedSAM(sam_model).to(device)
    elif args.method == "medsamv2":
        universal_sam_model = MedSAMV2(sam_model).to(device)
    elif args.method == "lora":
        universal_sam_model = SAM_LoRA(sam_model,r=args.lora_rank).to(device)
    elif args.method == "adapter":                  
        universal_sam_model = SAM_Adapter(sam_model, args.lora_rank).to(device)
    universal_sam_model = nn.DataParallel(universal_sam_model, device_ids=args.device_ids)

    work_dir = args.work_dir
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            print(f"load model from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            universal_sam_model.module.load_parameters(checkpoint["model"])
        work_dir = os.path.dirname(args.resume)

    if args.task_name == 'SAM_1024' or args.task_name == 'SAM_256':
        work_dir = join(work_dir, args.task_name)
        os.makedirs(work_dir, exist_ok=True)

    img_size = universal_sam_model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    evaluation = args.data_path.split('/')[-1]
    if "dsc" in args.metric:
        dscs_data = []
    print(f"save evaluation result to {join(work_dir, '{}_{}.md'.format(evaluation, args.num_epochs))}")
    with open(join(work_dir, '{}_{}.md'.format(evaluation, args.num_epochs)), mode="w") as f:
        f.write(f"# {evaluation} evaluation\n\n")
        task_path = args.data_path
        test_dataset = TaskDataset(task_path, train=False)
        test_dataloader = DataLoader(
                        test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
        dscs = []
        universal_sam_model.eval()
        pbar_test = tqdm(test_dataloader)
        with torch.no_grad():
            for data, label in pbar_test:
                if data["img"].shape[-1] != img_size:
                    data["box"] = box_transform.apply_boxes_torch(data["box"],
                                                                              tuple(data["img"].shape[-2:])).reshape(-1, 4)
                    data["img"] = img_transform(data["img"])
            data["img"] = data["img"].to(device, non_blocking=True)
            data["box"] = data["box"].to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            mask_pred = universal_sam_model(data)
            if mask_pred.shape[-1] != label.shape[-1]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-1], mode="bilinear", antialias=True)
            mask_prob = torch.sigmoid(mask_pred)
            mask = (mask_prob > 0.5).long()
            dsc_list = []

            for i in range(universal_sam_model.module.sam.mask_decoder.num_multimask_outputs):
                dsc_list.append(compute_dice_coefficients(label, mask[:, i].unsqueeze(1)))
                print(dsc_list)
            dsc = torch.stack(dsc_list, dim=0).max(dim=0)[0]
            print(torch.mean(dsc).item())
            dscs.append(dsc)
            dscs_data.append(dsc)
            pbar_test.set_postfix({"dsc": dsc.mean().item()})
        dsc = torch.cat(dscs).mean().item()
        f.write(f"  - {'zhuiti'}: DSC ({dsc:.4f})\n")
        dsc = torch.cat(dscs_data).mean().item()
        f.write(f"- ALL\n")
        f.write(f"  - Mean: DSC ({dsc:.4f})\n")



if __name__ == "__main__":
    main()
# max_dice = torch.max(dsc_list[-1]).item()
                # f.write(f"Sample {i+1}: Max Dice = {max_dice:.4f}")