import argparse
from datetime import datetime
import os
join = os.path.join
import random
import shutil
import json
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.dataset import GeneralDataset,GeneraltestDataset
from utils.loss import DiceCELoss
import torchvision.transforms.functional as TF
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

parser = argparse.ArgumentParser("SAM training", add_help=False)
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--task_name", type=str, default="samlora-PolyfinV2-10-32")
parser.add_argument("--method", type=str, default="lora")
parser.add_argument("--checkpoint", type=str, default="/home/yjx/weaksup-sam/checkpoint/sam_vit_b_01ec64.pth")
parser.add_argument("--work_dir", type=str, default="/home/yjx/weaksup-sam/work_dir/")
parser.add_argument("--data_path", type=str, default="/data/yjx23/Polyp-supsam/train_Polyp/")
parser.add_argument("--test_data_path", type=str, default="/data/yjx23/Polyp-supsam/test_Polyp")
parser.add_argument("--device", type=str, default="cuda:3")
parser.add_argument("--device_ids", type=int, default=[ 3,4 ], nargs='+')
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=32)
parser.add_argument("--weight_decay", type=float, default=0.01, 
                    help="weight decay (default: 0.01)")
parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                    help="learning rate (absolute lr default: 0.001)")
parser.add_argument("--resume", type=str,
                    default=None,
                    help="Resuming training from checkpoint")
parser.add_argument("--use_wandb", action="store_true", default=False, 
                    help="use wandb to monitor training")
parser.add_argument("--use_amp", action="store_true", default=False, 
                    help="use amp")
#特定模型参数设置
parser.add_argument("--lora_rank", type=int, default=32)
args = parser.parse_args()

model_save_path = join(args.work_dir, args.task_name)
device = torch.device(args.device)


def main():
    os.makedirs(model_save_path, exist_ok=True)
    with open(join(model_save_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    sam_model = sam_model_registry[args.model_type](image_size=256,keep_resolution = True,checkpoint=args.checkpoint)
    if args.method == "medsamv1":                        
        universal_sam_model = MedSAM(sam_model).to(device)
    elif args.method == "medsamv2":
        universal_sam_model = MedSAMV2(sam_model).to(device)
    elif args.method == "lora":
        universal_sam_model = SAM_LoRA(sam_model,r=args.lora_rank).to(device)
    elif args.method == "adapter":                  
        universal_sam_model = SAM_Adapter(sam_model, args.lora_rank).to(device)
    universal_sam_model = nn.DataParallel(universal_sam_model, device_ids=args.device_ids)
 
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, universal_sam_model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    # criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction='none')

    train_dataset = GeneralDataset(args.data_path,  train=True)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = GeneraltestDataset(args.test_data_path, train=False)

    print("Number of validation samples: ", len(val_dataset))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


    img_size = universal_sam_model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    num_epochs = args.num_epochs
    start_epoch = 0
    best_loss = 1e10
    best_dsc = 0
    best_dsc_10 = 0
    loss_log = []
    dsc_log = []
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            # start_epoch = checkpoint["epoch"] + 1
            universal_sam_model.module.load_parameters(checkpoint["model"])
            # optimizer.load_state_dict(checkpoint["optimizer"])


    for epoch in range(start_epoch, num_epochs):
        # train
        epoch_loss = 0
        step = 0
        universal_sam_model.train()
        pbar_train = tqdm(train_dataloader)
        pbar_train.set_description(f"Epoch [{epoch + 1}/{num_epochs}] Train")
        for data, data1,data2,data3,label in pbar_train:
            optimizer.zero_grad()
    
            #添加M2B
            digit1 = random.choice([0, 1, 2, 3])
            #图像保持不变
            if digit1 == 0:
                if data["img"].shape[-1] != img_size:
                    data["box"] = box_transform.apply_boxes_torch((data["box"].reshape(-1, 2, 2)),
                                                                    data["img"].shape[-2:]).reshape(-1, 4)
                    data["img"] = img_transform(data["img"])
                data["img"] = data["img"].to(device, non_blocking=True)
                data["box"] = data["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                # print(label.shape)
                # print(data["img"])
                # print(data["box"])
                pred1  = universal_sam_model(data)
        
                if pred1.shape[-1] != label.shape[-1]:
                    pred1 = F.interpolate(pred1, size=label.shape[-1], mode="bilinear", antialias=True)
                # pred1 = pred1[:, 0].unsqueeze(1)
            #图像顺时针旋转90度
            elif digit1 == 1:
                if data1["img"].shape[-1] != img_size:
                    data1["box"] = box_transform.apply_boxes_torch((data1["box"].reshape(-1, 2, 2)),
                                                                    data1["img"].shape[-2:]).reshape(-1, 4)
                    data1["img"] = img_transform(data["img"])
                data1["img"] = data1["img"].to(device, non_blocking=True)
                data1["box"] = data1["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                # print(label.shape)
                # print(data1["img"])
                # print(data1["box"])
                pred1  = universal_sam_model(data1)        
                if pred1.shape[-1] != label.shape[-1]:
                    pred1 = F.interpolate(pred1, size=label.shape[-1], mode="bilinear", antialias=True)
                # pred1 = pred1[:, 0].unsqueeze(1)
            #图像顺时针旋转180度
            elif digit1 == 2:
                if data2["img"].shape[-1] != img_size:
                    data2["box"] = box_transform.apply_boxes_torch((data2["box"].reshape(-1, 2, 2)),
                                                                    data2["img"].shape[-2:]).reshape(-1, 4)
                    data2["img"] = img_transform(data["img"])
                data2["img"] = data2["img"].to(device, non_blocking=True)
                data2["box"] = data2["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                # print(label.shape)
                # print(data2["img"])
                # print(data2["box"])
                pred1  = universal_sam_model(data2)
              
                if pred1.shape[-1] != label.shape[-1]:
                    pred1 = F.interpolate(pred1, size=label.shape[-1], mode="bilinear", antialias=True)
                # pred1 = pred1[:, 0].unsqueeze(1)
            #图像顺时针旋转270度
            else:
                if data3["img"].shape[-1] != img_size:
                    data3["box"] = box_transform.apply_boxes_torch((data3["box"].reshape(-1, 2, 2)),
                                                                    data3["img"].shape[-2:]).reshape(-1, 4)
                    data3["img"] = img_transform(data["img"])
                data3["img"] = data3["img"].to(device, non_blocking=True)
                data3["box"] = data3["box"].to(device, non_blocking=True)
                # print(data3["img"])
                # print(data3["box"])
                label = label.to(device, non_blocking=True)
                pred1  = universal_sam_model(data3)
       
                if pred1.shape[-1] != label.shape[-1]:
                    pred1 = F.interpolate(pred1, size=label.shape[-1], mode="bilinear", antialias=True)
                # pred1 = pred1[:, 0].unsqueeze(1)
            # 将pred1逆时针旋转k个90度
            pred1  = torch.rot90(pred1, k=digit1, dims=(2, 3))
            ## pred 2
            digit2 = random.choice([0, 1, 2, 3])
            #图像保持不变
            if digit2 == 0:
                if data["img"].shape[-1] != img_size:
                    data["box"] = box_transform.apply_boxes_torch((data["box"].reshape(-1, 2, 2)),
                                                                    data["img"].shape[-2:]).reshape(-1, 4)
                    data["img"] = img_transform(data["img"])
                data["img"] = data["img"].to(device, non_blocking=True)
                data["box"] = data["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                pred2  = universal_sam_model(data)
                if pred2.shape[-1] != label.shape[-1]:
                    pred2 = F.interpolate(pred2, size=label.shape[-1], mode="bilinear", antialias=True)
                # pred2 = pred2[:, 0].unsqueeze(1)
      
            #图像顺时针旋转90度
            elif digit2 == 1:
                if data1["img"].shape[-1] != img_size:
                    data1["box"] = box_transform.apply_boxes_torch((data1["box"].reshape(-1, 2, 2)),
                                                                    data1["img"].shape[-2:]).reshape(-1, 4)
                    data1["img"] = img_transform(data["img"])
                data1["img"] = data1["img"].to(device, non_blocking=True)
                data1["box"] = data1["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                pred2  = universal_sam_model(data1)
                if pred2.shape[-1] != label.shape[-1]:
                    pred2 = F.interpolate(pred2, size=label.shape[-1], mode="bilinear", antialias=True)
                # pred2 = pred2[:, 0].unsqueeze(1)
      
            #图像顺时针旋转180度
            elif digit2 == 2:
                if data2["img"].shape[-1] != img_size:
                    data2["box"] = box_transform.apply_boxes_torch((data2["box"].reshape(-1, 2, 2)),
                                                                    data2["img"].shape[-2:]).reshape(-1, 4)
                    data2["img"] = img_transform(data["img"])
                data2["img"] = data2["img"].to(device, non_blocking=True)
                data2["box"] = data2["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                pred2  = universal_sam_model(data2)
                if pred2.shape[-1] != label.shape[-1]:
                    pred2 = F.interpolate(pred2, size=label.shape[-1], mode="bilinear", antialias=True)
                # pred2 = pred2[:, 0].unsqueeze(1)
     
            #图像顺时针旋转270度
            else:
                if data3["img"].shape[-1] != img_size:
                    data3["box"] = box_transform.apply_boxes_torch((data3["box"].reshape(-1, 2, 2)),
                                                                    data3["img"].shape[-2:]).reshape(-1, 4)
                    data3["img"] = img_transform(data["img"])
                data3["img"] = data3["img"].to(device, non_blocking=True)
                data3["box"] = data3["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                pred2  = universal_sam_model(data3)
                if pred2.shape[-1] != label.shape[-1]:
                    pred2 = F.interpolate(pred2, size=label.shape[-1], mode="bilinear", antialias=True)
             
                # pred2 = pred2[:, 0].unsqueeze(1)
            # 将pred2逆时针旋转k个90度
            pred2  = torch.rot90(pred2, k=digit2, dims=(2, 3))
            losses = []
            for i in range(universal_sam_model.module.sam.mask_decoder.num_multimask_outputs):
                output1 = pred1[:, i].unsqueeze(1)
                for j in range(universal_sam_model.module.sam.mask_decoder.num_multimask_outputs):
              
                    output2 = pred2[:, j].unsqueeze(1)
                    print(label.shape)
                    ## loss_sc
                    loss_sc        = (torch.sigmoid(output1)-torch.sigmoid(output2)).abs()
                    loss_sc        = loss_sc[label[:,0:1]==1].mean()
                 
                    ## M2B transformation
                    pred          = torch.cat([output1, output2], dim=0)
                    mask          = torch.cat([label, label], dim=0)
                    predW, predH  = pred.max(dim=2, keepdim=True)[0], pred.max(dim=3, keepdim=True)[0]
                    pred          = torch.minimum(predW, predH)
                    pred, mask    = pred[:,0], mask[:,0]
                    mask = mask.to(torch.float32)
                    # print(pred.shape)
                    # print(label.shape)
                    # print(pred.dtype)
                    # print(label.dtype)
                    ## loss_ce + loss_dice 
                    loss_ce        = F.binary_cross_entropy_with_logits(pred, mask)
                    pred           = torch.sigmoid(pred)
                    inter          = (pred*mask).sum(dim=(1,2))
                    union          = (pred+mask).sum(dim=(1,2))
                    loss_dice      = 1-(2*inter/(union+1)).mean()
                    loss           = loss_ce + loss_dice + loss_sc
                    losses.append(loss)

            loss = torch.stack(losses, dim=0).min(dim=0)[0]
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            pbar_train.set_postfix({"loss": loss.item()})


        epoch_loss /= step
        loss_log.append(epoch_loss)



        print(
            f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )

        ## save the latest model
        checkpoint = {
            "model": universal_sam_model.module.save_parameters(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "universal_sam_model_latest.pth"))
        ## save the lowest model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": universal_sam_model.module.save_parameters(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "universal_sam_model_lowest.pth"))





        # validation
        epoch_dsc = 0
        size = 0
        universal_sam_model.eval()
        pbar_val = tqdm(val_dataloader)
        pbar_val.set_description(f"Epoch [{epoch + 1}/{num_epochs}] Val")
        with torch.no_grad():
            for data, label in pbar_val:
                if data["img"].shape[-1] != img_size:
                    data["box"] = box_transform.apply_boxes_torch((data["box"].reshape(-1, 2, 2)),
                                                                  data["img"].shape[-2:]).reshape(-1, 4)
                    data["img"] = img_transform(data["img"])
                data["img"] = data["img"].to(device, non_blocking=True)
                data["box"] = data["box"].to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                mask_pred = universal_sam_model(data)
                if mask_pred.shape[-1] != label.shape[-1]:
                    mask_pred = F.interpolate(mask_pred, size=label.shape[-1], mode="bilinear", antialias=True)
                mask_prob = torch.sigmoid(mask_pred)
                mask = (mask_prob > 0.5).long()

                dsc_ambiguous = []
                for i in range(universal_sam_model.module.sam.mask_decoder.num_multimask_outputs):
                    dsc_ambiguous.append(compute_dice_coefficients(label, mask[:, i].unsqueeze(1)))
                dsc = torch.stack(dsc_ambiguous, dim=0).max(dim=0)[0]

                epoch_dsc += dsc.sum().item()
                size += dsc.shape[0]
                pbar_val.set_postfix({"dsc": dsc.mean().item()})


        epoch_dsc /= size
        dsc_log.append(epoch_dsc)
        print(
            f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, DSC: {epoch_dsc}'
        )

        ## save the best model
        if epoch_dsc > best_dsc:
            best_dsc = epoch_dsc
            checkpoint = {
                "model": universal_sam_model.module.save_parameters(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "universal_sam_model_best.pth"))
        ## save the model
        if epoch_dsc > best_dsc_10:
            best_dsc_10 = epoch_dsc
            best_epoch = (epoch // 10 + 1) * 10
            checkpoint = {
                "model": universal_sam_model.module.save_parameters(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, f"universal_sam_model_{best_epoch}.pth"))
        if (epoch + 1) % 10 == 0:
            best_dsc_10 = 0

        # plot loss
        plt.plot(loss_log)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, "train_loss.png"))
        plt.close()

        # plot dsc
        plt.plot(dsc_log)
        plt.title("Validation DSC")
        plt.xlabel("Epoch")
        plt.ylabel("DSC")
        plt.savefig(join(model_save_path, "val_dsc.png"))
        plt.close()

        with open(join(model_save_path, "result.log"), mode='w') as f:
            for e in range(len(loss_log)):
                f.write(f'Epoch [{e}] - Loss: {loss_log[e]}, DSC: {dsc_log[e]}\n')
     
if __name__ == "__main__":
    main()
