import torchvision
import math
import os
import time
from echonet_dataset import Echo
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import os
import shutil
import sys
import pickle
from model.conunet import SupConUnet
from model.Resnetunet import ResNetUNet
import tqdm
import echonet_dataloader
from echonet_dataloader import utils
from config import models_genesis_config
from get_config_genesis import get_config
from config import models_genesis_config
#from datasets.two_dim.NumpyDataLoader import NumpyDataSet
apex_support = False
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
torch.manual_seed(0)



# Generate model based on whether we want to do segmentation, reconstruction or contrastive learning
def SupConSegModel(num_classes, in_channels=1, initial_filter_size=64,
                 kernel_size=3, do_instancenorm=True, mode="cls"):

    encoder = torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=False)
    initial_filter_size = encoder.classifier[-1].in_channels
    if mode == 'mlp':
        encoder.classifier[-1] = nn.Sequential(torch.nn.Conv2d(initial_filter_size, 256, kernel_size=1),
                                  torch.nn.Conv2d(256, num_classes, kernel_size=1))
    elif mode == "segmentation":
        encoder.classifier[-1] = torch.nn.Conv2d(encoder.classifier[-1].in_channels, 1, kernel_size=encoder.classifier[-1].kernel_size)  # change number of outputs to 1
    elif mode == "reconstruction":
        encoder.classifier[-1] = torch.nn.Conv2d(encoder.classifier[-1].in_channels, 3, kernel_size=encoder.classifier[-1].kernel_size)  # change number of outputs to 1

    else:
        raise NotImplemented("This mode is not supported yet")

    return encoder


def cal_dice(large_inter, large_union, small_inter, small_union):
    overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
    large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
    small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
    print("overall_dice = {:.4f},large_dice = {:.4f},small_dice = {:.4f}".format(overall_dice,large_dice,small_dice))
    


def run_epoch_self(model, dataloader, train, optim, device,config):
    """Run one epoch of training/evaluation for segmentation.
    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    n = 0

    model.train(train)

    loss_fn = nn.MSELoss()
    norm = nn.Softmax(dim =1)
    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (ef,large_frame, small_frame, large_trace, small_trace),(deformlarge,deformsmall) in dataloader:

                # Run prediction for diastolic frames and compute loss
                large_frame = large_frame.to(device).float()
                deformlarge = deformlarge.to(device).float()
                #y_large = model(large_frame)["out"]
                y_large = model(deformlarge)["out"]  
                y_small = F.normalize(y_large,dim = 1)/2+0.5
                #y_large = norm(y_large)
                loss_large = loss_fn(y_large,large_frame)

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device).float()
                deformsmall = deformsmall.to(device).float()
                #y_small = model(small_frame)["out"]
                y_small = model(deformsmall)["out"]
                # Comment this if you want
                y_small = F.normalize(y_small,dim = 1)/2+0.5
                #y_small = norm(y_small)
                loss_small = loss_fn(y_small,small_frame)
                
                # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                if n%100==0:
                    if config['use_wandb']:
                        if train: 
                            wandb.log({"loss":loss})

                # Accumulate losses and compute baselines
                total += loss.item()
                n = n+1

                # Show info on process bar
                pbar.set_postfix_str("{:.4f},{:.4f}".format(loss.item(),total/n))
                pbar.update()


    return total/n


if __name__ == "__main__":

    import torch.distributed as dist
    from loss_functions.nt_xent import NTXentLoss
    import os
    import shutil
    import sys
    import pickle
    from model.conunet import SupConUnet
    from model.Resnetunet import ResNetUNet
    import tqdm
    import os
    import echonet_dataloader
    from echonet_dataloader import utils
    from config import models_genesis_config
    from get_config_genesis import get_config
    import os
    from config import models_genesis_config
    #from datasets.two_dim.NumpyDataLoader import NumpyDataSet
    apex_support = False
    import numpy as np
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from runepoch import run_epoch
    # GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"

    # Set device for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get config
    conf = models_genesis_config()
    config = get_config()
    
    if config['use_wandb']:
        run = wandb.init(project="533R", entity="neda77aa", config=config)
    

    # Preparing echinet dataset
    tasks = ['EF','SmallFrame' , 'LargeFrame', 'SmallTrace' ,'LargeTrace']
    kwargs = {"target_type": tasks,
              "mean": 0,
              "std": 1,
              "mode":"self_supervised",
              "conf":conf}
    traindataset = Echo(root='/AS_Neda/echonet/', split="train",**kwargs)
    # Generating dataloader
    dataloader = torch.utils.data.DataLoader(
        traindataset, batch_size=16, num_workers=4, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

    valdataset = Echo(root='/AS_Neda/echonet/', split="val",**kwargs)
    valloader = torch.utils.data.DataLoader(
        valdataset, batch_size=16, num_workers=4, shuffle=True, pin_memory=(device.type == "cuda"))

    if config['train_mode'] == "reconstruction":
        # Preparing Segmentation model 
        model = SupConSegModel(num_classes=3, mode="reconstruction")
        if device.type == "cuda":
            model = torch.nn.DataParallel(model)
        model.to(device)

        # Set up optimizer
        lr=1e-5
        weight_decay=1e-4
        lr_step_period=None,
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        if lr_step_period is None:
            lr_step_period = math.inf
        #scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
        best_loss = 1000

        ## Self Supervised learning
        for epoch in range(150):
            print("Epoch #{}".format(epoch), flush=True)
            start_time = time.time()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

            loss = run_epoch_self(model, dataloader, True, optim, device,config)
            valloss = run_epoch_self(model, valloader, False, optim, device,config)
            print("loss:{:.4f},val_loss:{:.4f}".format(loss,valloss))
            if valloss<best_loss:
                best_loss = valloss
            #scheduler.step()
        # Save checkpoint
        save = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'loss': loss,
            'opt_dict': optim.state_dict(),
        }
        torch.save(save, os.path.join("pretrained_weights/self", "checkpoint.pt"))

    elif config['train_mode'] == "segmentation":
    # Finetuning for segmenation
    
        # Preparing Segmentation model 
        modelseg = SupConSegModel(num_classes=1, mode="segmentation")
        # Load best weights
        if config['pretrained']:
            checkpoint = torch.load(os.path.join("pretrained_weights/self", "checkpoint.pt"))
            modelseg.load_state_dict(checkpoint['state_dict'], strict=False)

        if device.type == "cuda":
            modelseg = torch.nn.DataParallel(modelseg)
        modelseg.to(device)
        
         # Set up optimizer
        lr=1e-5
        weight_decay=1e-5
        lr_step_period=None,
        optim = torch.optim.SGD(modelseg.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        if lr_step_period is None:
            lr_step_period = math.inf
        #scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
        best_loss = 1000

        # Finetuning for segmenation
        for epoch in range(20):
            print("Epoch #{}".format(epoch), flush=True)
            start_time = time.time()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

            loss, large_inter, large_union, small_inter, small_union = run_epoch(modelseg, dataloader, True, optim, device,config)
            cal_dice(large_inter, large_union, small_inter, small_union)

            valloss, val_large_inter, val_large_union, val_small_inter, val_small_union = run_epoch(modelseg, valloader, False, optim, device,config)
            cal_dice(val_large_inter, val_large_union, val_small_inter, val_small_union)


    if config['use_wandb']:
        wandb.finish()