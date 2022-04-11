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
from config import deformation_config
from get_config_genesis import get_config
from config import deformation_config
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
    return overall_dice
    


def run_epoch_self(model, dataloader, train, optim, device,config,num_epoch):
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
                # print("deformlarge",deformlarge.shape)
                y_large = model(deformlarge)  
                
      
                # y_small = F.normalize(y_large,dim = 1)/2+0.5
                #y_large = norm(y_large)
                loss_large = loss_fn(y_large,large_frame)

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device).float()
                deformsmall = deformsmall.to(device).float()
                #y_small = model(small_frame)["out"]
                y_small = model(deformsmall)
                # Comment this if you want
                # y_small = F.normalize(y_small,dim = 1)/2+0.5
                #y_small = norm(y_small)
                loss_small = loss_fn(y_small,small_frame)
                
                # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()


                # Accumulate losses and compute baselines
                total += loss.item()
                n = n+1

                # Show info on process bar
                pbar.set_postfix_str("{:.4f},{:.4f}".format(loss.item(),total/n))
                pbar.update()


    return total/n






def run_epoch(model, dataloader, train, optim, device,config):
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

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    model.train(train)

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (ef,large_frame, small_frame, large_trace, small_trace),_ in dataloader:
                # Count number of pixels in/out of human segmentation
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                # Count number of pixels in/out of computer segmentation
                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # Run prediction for diastolic frames and compute loss
                large_frame = large_frame.to(device).float()
                large_trace = large_trace.to(device).float()
                y_large = model(large_frame)
                #y_large = model(large_frame)
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device).float()
                small_trace = small_trace.to(device).float()
                y_small = model(small_frame)
                #y_small = model(small_frame)
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss.item()
                n += large_trace.size(0)
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                # Show info on process bar
                pbar.set_postfix_str("average loos:{:.4f} last loss: ({:.4f}) / entropy:{:.4f} {:.4f}, Dice:{:.4f}, {:.4f}".format(total / n / 112 / 112, loss.item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                pbar.update()
   
    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)
    overall_dice = 2 * (large_inter_list.sum() + small_inter_list.sum()) / (large_union_list.sum() + large_inter_list.sum() + small_union_list.sum() + small_inter_list.sum())
    large_dice = 2 * large_inter_list.sum() / (large_union_list.sum() + large_inter_list.sum())
    small_dice = 2 * small_inter_list.sum() / (small_union_list.sum() + small_inter_list.sum())
    
    return total / n / 112 / 112, overall_dice



def build_optimizer(model, optimizer, learning_rate):
    if optimizer == "sgd":
        weight_decay=learning_rate/10
        lr_step_period=None,
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        # if lr_step_period is None:
        #     lr_step_period = math.inf
        #scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
        
        
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                               lr=learning_rate)
        
    return optimizer #,scheduler

def build_dataset(batch_size,device,conf,split='train'):
    tasks = ['EF','SmallFrame' , 'LargeFrame', 'SmallTrace' ,'LargeTrace']
    kwargs = {"target_type": tasks,
              "mean": 0.,
              "std": 1.,
              "mode":"self_supervised",
              "conf":conf,
              "channels":1,
              "padding":8}
    dataset = Echo(root='/AS_Neda/echonet/', split="train",**kwargs)
    # Generating dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=(split=='train'), pin_memory=(device.type == "cuda"), drop_last=(split=='train'))

    return dataloader

def train_self(config=None):
    
    # Get config
    conf = deformation_config()
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        print(config)
        conf.local_rate = config.local_rate
        conf.nonlinear_rate = config.nonlinear_rate
        conf.paint_rate = config.paint_rate
        conf.display()
        # Set device for computations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        # Preparing echonet dataset
        trainloader = build_dataset(config.batch_size,device,conf,split='train')
        valloader = build_dataset(config.batch_size,device,conf,split='val')
        
        # Preparing Segmentation model 
        model = ResNetUNet(1)
        # print(model)
        if device.type == "cuda":
            model = torch.nn.DataParallel(model)
        model.to(device)

        # Set up optimizer
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            print("Epoch #{}".format(epoch), flush=True)
            loss = run_epoch_self(model, trainloader, True, optimizer, device,config,epoch)
            valloss = run_epoch_self(model, valloader, False, optimizer, device,config,epoch)
            wandb.log({"epoch": epoch, "loss": loss, "val_loss":valloss})  
            
            
            


def train(config=None):
    
    # Get config
    conf = deformation_config()
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        print(config)

        # Set device for computations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        # Preparing echonet dataset
        trainloader = build_dataset(config.batch_size,device,conf,split='train')
        valloader = build_dataset(config.batch_size,device,conf,split='val')
        
        # Preparing Segmentation model 
        modelseg = ResNetUNet(1)
        # Load best weights
        pretrained = False
        if pretrained:
            checkpoint = torch.load(os.path.join("pretrained_weights/self", "checkpoint_res.pt"))
            modelseg.load_state_dict(checkpoint['state_dict'], strict=False)

        if device.type == "cuda":
            modelseg = torch.nn.DataParallel(modelseg)
        modelseg.to(device)

        # Set up optimizer
        optimizer = build_optimizer(modelseg, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            print("Epoch #{}".format(epoch), flush=True)
            loss, overall_dice  = run_epoch(modelseg, trainloader, True, optimizer, device,config)
            valloss, overall_dice_val = run_epoch(modelseg, valloader, False, optimizer, device,config)

            wandb.log({"epoch": epoch, "loss": loss, "val_loss":valloss, "dice" :overall_dice, "val_dice":overall_dice_val}) 
            
            




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
    from config import deformation_config
    from get_config_genesis import get_config
    import os
    from config import deformation_config
    import torchvision.utils as vutils
    #from datasets.two_dim.NumpyDataLoader import NumpyDataSet
    apex_support = False
    import numpy as np
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    # GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "7,5"


    # Get config
    conf = deformation_config()
    #conf.display()
    config = get_config()
    

    
    if config['train_mode'] == "reconstruction":
        sweep_config = {'method':'bayes'}
        metric = {'name':'loss', 'goal':'minimize'}
        sweep_config['metric'] = metric
        parameters_dict = { 'optimizer':{
                            'values':['sgd']  #'values':['adam','sgd']
                            },
                            'learning_rate' :{
                             'values':[1e-5,1e-4,1e-3,0.005,0.0005]
                            },
                           'batch_size':{
                            'values':[16]
                            },
                            "epochs" : {
                              "values" : [10]
                            },
                            "local_rate" : {
                                "values" : [0,0.5]
                            },
                            "nonlinear_rate" : {
                               "values" : [0, 0.4]
                            },
                            "paint_rate" : {
                               "values" : [0.8]
                            },
                          }
        
        sweep_config['parameters'] = parameters_dict 
        sweep_id = wandb.sweep(sweep_config,project="sweep", entity="533r")
        wandb.agent(sweep_id, train_self, count=12)


    elif config['train_mode'] == "segmentation":
    # Finetuning for segmenation
        sweep_config = {'method':'random'}
        metric = {'name':'dice', 'goal':'maximize'}
        sweep_config['metric'] = metric
        parameters_dict = { 'optimizer':{
                            'values':['adam','sgd']
                            },
                            'learning_rate' :{
                             'values':[1e-5,1e-4,1e-3,0.005,0.0005,0.00005]
                            },
                           'batch_size':{
                            'values':[16]
                            },
                            "epochs" : {
                              "values" : [10]
                            }
                          }
        
        sweep_config['parameters'] = parameters_dict 
        sweep_id = wandb.sweep(sweep_config,project="sweep_segmentation", entity="533r")
        wandb.agent(sweep_id, train, count=6)
    

    if config['use_wandb']:
        wandb.finish()