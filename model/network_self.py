import os
import torch
import wandb
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchsummary import summary
import sys
from utils import *
import unet3d
from config import models_genesis_config
from get_config_genesis import get_config
import warnings
warnings.filterwarnings('ignore')
from torch import nn

class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, model, conf,config):
        """Initialize configuration."""
        self.conf = conf
        self.config = config
        self.model = model
        self.num_classes = config['num_classes']
        self.criterion = nn.MSELoss()

        if self.conf.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
        elif self.conf.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.conf.lr)
        else:
            raise

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.conf.patience * 0.8), gamma=0.5)
        self.intial_epoch = 0 
        if self.conf.weights != None:
            self.checkpoint=torch.load(self.conf.weights)
            self.model.load_state_dict(self.checkpoint['state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            self.intial_epoch=self.checkpoint['epoch']
            print("Loading weights from ",self.conf.weights)

        if self.config['use_cuda']: 
            self.model.cuda()
        # Recording the training losses and validation performance.
        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []

        # init auxiliary stuff such as log_func
        self._init_aux()

    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory wghere we save states such as trained model.
        if self.config['use_wandb']:
            self.log_dir = os.path.join(self.config['log_dir'], wandb.run.name)
        else:
            self.log_dir = self.config['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File that we save trained model into.
        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")

        # We save model that achieves the best performance: early stopping strategy.
        self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")
 
    
    def _save(self, pt_file):
        """Saving trained model."""

        # Save the trained model.
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            pt_file,
        )

    def _restore(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        self.model.load_state_dict(load_res["model"])
        # Loading optimizer.
        self.optimizer.load_state_dict(load_res["optimizer"])

    
    def _get_loss(self, pred, target):
        """ Compute loss function based on configs """
        if self.loss_type == 'cross_entropy':
            # BxC logits into Bx1 ground truth
            loss = F.cross_entropy(pred, target.float())
        elif self.loss_type == 'evidential':
            alpha = F.softplus(pred)+1
            target_oh = F.one_hot(target, self.num_classes)
            mse, kl = evidential_mse(alpha, target_oh, alpha.device)
            loss = torch.mean(mse + 0.1*kl)
        elif self.loss_type == 'laplace_cdf':
            pred_categorical = laplace_cdf(F.sigmoid(pred), self.num_classes, pred.device)
            target_oh= 0.9*F.one_hot(target.long(), self.num_classes) + 0.1/self.num_classes
            loss = F.binary_cross_entropy(pred_categorical, target_oh.squeeze(1))
            #print(loss)
        else:
            raise NotImplementedError
        return loss
    
    # obtain summary statistics of
    # argmax, max_percentage, entropy, evid.uncertainty for each function
    # expects logits BxC for classification, Bx2 for cdf
    def _get_prediction_stats(self, logits, nclasses):
        # convert logits to probabilities
        if self.loss_type == 'cross_entropy':
            prob = F.softmax(logits, dim=1)
            vacuity = -1
        elif self.loss_type == 'evidential':
            prob, vacuity = evidential_prob_vacuity(logits, nclasses)
        elif self.loss_type == 'laplace_cdf':
            prob = laplace_cdf(F.sigmoid(logits), nclasses, logits.device)
            vacuity = -1
        else:
            raise NotImplementedError
        max_percentage, argm = torch.max(prob, dim=1)
        entropy = torch.sum(-prob*torch.log(prob), dim=1)
        if nclasses!=2:
            uni = utils.test_unimodality(prob.cpu().numpy())
        else:
            uni = torch.ones(logits.size())
        return argm, max_percentage, entropy, vacuity, uni
            
    def train(self, loader_tr, loader_va):
        """Training pipeline."""
        # Switch model into train mode.
        self.model.train()
        best_loss = 10000 # Record the best validation metrics.
        num_epoch_no_improvement = 0
  
        for epoch in range(self.intial_epoch,self.conf.nb_epoch):
            self.scheduler.step(epoch)
            losses = []
            for [cine,cine_d,target1,target2] in tqdm(loader_tr):
                # Transfer data from CPU to GPU.
                if self.config['use_cuda']:
                    cine = cine.cuda()
                    cine_d = cine_d.cuda()
                    target1 = target1.cuda()
                    target2 = target2.cuda()
                    
                batch_size = len(cine)
        
                # get the logit output
                pred = self.model(cine_d.float())            
         
                # Calculate Loss
                loss = self.criterion(pred,cine)
                losses += [loss]
                #print(loss)
                


                # Calculate the gradient.
                loss.backward()
                # Update the parameters according to the gradient.
                self.optimizer.step()
                # Zero the parameter gradients in the optimizer
                self.optimizer.zero_grad()

            loss_avg = torch.mean(torch.stack(losses)).item()
            
            print("Epoch: %3d, loss_avg: %.5f" % (epoch, loss_avg))
            
            val_loss_avg= self.test(loader_va, mode="valid")
            
            
            # Save logs every epoch.
            if config['use_wandb']:
                wandb.log({"epoch":epoch ,"tr_loss":loss_avg , "val_loss_as":val_loss_avg})
                

            print("Epoch: %3d, val_loss_avg: %.5f" % (epoch , val_loss_avg))
            
            if val_loss_avg < best_loss:
                print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, val_loss_avg))
                best_loss = val_loss_avg
                num_epoch_no_improvement = 0
                #save model
                torch.save({
                    'epoch': epoch+1,
                    'state_dict' : self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },os.path.join(self.conf.model_path, "AS_best.pt"))
                print("Saving model ",os.path.join(self.conf.model_path,"AS_best.pt"))
            else:
                print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
                num_epoch_no_improvement += 1
            if num_epoch_no_improvement == conf.patience:
                print("Early Stopping")
                break

            # Recording training losses and validation performance.
            self.train_losses += [loss_avg]
            self.idx_steps += [epoch]

    @torch.no_grad()
    def test(self, loader_te, mode="test"):
        """Estimating the performance of model on the given dataset."""
        # Choose which model to evaluate.
        if mode == "test":
            self._restore(self.bestmodel_file)
        # Switch the model into eval mode.
        self.model.eval()
        accs = []
        losses = []
        num_samples = 0
        

        for [cine,cine_d,target1,target2] in tqdm(loader_te):
            if self.config['use_cuda']:
                cine = cine.cuda()
                cine_d = cine_d.cuda()
                target1 = target1.cuda()
                target2 = target2.cuda()

            # get the logit output

            pred = self.model(cine_d.float())            

            # Calculate Loss
            loss = self.criterion(pred,cine)
            losses += [loss]
                

            
        loss_avg = torch.mean(torch.stack(losses)).item()

        
        # Switch the model into training mode
        self.model.train()
        return loss_avg



if __name__ == "__main__":
    """Main for mock testing."""
    from get_config_genesis import get_config
    from data_loader import get_as_dataloader
    import os
    import unet3d
    from config import models_genesis_config
    os.environ["CUDA_VISIBLE_DEVICES"] = "7,1"

    conf = models_genesis_config()
    config = get_config()
    
    if config['use_wandb']:
        run = wandb.init(project="Self-supervised", entity="asproject", config=config)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = unet3d.UNet3D()
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)
    net = Network(model, conf, config)
    dataset_tr,dataloader_tr = get_as_dataloader(config, mode='train')
    dataset_va,dataloader_va = get_as_dataloader(config, mode='val')
    dataset_te,dataloader_te = get_as_dataloader(config, mode='test')
    
    if config['mode']=="train":
        net.train(dataloader_tr, dataloader_va)
        net.test(dataloader_te, mode="test")
    if config['mode']=="test":
        net.test(dataloader_te, mode="test")
    if config['use_wandb']:
        wandb.finish()